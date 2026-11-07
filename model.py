"""
Our bot: an expected-value player built on Monte Carlo equity against
*modelled* opponent ranges, with per-team opponent statistics.

Each turn it estimates, for every legal option (fold, check/call, and a few
raise sizes):

    EV(call)  = equity * pot_after_call - call_cost
    EV(raise) = P(all fold) * pot
              + P(called) * (equity_vs_continuing_range * final_pot - cost)

The opponent model is what makes this exploitative:
  * ranges: what a team raises, limps or calls with preflop is learned from
    how often it does each, so a tight team's raise means a narrow, strong
    range and we fold more; a loose team's raise means little.
  * aggression honesty: the hands a team shows at showdown after betting set
    the weakest hand we assume when it bets again.
  * fold rates by bet size: a team that folds a lot gets bluffed and
    stolen from; a calling station gets thin value bets and no bluffs.

Pickle-safe: only plain data and a random.Random in attributes.
"""
import random

from pb_cards import evaluate, to_int, category, PAIR
from pb_equity import Range, equity, hand_class
from pb_opponents import (new_stats, freq, record, size_bucket,
                          min_category_when_aggressive)
from pb_preflop import BANDS, EQ
from player_action import DEALER, PlayerAction

STREETS = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
STREETS_LEFT = {"preflop": 3, "flop": 2, "turn": 1, "river": 0}

# Tunable knobs (found by parameter search against a pool of bots; see the
# thread for the method). Model(params={...}) overrides any of them.
DEFAULTS = {
    "pf_realise": 0.9,      # share of preflop equity a call realises
    "post_realise": 0.04,   # realisation lost per street still to come
    "raise_realise": 0.95,  # realisation when a raise is called
    "fe_mult": 0.85,        # scales every learned fold rate (tuned: >1 loses)
    "keep_floor": 0.15,     # continuing range never narrower than this share
    "mix_prob": 0.15,       # chance to pick a near-best option instead
    "call_margin": 0.0,     # extra edge a call must show, as share of its cost
    "prior_scale": 1.0,     # weight of prior pseudo-counts vs observed data
    "open_small": 1.0,      # preflop open sizes, in bb above the current bet
    "open_mid": 1.5,
    "open_big": 2.0,
    "threebet": 3.0,        # preflop re-raise multiple of the current bet
}


class Model:
    def __init__(self, iters: int = 400, seed: int | None = None, params=None):
        self.iters = iters
        self.p = dict(DEFAULTS)
        self.p.update(params or {})
        self.rng = random.Random(seed)
        self.opponents = {}  # team name -> stats dict (pb_opponents)
        self.hand = []
        self.cards = []
        self.player_id = None
        self.debug = False
        self._bb_amt = 20
        self.board = []
        self._reset_hand()

    # ------------------------------------------------------------ hand state
    def _reset_hand(self):
        self.started = False
        self.log = []  # (street, pid, kind, situation)
        self.board_n = 0
        self.street = "preflop"
        self.street_bet = {}
        self.cur_bet = 0
        self.pot = 0
        self.aggr_pot = 1  # pot before the last bet/raise this street
        self.aggr_street = set()  # who bet/raised this street
        self.aggr_post = {}  # pid -> postflop bets/raises this hand
        self.called_post = {}  # pid -> bucket of last postflop call
        self.shown = {}  # pid -> revealed hole cards

    def seed(self, hand, player_id):
        self.hand = hand
        self.cards = [to_int(c) for c in hand]
        self.player_id = player_id
        self._reset_hand()

    def _team(self, gs, pid):
        teams = gs.teams
        name = teams[pid] if teams else None
        return name if name else f"seat{pid}"

    def _stats(self, gs, pid):
        name = self._team(gs, pid)
        if name not in self.opponents:
            self.opponents[name] = new_stats()
        return self.opponents[name]

    def _start(self, gs):
        """Blinds are posted before any action is shown to us."""
        self.started = True
        n = gs.num_players
        if n == 2:
            sb_seat, bb_seat = gs.button, (gs.button + 1) % 2
        else:
            sb_seat, bb_seat = (gs.button + 1) % n, (gs.button + 2) % n
        self.street_bet = {p: 0 for p in range(n)}
        self.street_bet[sb_seat] = gs.small_blind
        self.street_bet[bb_seat] = gs.big_blind
        self.bb_seat = bb_seat
        self.cur_bet = gs.big_blind
        self.pot = gs.small_blind + gs.big_blind
        self.aggr_pot = self.pot

    # ------------------------------------------------------------ ingest
    def _ingest(self, actions, gs):
        if not self.started:
            self._start(gs)
        for a in actions:
            if a.kind == "reveal_card":
                if a.player_id == DEALER:
                    self.board_n += 1
                    if self.board_n in STREETS:
                        self.street = STREETS[self.board_n]
                        self.street_bet = {p: 0 for p in self.street_bet}
                        self.cur_bet = 0
                        self.aggr_pot = max(self.pot, 1)
                        self.aggr_street = set()
                else:
                    self.shown.setdefault(a.player_id, []).append(to_int(a.card))
                continue
            if a.kind not in ("fold", "call", "raise"):
                continue
            p = a.player_id
            owed = self.cur_bet - self.street_bet.get(p, 0)
            situation = self._situation(p, owed)
            if p != self.player_id and situation:
                kind = a.kind if owed > 0 or a.kind != "fold" else "call"
                record(self._stats(gs, p), situation, kind)
            self.log.append((self.street, p, a.kind, situation))
            if a.kind == "call":
                self.pot += max(owed, 0)
                self.street_bet[p] = self.cur_bet
                if self.street != "preflop" and owed > 0:
                    self.called_post[p] = situation
            elif a.kind == "raise":
                self.aggr_pot = max(self.pot + max(owed, 0), 1)
                self.pot += a.amount - self.street_bet.get(p, 0)
                self.street_bet[p] = a.amount
                self.cur_bet = a.amount
                self.aggr_street.add(p)
                if self.street != "preflop":
                    self.aggr_post[p] = self.aggr_post.get(p, 0) + 1

    def _situation(self, p, owed):
        if self.street == "preflop":
            raised = self.cur_bet > self._bb()
            if raised:
                if p in self.aggr_street:
                    return "vs_3bet"
                return "vs_raise_big" if owed > 2.5 * self.aggr_pot else "vs_raise"
            if p == self.bb_seat:
                return "vs_limp"
            return "open"
        if owed <= 0:
            return "post_open"
        if p in self.aggr_street:
            return "vs_raise_post"
        if p in self.called_post:
            return "vs_bet_again"
        return size_bucket(owed / max(self.aggr_pot, 1))

    def _bb(self):
        return self._bb_amt

    # ------------------------------------------------------------ ranges
    def _range(self, gs, p):
        st = self._stats(gs, p)
        lo, hi = 0.0, 1.0
        for street, who, kind, sit in self.log:
            if who != p or street != "preflop" or not sit:
                continue
            fr = freq(st, sit, "raise", self.p["prior_scale"])
            fc = freq(st, sit, "call", self.p["prior_scale"])
            if kind == "raise":
                a, b = 0.0, fr
            elif kind == "call":
                a, b = fr, min(1.0, fr + fc)
            else:
                continue
            lo, hi = lo + (hi - lo) * a, lo + (hi - lo) * b
        min_cat = 0
        if self.street != "preflop":
            times = self.aggr_post.get(p, 0)
            if p in self.aggr_street or times:
                min_cat = min_category_when_aggressive(st, times)
                if p not in self.aggr_street:
                    min_cat = min(min_cat, PAIR)
            elif p in self.called_post:
                sit = self.called_post[p]
                if freq(st, sit, "fold", self.p["prior_scale"]) >= 0.45:
                    min_cat = PAIR
        return Range(lo, hi, min_cat)

    def _fold_rate(self, gs, p, ratio):
        return min(0.98, self.p["fe_mult"] * self._fold_freq(gs, p, ratio))

    def _fold_freq(self, gs, p, ratio):
        st = self._stats(gs, p)
        ps = self.p["prior_scale"]
        if self.street == "preflop":
            if p in self.aggr_street:
                return freq(st, "vs_3bet", "fold", ps)
            return freq(st, "vs_raise_big" if ratio > 2.5 else "vs_raise", "fold", ps)
        if p in self.aggr_street:
            return freq(st, "vs_raise_post", "fold", ps)
        if p in self.called_post:
            return freq(st, "vs_bet_again", "fold", ps)
        return freq(st, size_bucket(ratio), "fold", ps)

    def _equity(self, ranges, iters):
        if (self.street == "preflop" and len(ranges) == 1
                and ranges[0].lo == 0.0 and ranges[0].min_cat == 0):
            return _table_equity(hand_class(*self.cards), ranges[0].hi)
        board = self.board
        return equity(self.cards, board, ranges, iters, self.rng)

    # ------------------------------------------------------------ decide
    def run(self, actions, game_state):
        gs = game_state
        try:
            self._bb_amt = gs.big_blind
            self._ingest(actions, gs)
            return self._decide(gs)
        except Exception as error:  # never fault: check or fold instead
            if self.debug:
                raise
            if gs.call_amount == 0:
                return PlayerAction.call(self.player_id)
            return PlayerAction.fold(self.player_id)

    def _decide(self, gs):
        me = self.player_id
        self.board = [to_int(c) for c in gs.board]
        opps = [p for p in gs.players_in_hand if p != me]
        ranges = [self._range(gs, p) for p in opps]
        n = len(opps)
        iters = self.iters if n == 1 else max(150, self.iters // n)
        eq = self._equity(ranges, iters)

        pot, owe = gs.pot, gs.call_amount
        my_stack = gs.stacks[me]
        opp_all_in = all(gs.status(p) == "all_in" for p in opps)
        P = self.p
        realise = 1.0 - P["post_realise"] * STREETS_LEFT[self.street] - 0.03 * (n - 1)
        if self.street == "preflop":
            realise = P["pf_realise"]

        options = {}
        if owe == 0:
            options[("call", 0)] = eq * realise * pot
        else:
            cost = min(owe, my_stack)
            r = 1.0 if (cost >= my_stack or opp_all_in) else realise
            options[("fold", 0)] = 0.0
            options[("call", 0)] = eq * r * (pot + cost) - cost * (1 + P["call_margin"])

        if gs.can_raise and not opp_all_in:
            cache = {}
            for total in self._sizes(gs):
                extra = total - gs.bets[me]
                fold_all, cont = 1.0, []
                for p, rg in zip(opps, ranges):
                    p_call = min(total, gs.bets[p] + gs.stacks[p]) - gs.bets[p]
                    f = self._fold_rate(gs, p, p_call / max(pot + owe, 1))
                    fold_all *= f
                    keep = max(P["keep_floor"], 1.0 - f)
                    cont.append(Range(rg.lo, rg.lo + (rg.hi - rg.lo) * keep, rg.min_cat))
                key = tuple(round(c.hi, 2) for c in cont)
                if key not in cache:
                    cache[key] = self._equity(cont, max(150, iters // 2))
                eq_c = cache[key]
                callers_add = sum(min(total, gs.bets[p] + gs.stacks[p]) - gs.bets[p] for p in opps[:1])
                r = 1.0 if total >= gs.max_raise_to else (P["raise_realise"] if self.street != "river" else 1.0)
                ev = fold_all * pot + (1 - fold_all) * (eq_c * r * (pot + extra + callers_add) - extra)
                options[("raise", total)] = ev

        best = max(options, key=options.get)
        near = [k for k, v in options.items()
                if k != best and v >= options[best] - 0.01 * max(pot, 1) and k[0] != "fold"]
        choice = best
        if near and self.rng.random() < P["mix_prob"]:
            choice = self.rng.choice(near)
        if self.debug:
            print(f"{self.street} {hand_class(*self.cards)} eq={eq:.2f} ranges={ranges} "
                  + " ".join(f"{k[0]}{k[1] or ''}={v:.0f}" for k, v in options.items())
                  + f" -> {choice}")
        kind, total = choice
        if kind == "raise":
            return PlayerAction.raise_to(me, total)
        if kind == "fold":
            return PlayerAction.fold(me)
        return PlayerAction.call(me)

    def _sizes(self, gs):
        lo, hi = gs.raise_range
        bb, pot, cur = gs.big_blind, gs.pot, gs.current_bet
        if self.street == "preflop":
            if cur <= bb:
                cands = [cur + int(self.p[k] * bb) for k in ("open_small", "open_mid", "open_big")]
            else:
                cands = [int(cur * self.p["threebet"])]
        else:
            owe = gs.call_amount
            cands = [cur + (pot + owe) * f for f in (0.33, 0.66, 1.0)]
        cands.append(hi)
        return sorted({int(min(hi, max(lo, c))) for c in cands})

    # ------------------------------------------------------------ learning
    def end_game(self, actions, game_state):
        gs = game_state
        try:
            self._bb_amt = gs.big_blind
            self._ingest(actions, gs)
            board = [to_int(c) for c in gs.board]
            for p in range(gs.num_players):
                if p == self.player_id:
                    continue
                st = self._stats(gs, p)
                st["hands"] += 1
                hole = self.shown.get(p)
                if hole and len(hole) == 2 and len(board) == 5 and self.aggr_post.get(p):
                    st["aggr_shown"].append(category(evaluate(hole + board)))
                    del st["aggr_shown"][:-60]
        except Exception:
            if self.debug:
                raise



def _table_equity(cls: str, hi: float) -> float:
    """Heads-up preflop equity vs the top `hi` of hands, interpolated."""
    row = EQ[cls]
    if hi >= BANDS[0]:
        return row[0]
    for i in range(1, len(BANDS)):
        if hi >= BANDS[i]:
            t = (hi - BANDS[i]) / (BANDS[i - 1] - BANDS[i])
            return row[i] + t * (row[i - 1] - row[i])
    return row[-1]
