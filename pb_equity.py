"""Monte Carlo equity against opponent ranges.

A range is a band of starting hands by preflop strength, (lo, hi) as
fractions of all 1326 combos (0 = AA, 1 = worst), plus an optional minimum
made-hand category on the board (e.g. "they bet, so at least a pair").
"""
import random

from pb_cards import RANKS, evaluate, PAIR
from pb_preflop import ORDER

_rnd = random.Random()


def hand_class(a: int, b: int) -> str:
    r1, r2 = a >> 2, b >> 2
    if r1 < r2:
        r1, r2 = r2, r1
    if r1 == r2:
        return RANKS[r1] * 2
    return RANKS[r1] + RANKS[r2] + ("s" if (a & 3) == (b & 3) else "o")


def _build_combos():
    combos = []
    for a in range(52):
        for b in range(a):
            combos.append((ORDER[hand_class(a, b)], a, b))
    combos.sort()
    return [(a, b) for _, a, b in combos]


COMBOS = _build_combos()  # all 1326 hole-card pairs, strongest first
N_COMBOS = len(COMBOS)


def percentile(a: int, b: int) -> float:
    """Fraction of combos at least as strong as this hand (AA ~ 0.005)."""
    return ORDER[hand_class(a, b)]


class Range:
    __slots__ = ("lo", "hi", "min_cat")

    def __init__(self, lo=0.0, hi=1.0, min_cat=0):
        hi = min(1.0, max(hi, 0.02))
        lo = max(0.0, min(lo, hi - 0.02))
        self.lo, self.hi, self.min_cat = lo, hi, min_cat

    def __repr__(self):
        return f"Range({self.lo:.2f}-{self.hi:.2f}, cat>={self.min_cat})"


def _sample(rng: Range, dead: set, board: list, rnd, tries=40):
    lo, hi = int(rng.lo * N_COMBOS), max(int(rng.hi * N_COMBOS), int(rng.lo * N_COMBOS) + 1)
    fallback = None
    for _ in range(tries):
        a, b = COMBOS[rnd.randrange(lo, hi)]
        if a in dead or b in dead:
            continue
        if rng.min_cat and evaluate([a, b] + board) < rng.min_cat * 16 ** 5:
            fallback = fallback or (a, b)
            continue
        return a, b
    if fallback:
        return fallback
    while True:  # range impossible with these dead cards: any hand
        a, b = COMBOS[rnd.randrange(N_COMBOS)]
        if a not in dead and b not in dead:
            return a, b


def equity(hole, board, ranges, iters=400, rnd=None) -> float:
    """Our share of the pot at showdown (ties split) against these ranges."""
    rnd = rnd or _rnd
    hole, board = list(hole), list(board)
    known = set(hole) | set(board)
    need = 5 - len(board)
    deck_base = [c for c in range(52) if c not in known]
    won = 0.0
    for _ in range(iters):
        dead = set(known)
        opps = []
        for r in ranges:
            a, b = _sample(r, dead, board, rnd)
            dead.add(a)
            dead.add(b)
            opps.append((a, b))
        if need:
            rest = [c for c in deck_base if c not in dead]
            full = board + rnd.sample(rest, need)
        else:
            full = board
        me = evaluate(hole + full)
        ties = 0
        for a, b in opps:
            s = evaluate([a, b] + full)
            if s > me:
                break
            if s == me:
                ties += 1
        else:
            won += 1.0 / (ties + 1)
    return won / iters
