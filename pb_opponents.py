"""What we learn about each opponent team, kept as plain dicts so the model
pickles cleanly. Every frequency has a prior (pseudo-counts) so the first
hands give sensible numbers and the data takes over as it accumulates.
"""
from pb_cards import PAIR, TWO_PAIR

# Prior pseudo-counts per situation: a "textbook" opponent.
PRIORS = {
    # preflop, first to act (heads-up: the small blind / button)
    "open": {"fold": 3.0, "call": 1.5, "raise": 4.5},
    # preflop, facing a limp (big blind): check or raise
    "vs_limp": {"call": 6.0, "raise": 3.0},
    # preflop, facing a raise
    "vs_raise": {"fold": 4.5, "call": 3.5, "raise": 1.5},
    # preflop, facing a big raise or shove (call > 2.5x the pot before it)
    "vs_raise_big": {"fold": 5.5, "call": 3.0, "raise": 0.5},
    # preflop, facing a re-raise after they raised
    "vs_3bet": {"fold": 2.0, "call": 3.0, "raise": 3.0},
    # postflop, facing our bet, by its size relative to the pot
    "vs_bet_small": {"fold": 3.0, "call": 5.0, "raise": 1.5},
    "vs_bet_mid": {"fold": 4.0, "call": 4.5, "raise": 1.0},
    "vs_bet_big": {"fold": 5.0, "call": 4.0, "raise": 1.0},
    # postflop, facing a bet after already calling one earlier this hand
    "vs_bet_again": {"fold": 2.5, "call": 6.0, "raise": 1.0},
    # postflop, facing a raise after they bet
    "vs_raise_post": {"fold": 4.0, "call": 4.0, "raise": 1.5},
    # postflop, unopened: check or bet
    "post_open": {"call": 6.0, "raise": 3.5},
}


def size_bucket(ratio: float) -> str:
    if ratio <= 0.5:
        return "vs_bet_small"
    if ratio <= 0.9:
        return "vs_bet_mid"
    return "vs_bet_big"


def new_stats() -> dict:
    return {
        "hands": 0,
        "counts": {k: {a: 0.0 for a in v} for k, v in PRIORS.items()},
        # made-hand categories shown at showdown after betting/raising
        # postflop in that hand (how honest their aggression is)
        "aggr_shown": [],
    }


def freq(stats: dict, situation: str, action: str, prior_scale: float = 1.0) -> float:
    prior, seen = PRIORS[situation], stats["counts"][situation]
    total = prior_scale * sum(prior.values()) + sum(seen.values())
    return (prior_scale * prior[action] + seen[action]) / total


def record(stats: dict, situation: str, action: str):
    seen = stats["counts"][situation]
    if action in seen:
        seen[action] += 1


def observations(stats: dict, situation: str) -> float:
    return sum(stats["counts"][situation].values())


def min_category_when_aggressive(stats: dict, times_aggressive: int) -> int:
    """Weakest made hand we assume when they bet/raise postflop. Learned from
    showdowns once we have some, else from how often they bet at all."""
    shown = stats["aggr_shown"]
    if len(shown) >= 6:
        ordered = sorted(shown)
        # the hand they bet with at the 20th percentile (lower for one bet)
        q = 0.15 if times_aggressive <= 1 else 0.3
        return ordered[int(q * (len(ordered) - 1))]
    bet_rate = freq(stats, "post_open", "raise")
    if bet_rate < 0.25:
        return TWO_PAIR if times_aggressive >= 2 else PAIR
    if bet_rate < 0.45:
        return PAIR if times_aggressive >= 1 else 0
    return 0


def describe(name: str, stats: dict) -> str:
    def f(s, a):
        return f"{freq(stats, s, a):.0%}"
    return (f"{name}: {stats['hands']}h open-raise {f('open', 'raise')} "
            f"open-fold {f('open', 'fold')} fold-vs-raise {f('vs_raise', 'fold')} "
            f"fold-vs-bet s/m/b {f('vs_bet_small', 'fold')}/{f('vs_bet_mid', 'fold')}/"
            f"{f('vs_bet_big', 'fold')} bets {f('post_open', 'raise')} "
            f"shown {sorted(stats['aggr_shown'])[:12]}")
