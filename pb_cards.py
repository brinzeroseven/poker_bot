"""Fast integer cards and a 7-card hand evaluator (card_utils.evaluate is
too slow for Monte Carlo: it validates and tries all 21 five-card subsets).

Cards are ints 0..51: rank = c // 4 (0=2 .. 12=A), suit = c % 4.
Higher evaluate() score = better hand.
"""
from itertools import combinations

RANKS = "23456789TJQKA"
SUITS = "CDHS"  # same order as card_utils


def to_int(c) -> int:
    """card_utils.Card (or "AS"-style string) -> 0..51."""
    if isinstance(c, str):
        return RANKS.index(c[0].upper()) * 4 + SUITS.index(c[1].upper())
    return RANKS.index(c.rank) * 4 + SUITS.index(c.suit)


def to_str(c: int) -> str:
    return RANKS[c // 4] + SUITS[c % 4]


# Hand categories
HIGH, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)


def _straight_high(rank_mask: int) -> int:
    """Return high rank of best straight in a 13-bit mask, or -1."""
    # shift ranks up one bit and put the ace at bit 0 so A-2-3-4-5 is found
    m = (rank_mask << 1) | ((rank_mask >> 12) & 1)
    for hi in range(13, 3, -1):
        need = 0b11111 << (hi - 4)
        if m & need == need:
            return hi - 1  # back to 0..12 rank (ace-low straight -> 3, i.e. '5')
    return -1


def _score(cat: int, kickers) -> int:
    s = cat
    for k in kickers[:5]:
        s = s * 16 + k
    for _ in range(5 - len(kickers[:5])):
        s *= 16
    return s


def evaluate(cs) -> int:
    """Evaluate 5..7 cards. Returns comparable int."""
    counts = [0] * 13
    suit_cards = [[], [], [], []]
    for c in cs:
        r = c >> 2
        counts[r] += 1
        suit_cards[c & 3].append(r)

    # flush / straight flush
    for sc in suit_cards:
        if len(sc) >= 5:
            mask = 0
            for r in sc:
                mask |= 1 << r
            sh = _straight_high(mask)
            if sh >= 0:
                return _score(STRAIGHT_FLUSH, [sh])
            return _score(FLUSH, sorted(sc, reverse=True)[:5])

    quads, trips, pairs, singles = [], [], [], []
    mask = 0
    for r in range(12, -1, -1):
        n = counts[r]
        if n:
            mask |= 1 << r
        if n == 4:
            quads.append(r)
        elif n == 3:
            trips.append(r)
        elif n == 2:
            pairs.append(r)
        elif n == 1:
            singles.append(r)

    if quads:
        q = quads[0]
        rest = [r for r in range(12, -1, -1) if counts[r] and r != q]
        return _score(QUADS, [q, rest[0]])
    if trips and (len(trips) >= 2 or pairs):
        t = trips[0]
        p = max(trips[1:] + pairs)
        return _score(FULL_HOUSE, [t, p])
    sh = _straight_high(mask)
    if sh >= 0:
        return _score(STRAIGHT, [sh])
    if trips:
        return _score(TRIPS, [trips[0]] + singles[:2])
    if len(pairs) >= 2:
        p1, p2 = pairs[0], pairs[1]
        kick = max(singles + pairs[2:] or [0])
        return _score(TWO_PAIR, [p1, p2, kick])
    if pairs:
        return _score(PAIR, [pairs[0]] + singles[:3])
    return _score(HIGH, singles[:5])


def category(score: int) -> int:
    return score // (16 ** 5)


def evaluate_slow(cs) -> int:
    """Reference implementation for tests: best 5-card subset."""
    return max(evaluate(list(c)) for c in combinations(cs, 5))
