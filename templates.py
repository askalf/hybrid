#!/usr/bin/env python3
"""
Deterministic word-problem templates — transcription without the model.

The derive tier (equations.py) proved the architecture's key fact: the local model's
real job on a word problem is TRANSCRIPTION — turn the problem's stated relationships
into equations — and the exact oracle does the rest. Live testing then proved the
inverse: transcription is the one surface the oracle cannot check, so it is exactly
where a cheap model can still serve a wrong answer ("setup re-derived" from a garbled
but self-consistent system).

This module removes the model from transcription for the handful of shapes that
dominate everyday quantitative queries. Each template recognizes ONE rigid shape,
extracts its slots deterministically, and computes the closed form exactly over
`fractions.Fraction` — no model, no tokens, no latency, correct by construction:

  rate         "Each pallet holds 3,672 cans. How many cans are on 38 pallets?"
               "A printer prints 2,417 pages per hour. How many pages in 94 hours?"
  sum-diff     "A bat and a ball cost $1.10 total. The bat costs $1.00 more than
                the ball. How much is the ball?"
  reverse-pct  "A shirt costs $40 after a 20% discount. What was the original price?"
  shift        "A number decreased by 12 is 39. What is the number?"
  combo        "Movie tickets cost $9 for kids and $14 for adults. What do 3 kids
                and 2 adults pay in total?"

Design rule, stricter even than solver.py: CONSERVATIVE. A template answers only when
  - the whole query matches the shape (declaration AND question),
  - every number in the query is consumed by the shape's slots (the v1.1.1 lesson:
    a partial parse that silently drops a quantity is the worst failure mode),
  - no number-WORDS ("half", "twice", "dozen") lurk outside the slots,
  - the nouns agree between declaration and question (stemmed),
  - money markers are consistent, and every quantity involved is positive.
Anything else returns None and the query falls through to the model tiers unchanged —
set-logic riddles ("Emma has 5 brothers..."), work-rate traps, exponential growth, and
multi-step phrasings all decline here by construction.

    from templates import solve
    solve("A book costs $18.75. How much do 4 books cost?")   # -> ("75", "rate")
    solve("Emma has 5 brothers. Each brother has 3 sisters. How many sisters does Emma have?")
    # -> None (not a priced/measured shape — the model tiers handle it)
"""
import re
from fractions import Fraction

from equations import fmt

_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")

# Number-token pattern used inside shape regexes: one _NUM token, kept with its $ so
# money-consistency can be checked afterward.
N = r"(\$?\d[\d,]*(?:\.\d+)?)"

# Number-words outside a slot mean the shape did NOT consume every quantity — decline.
_NUMBER_WORDS = re.compile(
    r"\b(?:half|halves|third|quarter|dozen|couple|pair|double|twice|triple|thrice|"
    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|score)\b")

# Auxiliaries/particles the rate question's optional subject-noun slot must not eat.
_AUX = {"do", "does", "did", "are", "is", "was", "were", "will", "would", "can",
        "could", "should", "it", "in", "on", "of", "for", "at", "to", "the", "a", "an"}


def _frac(tok):
    """'$1,847.50' -> an exact Fraction (through the string, never a float)."""
    return Fraction(tok.replace("$", "").replace(",", ""))


def _is_money(tok):
    return tok.startswith("$")


def _stem(w):
    """Naive singular/plural fold: 'pallets' == 'pallet'. A miss just declines."""
    w = w.lower()
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def _same(a, b):
    return _stem(a) == _stem(b)


def _norm(query):
    """Lowercase, collapse whitespace, spell '%' one way. Numbers keep $ and commas."""
    q = " ".join(query.lower().split())
    q = re.sub(r"(\d)\s*percent\b", r"\1%", q)
    return q


# ---------------------------------------------------------------------------
# rate — a per-item or per-time-unit rate, asked for N items/units. Exactly the
# class the local model multiplies wrong with full confidence; here it is a single
# exact product. Two numbers: the rate and the count.
# ---------------------------------------------------------------------------

_RATE_DECL = [
    # "each pallet holds 3,672 cans" / "each crate weighs 23.7 kg" / "each fuel tank holds 13.9 liters"
    re.compile(r"\beach (?:[a-z]+ )?(?P<en>[a-z]+) (?:holds?|weighs?|contains?|carries|"
               r"produces?|makes?|draws?|uses?|costs?) " + N + r"\s*(?P<unit>[a-z]*)"),
    # "a book costs $18.75" / "one ticket costs $12.50"
    re.compile(r"\b(?:a|an|one) (?:[a-z]+ )?(?P<en>[a-z]+) costs? " + N + r"(?P<unit>)"),
    # "notebooks at $12.50 each"
    re.compile(r"\b(?P<en>[a-z]+) at " + N + r" each(?P<unit>)"),
    # "makes 1,847 widgets per day" / "draws 1,384 watts per rack"
    re.compile(r"\b" + N + r" (?P<unit>[a-z]+) per (?P<en>[a-z]+)\b"),
]

_RATE_Q = [
    # "how much do 4 books cost" / "how many cans are on 38 pallets" / "how many in 47 minutes"
    re.compile(r"\bhow (?:much|many)(?: (?P<qu>[a-z]+))?(?: [a-z]+)*? " + N + r" (?P<qn>[a-z]+)"),
    # "what do 41 crates weigh" / "what do 7 notebooks cost"
    re.compile(r"\bwhat (?:do|does|will|would) " + N +
               r" (?P<qn>[a-z]+) (?:weigh|cost|hold|pay|produce|make|use)(?P<qu>)"),
    # "what is the total draw of 219 racks" / "total weight of 41 crates"
    re.compile(r"\btotal [a-z]+ (?:of|for) " + N + r" (?P<qn>[a-z]+)(?P<qu>)"),
]


def _rate(q, toks):
    if len(toks) != 2:
        return None
    for dp in _RATE_DECL:
        d = dp.search(q)
        if not d:
            continue
        for qp in _RATE_Q:
            m = qp.search(q, d.end())
            if not m:
                continue
            # the count must attach to the per/each noun...
            if not _same(m.group("qn"), d.group("en")):
                continue
            # ...and a named asked-for quantity must be the rate's unit noun
            qu, unit = m.group("qu") or "", d.group("unit") or ""
            if qu in _AUX:
                qu = ""
            if qu and unit and not _same(qu, unit):
                continue
            if qu and not unit:
                continue
            # group ORDER differs across the declaration/question shapes; the value
            # is simply each match's one numeric group
            rate = _frac(next(g for g in d.groups() if g and _NUM.fullmatch(g)))
            count = _frac(next(g for g in m.groups() if g and _NUM.fullmatch(g)))
            if rate <= 0 or count <= 0:
                return None
            return rate * count
    return None


# ---------------------------------------------------------------------------
# sum-diff — two items with a stated total and a stated gap ("the bat costs $1.00
# more than the ball"). Closed form: smaller = (total - gap) / 2. The classic trap
# answer ($0.10) is exactly what this template cannot produce.
# ---------------------------------------------------------------------------

_SUMDIFF = re.compile(
    r"\b(?:a|an|the) (?P<a>[a-z]+) and (?:a|an|the) (?P<b>[a-z]+) cost " + N +
    r"(?: in total| total| together| altogether)?\b.*?"
    r"\bthe (?P<c>[a-z]+) (?:costs?|is|weighs?) " + N + r" more than the (?P<d>[a-z]+)\b.*?"
    r"\bhow much (?:is|does|was|for) the (?P<e>[a-z]+)")

_SUMDIFF_ABSTRACT = re.compile(
    r"\bsum of two numbers is " + N + r"\b.*?\btheir difference is " + N +
    r"\b.*?\bwhat is the (?P<which>larger|largest|bigger|greater|smaller|smallest) (?:number|one)")


def _sum_diff(q, toks):
    if len(toks) != 2:
        return None
    m = _SUMDIFF_ABSTRACT.search(q)
    if m:
        s, d = _frac(m.group(1)), _frac(m.group(2))
        if s <= d or d <= 0:
            return None
        big = m.group("which") in ("larger", "largest", "bigger", "greater")
        return (s + d) / 2 if big else (s - d) / 2
    m = _SUMDIFF.search(q)
    if not m:
        return None
    a, b, c, d_, e = (m.group(k) for k in "abcde")
    items = {_stem(a), _stem(b)}
    if len(items) != 2 or {_stem(c), _stem(d_)} != items or _stem(e) not in items:
        return None
    s_tok, d_tok = m.group(3), m.group(5)
    if _is_money(s_tok) != _is_money(d_tok):
        return None
    s, d = _frac(s_tok), _frac(d_tok)
    if s <= d or d <= 0:                     # a "gap" >= the total is a trick, not a shape
        return None
    return (s + d) / 2 if _same(e, c) else (s - d) / 2


# ---------------------------------------------------------------------------
# reverse-pct — a value AFTER a percentage change, asked for the original. The
# canonical small-model miss (the "$48" answer); the closed form is v / (1 ± p/100).
# ---------------------------------------------------------------------------

_DOWNS = ("discount", "reduction", "decrease", "markdown",
          "discounted", "marked down", "reduced", "decreased")
_DOWN = r"discount|reduction|decrease|markdown"
_UP = r"raise|increase|markup"

_REVPCT = [
    # "after a 15% discount, a lamp costs $68" / "after a 25% raise, my salary is $75,000"
    re.compile(r"\bafter (?:a |an )?" + N + r"% (?P<dir>" + _DOWN + "|" + _UP + r")\b[^0-9$]*?" + N),
    # "a shirt costs $40 after a 20% discount"   (value first -> 'flip' marks the order)
    re.compile(r"\bcosts? " + N + r"(?P<flip>) after (?:a |an )?" + N +
               r"% (?P<dir>" + _DOWN + "|" + _UP + r")\b"),
    # "a jacket was discounted 30% and now costs $84"
    re.compile(r"\b(?P<dir>discounted|marked down|reduced) (?:by )?" + N + r"% and now costs " + N),
]

_REVPCT_NUMBER = re.compile(
    r"\b(?:a |the )?number (?P<dir>increased|decreased|reduced) by " + N + r"% is " + N)

_ORIGINAL_Q = re.compile(r"\b(original|before|to begin with|initially|what is the number)\b")


def _reverse_pct(q, toks):
    if len(toks) != 2 or not _ORIGINAL_Q.search(q):
        return None
    for pat in _REVPCT + [_REVPCT_NUMBER]:
        m = pat.search(q)
        if not m:
            continue
        nums = [x for x in m.groups() if x and _NUM.fullmatch(x)]
        if len(nums) != 2:
            continue
        p, v = (nums[1], nums[0]) if "flip" in m.re.groupindex else (nums[0], nums[1])
        p, v = _frac(p), _frac(v)
        down = m.group("dir") in _DOWNS
        if v <= 0 or p <= 0 or (down and p >= 100):
            return None
        return v / (1 - p / 100) if down else v / (1 + p / 100)
    return None


# ---------------------------------------------------------------------------
# shift — "a number decreased by 12 is 39": one add or subtract, stated in words.
# ---------------------------------------------------------------------------

_SHIFT = re.compile(
    r"\b(?:a |the )?number (?P<dir>increased|decreased|reduced) by " + N + r" is " + N)


def _shift(q, toks):
    if len(toks) != 2 or "%" in q:
        return None
    m = _SHIFT.search(q)
    if not m or not _ORIGINAL_Q.search(q):
        return None
    a, b = _frac(m.group(2)), _frac(m.group(3))
    if a <= 0 or b <= 0:
        return None
    return b - a if m.group("dir") == "increased" else b + a


# ---------------------------------------------------------------------------
# combo — two priced kinds, asked for a stated mix: the sum of two exact products.
# ---------------------------------------------------------------------------

_COMBO_DECL = [
    # "adult tickets cost $12 and child tickets cost $7"
    re.compile(r"\b(?P<n1>[a-z]+) tickets? cost " + N + r" and (?P<n2>[a-z]+) tickets? cost " + N),
    # "movie tickets cost $9 for kids and $14 for adults"
    re.compile(r"\btickets? cost " + N + r" for (?P<n1>[a-z]+) and " + N + r" for (?P<n2>[a-z]+)"),
]

_COMBO_Q = re.compile(
    r"\b(?:what do|how much do|how much will|what will) " + N + r" (?P<m1>[a-z]+) and " + N +
    r" (?P<m2>[a-z]+)(?: tickets?)? (?:pay|cost)")


def _combo(q, toks):
    if len(toks) != 4:
        return None
    for dp in _COMBO_DECL:
        d = dp.search(q)
        if not d:
            continue
        m = _COMBO_Q.search(q, d.end())
        if not m:
            continue
        if dp is _COMBO_DECL[0]:
            n1, p1, n2, p2 = d.group("n1"), d.group(2), d.group("n2"), d.group(4)
        else:
            p1, n1, p2, n2 = d.group(1), d.group("n1"), d.group(3), d.group("n2")
        if _is_money(p1) != _is_money(p2):
            return None
        prices = {_stem(n1): _frac(p1), _stem(n2): _frac(p2)}
        c1, m1, c2, m2 = _frac(m.group(1)), m.group("m1"), _frac(m.group(3)), m.group("m2")
        if len(prices) != 2 or {_stem(m1), _stem(m2)} != set(prices):
            return None
        if c1 <= 0 or c2 <= 0 or any(v <= 0 for v in prices.values()):
            return None
        return c1 * prices[_stem(m1)] + c2 * prices[_stem(m2)]
    return None


_TEMPLATES = [
    ("rate", _rate),
    ("sum-diff", _sum_diff),
    ("reverse-pct", _reverse_pct),
    ("shift", _shift),
    ("combo", _combo),
]


def solve(query):
    """Return (answer_string, template_name) when the query IS one of the shapes,
    else None. Same contract discipline as solver.solve: a returned answer consumed
    every number in the query and is exact; anything ambiguous falls through."""
    if not query or len(query) > 300:
        return None
    q = _norm(query)
    if re.search(r"-\s*\d", q):              # negatives are outside every shape here
        return None
    toks = _NUM.findall(q)
    if not 2 <= len(toks) <= 4:
        return None
    probe = q.replace("two numbers", " ")    # the abstract sum-diff shape names its pair
    for t in toks:                           # number-words OUTSIDE the digit slots -> decline
        probe = probe.replace(t, " ", 1)
    if _NUMBER_WORDS.search(probe):
        return None
    for name, fn in _TEMPLATES:
        try:
            val = fn(q, toks)
        except (ValueError, ZeroDivisionError):
            val = None
        if val is not None and val > 0:
            return fmt(val), name
    return None


if __name__ == "__main__":
    import sys
    r = solve(" ".join(sys.argv[1:]) or sys.stdin.read().strip())
    print(r if r else "(no template)")
