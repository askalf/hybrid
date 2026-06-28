#!/usr/bin/env python3
"""
Deterministic arithmetic solver — the principled fix for confident-wrongness.

A router built on the cheap model's own signals (classification, self-consistency)
inherits the cheap model's blind spots: it can't tell confident-and-right from
confident-and-wrong. "17 to the power of 4" came back a *unanimous* 6859 (it is
83,521). No amount of self-agreement catches that.

But that whole class of failure is closed-form arithmetic, and for closed-form
arithmetic we don't need a smarter model — we need an exact oracle. Python's
arbitrary-precision integer math IS that oracle: free, instant, on-box, correct by
construction. So we try to reduce a query to a safe arithmetic expression and, if
(and ONLY if) it reduces cleanly, evaluate it exactly.

Design rule: CONSERVATIVE. solve() returns the exact answer string, or None if the
query is not unambiguously a closed-form computation. A false "I solved it" is worse
than falling through to the LLM, so anything ambiguous -> None -> the router's other
tiers handle it. The solver never guesses.

Beyond bare arithmetic the same exact-oracle principle covers a few more closed classes —
unit conversions (Fractions, so `1 inch = 25.4 mm` stays exact), percentage-change, and
simple multiples — each free, on-box, and correct by construction. Cross-dimension or
unknown-unit requests fall through, same conservatism.

    from solver import solve
    solve("What is 17 to the power of 4?")   # -> "83521"
    solve("How many feet in 3 miles?")        # -> "15840"  (exact unit conversion)
    solve("What is 20% off 50?")              # -> "40"
    solve("What is the capital of Japan?")    # -> None  (not arithmetic)
"""
import ast
import math
import re
from fractions import Fraction

# Hard ceilings so a pathological input (9**9**9, 100000 factorial) can't wedge the
# box. Over the ceiling -> refuse (return None) and let escalation deal with it.
_MAX_RESULT_DIGITS = 5000
_MAX_FACTORIAL = 5000

# Function names the safe evaluator is allowed to call. Everything else is rejected.
_ALLOWED_FUNCS = {"sqrt", "factorial"}


def _sqrt(x):
    """Exact integer root when the input is a perfect square, else a float."""
    if isinstance(x, int) and x >= 0:
        r = math.isqrt(x)
        if r * r == x:
            return r
    if x < 0:
        raise ValueError("sqrt of negative")
    return math.sqrt(x)


def _factorial(x):
    if not (isinstance(x, int) and x >= 0):
        raise ValueError("factorial needs a non-negative integer")
    if x > _MAX_FACTORIAL:
        raise ValueError("factorial too large")
    return math.factorial(x)


_FUNC_IMPL = {"sqrt": _sqrt, "factorial": _factorial}


# ---------------------------------------------------------------------------
# Natural-language -> arithmetic-expression normalization.
# Every transform is intentionally narrow. If a query survives normalization with
# any alphabetic token that isn't a whitelisted function name, we bail (return None)
# rather than risk mis-reading it.
# ---------------------------------------------------------------------------

# Leading question stems we strip ("what is 5+5?" -> "5+5").
_STEM = re.compile(
    r"^\s*(what(?:'s| is| does| would)?|whats|how much is|how many is|calculate|"
    r"compute|evaluate|solve|the value of|tell me|give me)\b[:\s]*", re.I)

# "<a> percent of <b>" / "<a>% of <b>"  ->  "(<a>/100)*<b>"   (handled before % means mod)
_PERCENT_OF = re.compile(r"(\d[\d,.]*)\s*(?:percent|%)\s*of\b", re.I)


def _normalize(q):
    s = q.strip().lower()
    s = s.rstrip("?. ")  # keep a trailing '!' — it may be factorial notation
    s = _STEM.sub("", s)
    s = s.replace("$", "").replace("√", " square root of ")
    s = s.replace("²", "**2").replace("³", "**3")

    # percentage change: "<n>% more/less than <m>", "<n>% off <m>", "increase/decrease
    # <m> by <n>%", "<m> increased/reduced by <n>%" -> exact arithmetic. Runs before the
    # "% of" and the bare "% -> mod" mappings so the '%' here is read as a percentage.
    s = re.sub(r"(\d[\d,.]*)\s*(?:percent|%)\s+more than\s+(\d[\d,.]*)", r"(\2)*(1+(\1)/100)", s)
    s = re.sub(r"(\d[\d,.]*)\s*(?:percent|%)\s+less than\s+(\d[\d,.]*)", r"(\2)*(1-(\1)/100)", s)
    s = re.sub(r"(\d[\d,.]*)\s*(?:percent|%)\s+off(?:\s+of)?\s+(\d[\d,.]*)", r"(\2)*(1-(\1)/100)", s)
    s = re.sub(r"\b(?:increase|raise)\s+(\d[\d,.]*)\s+by\s+(\d[\d,.]*)\s*(?:percent|%)", r"(\1)*(1+(\2)/100)", s)
    s = re.sub(r"\b(?:decrease|reduce|lower)\s+(\d[\d,.]*)\s+by\s+(\d[\d,.]*)\s*(?:percent|%)", r"(\1)*(1-(\2)/100)", s)
    s = re.sub(r"(\d[\d,.]*)\s+increased by\s+(\d[\d,.]*)\s*(?:percent|%)", r"(\1)*(1+(\2)/100)", s)
    s = re.sub(r"(\d[\d,.]*)\s+(?:decreased|reduced)\s+by\s+(\d[\d,.]*)\s*(?:percent|%)", r"(\1)*(1-(\2)/100)", s)

    # percent-of BEFORE any % -> mod mapping
    s = _PERCENT_OF.sub(r"(\1/100)*", s)

    # exponentiation phrasings
    s = re.sub(r"\b(?:to the power of|raised to the power of|raised to|to the power)\b",
               " ** ", s)
    s = re.sub(r"\bsquared\b", " **2 ", s)
    s = re.sub(r"\bcubed\b", " **3 ", s)

    # factorial:  "5!", "5 factorial", "factorial of 5"  -> factorial(5)
    s = re.sub(r"factorial of\s*(\d[\d,]*)", r"factorial(\1)", s)
    s = re.sub(r"(\d[\d,]*)\s*factorial", r"factorial(\1)", s)
    s = re.sub(r"(\d[\d,]*)\s*!", r"factorial(\1)", s)

    # square root: wrap only the immediately-following number or (parenthesized group)
    s = re.sub(r"\b(?:square root of|sqrt of|sqrt)\s*(\d[\d,.]*|\([^()]*\))",
               r"sqrt(\1)", s)

    # word operators
    s = re.sub(r"\b(?:plus|added to)\b", " + ", s)
    s = re.sub(r"\b(?:minus|subtract(?:ed by)?|less)\b", " - ", s)
    s = re.sub(r"\b(?:times|multiplied by|product of)\b", " * ", s)
    s = re.sub(r"\b(?:divided by|over)\b", " / ", s)
    s = re.sub(r"\b(?:modulo|mod)\b", " % ", s)
    s = re.sub(r"\^", " ** ", s)

    # multiples / simple fractions of a quantity ("half of 60", "double 21")
    s = re.sub(r"\bhalf of\b", " 0.5 * ", s)
    s = re.sub(r"\b(?:a\s+)?third of\b", " (1/3) * ", s)
    s = re.sub(r"\b(?:a\s+)?quarter of\b", " 0.25 * ", s)
    s = re.sub(r"\b(?:double|twice)\b", " 2 * ", s)
    s = re.sub(r"\b(?:triple|thrice)\b", " 3 * ", s)

    # leftover glue words that are safe to drop once operators are symbolic
    s = re.sub(r"\b(?:of|the|a|an|is|are|value)\b", " ", s)
    s = re.sub(r"\bequals?\b", " ", s)

    # strip thousands separators inside numbers (1,000 -> 1000)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)

    return s.strip()


def _looks_arithmetic(s):
    """True only if nothing but numbers, operators, and whitelisted funcs remain."""
    if not s or not any(ch.isdigit() for ch in s):
        return False
    probe = s
    for fn in _ALLOWED_FUNCS:
        probe = probe.replace(fn, " ")
    # any surviving letter means an unparsed word -> not safe to claim
    return not re.search(r"[a-z]", probe)


# ---------------------------------------------------------------------------
# Safe AST evaluator. Only the node types below are permitted; anything else
# (names, attributes, comprehensions, calls to non-whitelisted funcs) raises.
# ---------------------------------------------------------------------------
_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: None,  # handled specially for the digit-ceiling guard
}
_UNARYOPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}


def _pow_guard(base, exp):
    if isinstance(base, int) and isinstance(exp, int) and exp > 1 and base not in (-1, 0, 1):
        digits = exp * math.log10(abs(base) or 1) if base else 0
        if digits > _MAX_RESULT_DIGITS:
            raise ValueError("result too large")
    return base ** exp


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp):
        op = type(node.op)
        if op not in _BINOPS:
            raise ValueError("operator not allowed")
        a, b = _eval(node.left), _eval(node.right)
        return _pow_guard(a, b) if op is ast.Pow else _BINOPS[op](a, b)
    if isinstance(node, ast.UnaryOp):
        op = type(node.op)
        if op not in _UNARYOPS:
            raise ValueError("unary op not allowed")
        return _UNARYOPS[op](_eval(node.operand))
    if isinstance(node, ast.Call):
        if (not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS
                or node.keywords or len(node.args) != 1):
            raise ValueError("call not allowed")
        return _FUNC_IMPL[node.func.id](_eval(node.args[0]))
    raise ValueError(f"node not allowed: {type(node).__name__}")


def _format(value):
    """Exact integers as integers; floats trimmed to a sane precision."""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{round(value, 6):.6f}".rstrip("0").rstrip(".")
    raise ValueError("non-numeric result")


# ---------------------------------------------------------------------------
# Exact unit conversion — a separate oracle from the arithmetic path. "How many feet
# in 3 miles" isn't an arithmetic expression, but it IS exact (1 inch = 25.4 mm, by
# definition), so we convert with Fractions and stay correct by construction. Fires
# ONLY for known units in the SAME dimension; anything else returns None and the query
# falls through (a mile is never "converted" to a kilogram). Same conservatism as solve.
# ---------------------------------------------------------------------------
_UNITS = {}  # name -> (dimension, Fraction factor to that dimension's base unit)


def _unit(dim, factor, *names):
    for n in names:
        _UNITS[n] = (dim, Fraction(factor))


_unit("time", 1, "second", "seconds", "sec", "secs")
_unit("time", 60, "minute", "minutes", "min", "mins")
_unit("time", 3600, "hour", "hours", "hr", "hrs")
_unit("time", 86400, "day", "days")
_unit("time", 604800, "week", "weeks")
_unit("len", 1, "mm", "millimeter", "millimeters")
_unit("len", 10, "cm", "centimeter", "centimeters")
_unit("len", 1000, "m", "meter", "meters", "metre", "metres")
_unit("len", 1000000, "km", "kilometer", "kilometers", "kilometre", "kilometres")
_unit("len", Fraction(254, 10), "inch", "inches")            # 1 in = 25.4 mm exactly
_unit("len", Fraction(3048, 10), "foot", "feet", "ft")
_unit("len", Fraction(9144, 10), "yard", "yards", "yd")
_unit("len", Fraction(16093440, 10), "mile", "miles")
_unit("mass", 1, "gram", "grams")
_unit("mass", 1000, "kilogram", "kilograms", "kg")
_unit("mass", Fraction(45359237, 100000), "pound", "pounds", "lb", "lbs")   # 1 lb = 453.59237 g
_unit("mass", Fraction(45359237, 1600000), "ounce", "ounces", "oz")         # 1 oz = lb/16
_unit("vol", 1, "milliliter", "milliliters", "ml")
_unit("vol", 1000, "liter", "liters", "litre", "litres")
_unit("vol", Fraction(3785411784, 1000000), "gallon", "gallons")            # US gallon = 3.785411784 L

# "how many <u2> in <n> <u1>"  and  "[convert] <n> <u1> in|into|to <u2>"
_CONV1 = re.compile(r"how many ([a-z]+)\s+(?:are\s+)?(?:there\s+)?in\s+\b(\d[\d,]*\.?\d*|a|an|one)\b\s*([a-z]+)", re.I)
_CONV2 = re.compile(r"(?:convert\s+)?\b(\d[\d,]*\.?\d*|a|an|one)\b\s*([a-z]+)\s+(?:in|into|to)\s+([a-z]+)", re.I)


def _fmt_frac(v):
    """Exact Fraction -> int string if whole, else a trimmed 6-place decimal."""
    if v.denominator == 1:
        return str(v.numerator)
    return f"{round(float(v), 6):.6f}".rstrip("0").rstrip(".")


def _convert(n_str, u1, u2):
    a, b = _UNITS.get(u1.lower()), _UNITS.get(u2.lower())
    if not a or not b or a[0] != b[0]:        # unknown unit, or different dimensions
        return None
    t = n_str.lower().replace(",", "")
    n = Fraction(1) if t in ("a", "an", "one") else Fraction(t)
    return _fmt_frac(n * a[1] / b[1])


def _try_convert(query):
    """Exact unit conversion if the query is unambiguously one, else None."""
    m = _CONV1.search(query)
    if m:                                     # groups: (u2, n, u1)
        return _convert(m.group(2), m.group(3), m.group(1))
    m = _CONV2.search(query)
    if m:                                     # groups: (n, u1, u2)
        return _convert(m.group(1), m.group(2), m.group(3))
    return None


def solve(query):
    """Return the exact answer string for a closed-form arithmetic query, else None."""
    if not query or len(query) > 200:
        return None
    conv = _try_convert(query)
    if conv is not None:
        return conv
    try:
        expr = _normalize(query)
        if not _looks_arithmetic(expr):
            return None
        tree = ast.parse(expr, mode="eval")
        return _format(_eval(tree))
    except Exception:
        return None  # any failure -> decline, never guess


def eval_expr(expr):
    """Safely evaluate a BARE arithmetic expression (no NL normalization). Returns a
    number, or None if it isn't a clean arithmetic expression. Used by the verifier to
    re-check equations the local model asserts in its answer. '^' is read as power
    (the human meaning), not Python's bitwise-xor."""
    try:
        s = expr.strip().replace("$", "").replace("^", "**")
        s = re.sub(r"(?<=\d),(?=\d)", "", s)  # 1,000 -> 1000
        if not s or not any(c.isdigit() for c in s) or re.search(r"[a-z]", s, re.I):
            return None
        return _eval(ast.parse(s, mode="eval"))
    except Exception:
        return None


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:])
    print(solve(q) if q else "usage: python solver.py <arithmetic question>")
