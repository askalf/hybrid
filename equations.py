#!/usr/bin/env python3
"""
Setup re-derivation — an exact oracle for the RELATIONSHIPS, not just the arithmetic.

The plug-back verifier (verify.py) checks the model's answer against the constraints the
model transcribes — but only when those constraints arrive as pure numbers. Asked for its
working, the cheap model often goes SYMBOLIC instead (`x + (x + 1.00) = 1.10`), which the
numeric oracle can't touch, so the answer used to fall through to self-consistency — the
weakest tier, and exactly where the wrong-setup traps slip.

This module makes the symbols checkable. The model transcribes the problem's stated
relationships as equations over named unknowns; we solve the linear system OURSELVES,
exactly (Gaussian elimination over `fractions.Fraction` — no floats, no model), and
compare the derived value against the answer the model committed to:

    EQN: bat + ball = 1.10
    EQN: bat = ball + 1.00
    ANSWER: ball = 0.10        ->  the system derives ball = 1/20 = 0.05  ->  MISMATCH

A mismatch means the model mis-solved its own transcription of the problem — a HARD
escalate signal, strictly stronger than plug-back: plug-back checks that an answer is
CONSISTENT with the stated setup; this DERIVES the answer from the stated setup
independently, so a System-1 slip is caught even when the model would have back-filled
a tautological numeric check.

Design rule, same as solver.py: CONSERVATIVE. Anything nonlinear, inconsistent,
underdetermined, or ambiguous -> None / "none", and the router's other tiers handle it.

Honest boundary, same as verify.py: the derivation is only as good as the transcription.
If the model's misconception leaks INTO the equations (a wrong rate written as if the
problem stated it), the derived value matches the wrong answer and both sail through.
We re-derive the answer from the stated setup; we cannot verify the setup itself.

    from equations import verdict
    verdict("EQN: bat + ball = 1.10\\nEQN: bat = ball + 1.00\\nANSWER: ball = 0.10")
    # -> ("mismatch", {...derived: 0.05...})
"""
import ast
import re
from fractions import Fraction


class _NonLinear(Exception):
    """The expression doesn't reduce to a linear form (var*var, var/var, var**n...)."""


# ---------------------------------------------------------------------------
# Expression -> linear form.  A linear form is (coeffs, const): a dict of
# {variable: Fraction coefficient} plus a Fraction constant term. We evaluate the
# AST over this algebra; any operation that would leave the linear world raises.
# ---------------------------------------------------------------------------

def _frac(v):
    """Numeric literal -> exact Fraction. Floats go through str() so the literal the
    model WROTE ('1.10') is what we keep — Fraction('1.1') == 11/10, no binary noise."""
    if isinstance(v, int):
        return Fraction(v)
    if isinstance(v, float):
        return Fraction(str(v))
    raise _NonLinear("non-numeric constant")


def _add(a, b, sign=1):
    coeffs = dict(a[0])
    for k, c in b[0].items():
        coeffs[k] = coeffs.get(k, Fraction(0)) + sign * c
    return {k: c for k, c in coeffs.items() if c != 0}, a[1] + sign * b[1]


def _scale(a, k):
    return {v: c * k for v, c in a[0].items()}, a[1] * k


def _lin(node):
    """AST node -> linear form, or raise _NonLinear."""
    if isinstance(node, ast.Expression):
        return _lin(node.body)
    if isinstance(node, ast.Constant):
        return {}, _frac(node.value)
    if isinstance(node, ast.Name):
        return {node.id.lower(): Fraction(1)}, Fraction(0)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        a = _lin(node.operand)
        return a if isinstance(node.op, ast.UAdd) else _scale(a, Fraction(-1))
    if isinstance(node, ast.BinOp):
        a, b = _lin(node.left), _lin(node.right)
        if isinstance(node.op, ast.Add):
            return _add(a, b)
        if isinstance(node.op, ast.Sub):
            return _add(a, b, sign=-1)
        if isinstance(node.op, ast.Mult):
            if not a[0]:                      # const * linear
                return _scale(b, a[1])
            if not b[0]:                      # linear * const
                return _scale(a, b[1])
            raise _NonLinear("var * var")
        if isinstance(node.op, ast.Div):
            if b[0] or b[1] == 0:             # divide only by a nonzero constant
                raise _NonLinear("divide by variable or zero")
            return _scale(a, 1 / b[1])
        if isinstance(node.op, ast.Pow):
            if a[0] or b[0]:                  # constants only — anything else is nonlinear
                raise _NonLinear("power of variable")
            e = b[1]
            if e.denominator != 1 or abs(e.numerator) > 16:
                raise _NonLinear("exponent out of range")   # word-problem setups don't need 9**9**9
            return {}, a[1] ** e.numerator
    raise _NonLinear(f"node not allowed: {type(node).__name__}")


def _delatex(s):
    """The model often answers in LaTeX ('\\( 1.5 \\times C = 1.5 \\)', '\\frac{2}{3}').
    Translate the arithmetic subset; any OTHER LaTeX command keeps its braces, fails the
    parse, and the line is dropped — conservative by construction."""
    s = re.sub(r"\\[()\[\]]", " ", s)                     # \( \) \[ \] display delimiters
    s = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\(?:times|cdot)\b", " * ", s)
    s = re.sub(r"\\div\b", " / ", s)
    return s


def _normalize(expr):
    """Model-written equation side -> parseable Python expression. Narrow transforms
    only: LaTeX arithmetic, $, thousands commas, ^ as power, 'n%' as n/100, and implicit
    multiplication ('1.5 chickens' -> '1.5*chickens'). Anything still unparseable is
    skipped upstream."""
    s = _delatex(expr).strip().lower().replace("$", "").replace("^", "**")
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", s)
    # digit then word -> multiplication ('1.5 chickens', '2x'); the e-digit lookahead
    # spares scientific notation ('2e5' stays a number)
    s = re.sub(r"(\d)\s*(?!e\d)(?=[a-z(])", r"\1*", s)
    return s


def parse_equation(line):
    """One 'lhs = rhs' string -> a linear-form pair, or None if it isn't a clean linear
    equation. Both sides must parse; at least one side must mention a variable (a pure
    numeric identity is verify.py's job, not ours)."""
    if line.count("=") != 1:
        return None
    lhs, rhs = (s.strip() for s in line.split("="))
    if not lhs or not rhs:
        return None
    try:
        a = _lin(ast.parse(_normalize(lhs), mode="eval"))
        b = _lin(ast.parse(_normalize(rhs), mode="eval"))
    except (_NonLinear, SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return None
    if not a[0] and not b[0]:
        return None
    return a, b


# ---------------------------------------------------------------------------
# System solving — Gaussian elimination over Fractions, exact by construction.
# ---------------------------------------------------------------------------

def solve_system(pairs):
    """[(lhs_form, rhs_form)] -> {var: Fraction} for every UNIQUELY determined variable.
    Returns None if the parsed system is self-contradictory (0 = nonzero) — a garbled
    transcription we refuse to reason from — or if nothing is determined. Variables the
    system leaves free are simply absent (conservative: report only what is pinned)."""
    rows = []
    vars_ = sorted({v for a, b in pairs for v in {**a[0], **b[0]}})
    if not vars_:
        return None
    idx = {v: i for i, v in enumerate(vars_)}
    for a, b in pairs:                      # lhs = rhs  ->  (lhs - rhs) coeffs | const
        diff = _add(a, b, sign=-1)
        row = [Fraction(0)] * len(vars_) + [-diff[1]]
        for v, c in diff[0].items():
            row[idx[v]] = c
        rows.append(row)

    # forward elimination to reduced row-echelon form
    pivot_of = {}                           # column -> row holding its pivot
    r = 0
    for col in range(len(vars_)):
        piv = next((i for i in range(r, len(rows)) if rows[i][col] != 0), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        rows[r] = [x / rows[r][col] for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][col] != 0:
                f = rows[i][col]
                rows[i] = [x - f * y for x, y in zip(rows[i], rows[r])]
        pivot_of[col] = r
        r += 1

    if any(all(x == 0 for x in row[:-1]) and row[-1] != 0 for row in rows):
        return None                         # inconsistent: the transcription contradicts itself

    out = {}
    for v, col in idx.items():
        row = pivot_of.get(col)
        if row is not None and all(rows[row][c] == 0 for c in range(len(vars_)) if c != col):
            out[v] = rows[row][-1]          # pivot row touches only this variable -> pinned
    return out or None


# ---------------------------------------------------------------------------
# Response parsing — EQN: lines (or any variable-bearing equation line) + ANSWER: line.
# ---------------------------------------------------------------------------

_EQN_LINE = re.compile(r"(?im)^\s*(?:eqn|equation)\s*:\s*(.+?)\s*$")
_ANSWER_LINE = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")
_ANSWER_EQ = re.compile(r"([a-z][a-z0-9_]*)\s*=\s*(.+)", re.I)
_NUM = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")
_LITERAL = re.compile(r"^[-+]?\$?\d[\d,]*(?:\.\d+)?$")


def parse_equations(text):
    """Every parseable linear equation in the response: tagged EQN: lines first; if the
    model skipped the tag, any plain line holding exactly one '=' with a letter on it.
    ANSWER lines are never read as equations — the answer is what we check AGAINST."""
    lines = _EQN_LINE.findall(text)
    if not lines:
        lines = [ln for ln in text.splitlines()
                 if ln.count("=") == 1 and re.search(r"[a-z]", ln, re.I)
                 and not _ANSWER_LINE.match(ln)]
    return [p for p in (parse_equation(ln) for ln in lines) if p is not None]


def parse_answer(text):
    """The ANSWER: line -> (variable-or-None, claimed float, decimals shown, raw line).
    Accepts 'ANSWER: ball = 0.05', 'ANSWER: $0.05', 'ANSWER: the ball costs 5 cents'
    (last number wins), and exact-expression forms — 'ANSWER: C = 2/3' or LaTeX
    '\\( C = \\frac{2}{3} \\)' — which get evaluated exactly and matched tightly (an
    expression the model wrote is exact; only a rounded LITERAL earns rounding slack).
    Returns None if there's no ANSWER line or no number on it."""
    m = None
    for m in _ANSWER_LINE.finditer(text):
        pass                                # keep the LAST answer line (models restate)
    if not m:
        return None
    line = _delatex(m.group(1)).strip()
    em = _ANSWER_EQ.match(line)
    var, rhs = (em.group(1).lower(), em.group(2).strip()) if em else (None, line)
    rhs = re.sub(r"%$", "", rhs).strip()    # '50%' names the number 50, not 0.5
    if not _LITERAL.match(rhs.replace(" ", "")):
        try:                                # a non-literal RHS may be an exact expression
            coeffs, const = _lin(ast.parse(_normalize(rhs), mode="eval"))
            if not coeffs:
                return var, float(const), 9, line   # exact -> half-ULP of 9 shown decimals
        except (_NonLinear, SyntaxError, ValueError, ZeroDivisionError, OverflowError):
            pass                            # not a clean exact expression -> last-number path below
    nums = _NUM.findall(rhs) or _NUM.findall(line)
    if not nums:
        return None
    tok = nums[-1].replace("$", "").replace(",", "")
    ndec = len(tok.split(".")[1]) if "." in tok else 0
    return var, float(tok), ndec, line


def verdict(text):
    """('mismatch'|'derived'|'none', detail).
    mismatch -> the equations the model transcribed derive a DIFFERENT value than the
                answer it gave (HARD escalate: it mis-solved its own setup).
    derived  -> the transcribed system independently re-derives the model's answer
                (trust it — stronger than agreement, weaker than 'the setup is right').
    none     -> nothing derivable (no clean linear system, no pinned answer variable,
                or we can't tell WHICH variable the answer names) — fall through."""
    pairs = parse_equations(text)
    ans = parse_answer(text)
    if not pairs or ans is None:
        return "none", None
    solved = solve_system(pairs)
    if not solved:
        return "none", None
    var, claimed, ndec, line = ans
    if var is None and len(solved) == 1:
        var = next(iter(solved))            # only one unknown pinned -> unambiguous
    if var is None or var not in solved:
        return "none", None                 # can't say which quantity the answer names
    derived = solved[var]
    tol = 0.5 * (10 ** -ndec) + 1e-9        # half-ULP of the precision the answer shows
    detail = {"var": var, "derived": derived, "claimed": claimed, "answer": line,
              "eqns": len(pairs), "ok": abs(float(derived) - claimed) <= tol}
    return ("derived" if detail["ok"] else "mismatch"), detail


def fmt(fr):
    """Fraction -> friendly string: whole numbers bare, else a trimmed decimal."""
    if fr.denominator == 1:
        return str(fr.numerator)
    return f"{float(fr):.6f}".rstrip("0").rstrip(".")


if __name__ == "__main__":
    import sys
    t = " ".join(sys.argv[1:]) or sys.stdin.read()
    status, d = verdict(t)
    print("verdict:", status)
    if d:
        print(f"  {d['var']}: derived {fmt(d['derived'])} vs answered {d['claimed']}"
              f"  ({d['eqns']} eqn)")
