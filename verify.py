#!/usr/bin/env python3
"""
Verify-the-local-answer: deterministically re-check what a local answer commits to.

Two free oracles, one router tier:

  (1) Computation check — the model states its calculation as `CALC: <expr> = <result>`
      and we re-derive <expr> exactly. A false equation is the model's own stated math
      being wrong:  "$85 ... CALC: 7 * 12.50 = 85"  ->  7*12.50 is 87.5  ->  WRONG.

  (2) Constraint check (v3.2) — for a word problem the model states its answer as
      variable VALUES plus the problem's RELATIONSHIPS, and we substitute the values
      back into every relationship:

          VARS:  ball = 0.10 ; bat = 1.10
          CHECK: bat + ball = 1.10      ->  1.10 + 0.10 = 1.20 != 1.10   ->  WRONG
          CHECK: bat - ball = 1.00      ->  1.10 - 0.10 = 1.00 == 1.00   ->  ok

      No assignment with ball = 0.10 satisfies both constraints, so the trap answer is
      caught — deterministically, no frontier call. This moves the line the solver and
      the bare CALC check could not reach: it catches a wrong *solve* of a right *setup*.

A detected mismatch is a HARD escalate signal (the answer is provably inconsistent with
the problem the model itself transcribed), not a vote — strictly stronger than
self-consistency.

Honest boundary: the check is only as good as the constraints the model transcribes. If
the model misreads the problem and writes a wrong *relationship* (a rate it never grasped),
a self-consistent-but-wrong answer passes. We verify the answer against the stated setup;
we cannot verify the setup. So a pass is labelled "constraints hold", never "correct".

    from verify import verdict
    verdict("$0.10\nVARS: ball=0.10; bat=1.10\nCHECK: bat+ball=1.10")  # -> ("wrong", [...])
    verdict("21 eggs.\nCALC: 3 * 7 = 21")                              # -> ("checked", [...])
"""
import re
from solver import eval_expr

# A bare "<arith expr> = <number>" anywhere (e.g. an appended CALC). LHS is digit-anchored
# so prose like "the answer = Tokyo" or "version 3 = stable" never matches.
_EQ = re.compile(r"([\d][\d\s+\-*/^().,$]*?)\s*=\s*([-+]?\$?\d[\d,]*(?:\.\d+)?\s*%?)")

# The working lines the prompt asks for. CALC/CHECK hold equations (which may name
# variables); VARS holds the variable values, parsed apart by parse_vars().
_WORK_LINE = re.compile(r"(?im)^\s*(?:check|calc|verify)\s*:\s*(.+?)\s*$")
_VARS_LINE = re.compile(r"(?im)^\s*vars?\s*:\s*(.+?)\s*$")
_NAME = re.compile(r"[a-z][a-z0-9_]*", re.I)


def _claimed(tok):
    """Parse an asserted numeric token -> (value, decimals-shown). Handles $, commas, %."""
    t = tok.replace("$", "").replace(" ", "")
    pct = t.endswith("%")
    t = t.rstrip("%").replace(",", "")
    try:
        v = float(t)
    except ValueError:
        return None, 0
    ndec = len(t.split(".")[1]) if "." in t else 0
    return (v / 100 if pct else v), (ndec + 2 if pct else ndec)


def parse_vars(text):
    """Parse a 'VARS: a = 1; b = 2.5' block into {name: value}. Empty dict if none."""
    out = {}
    for line in _VARS_LINE.findall(text):
        for part in re.split(r"[;,]", line):
            m = re.match(r"\s*([a-z][a-z0-9_]*)\s*=\s*(.+)", part, re.I)
            if not m:
                continue
            val, _ = _claimed(m.group(2).strip())
            if val is not None:
                out[m.group(1).lower()] = val
    return out


def _subst(expr, vars):
    """Replace whole-word variable names with their values, parenthesized so a negative
    value can't fuse with a preceding operator (`a - -0.1` would mis-parse)."""
    return _NAME.sub(
        lambda m: f"({vars[m.group(0).lower()]!r})" if m.group(0).lower() in vars else m.group(0),
        expr)


def _shown_decimals(s):
    """Most decimal places shown by any number literal in s — sets the match tolerance."""
    return max([len(d) for d in re.findall(r"\d+\.(\d+)", s)] + [0])


def _candidates(text):
    """Yield (lhs, rhs) for every checkable equation: explicit CALC/CHECK lines first
    (parsed whole — the RHS may itself be an expression, e.g. `10+5 = 3*(10-5)`), then
    any bare digit-anchored equation in the *remaining* prose. The work lines are masked
    out before the bare scan so the digit-anchored _EQ can't prefix-match a CHECK's
    expression RHS (`= 3 * (10 - 5)` must not be read as `= 3`)."""
    seen = set()
    for content in _WORK_LINE.findall(text):
        if content.count("=") != 1:
            continue
        lhs, rhs = (s.strip() for s in content.split("="))
        key = (lhs.lower(), rhs.lower())
        if key not in seen:
            seen.add(key)
            yield lhs, rhs
    for m in _EQ.finditer(_WORK_LINE.sub(" ", text)):
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        key = (lhs.lower(), rhs.lower())
        if key not in seen:
            seen.add(key)
            yield lhs, rhs


def check_claims(text):
    """[{expr, claimed, actual, ok}] for every equation/constraint the answer asserts.

    A VARS block is substituted first, so a CHECK line transcribed from the problem's
    relationships is tested against the model's own answer values. Each side is then
    evaluated exactly; a claim is `ok` iff the two sides agree to the precision shown —
    so 1/3 = 0.33 passes, but the bat-and-ball's `bat + ball = 1.10` under ball=0.10
    does not."""
    vars = parse_vars(text)
    out = []
    for lhs, rhs in _candidates(text):
        if not re.search(r"[+\-*/^]", lhs + " " + rhs):   # a real operation, not "5 = 5"
            continue
        actual = eval_expr(_subst(lhs, vars))             # left side, exact
        claimed = eval_expr(_subst(rhs, vars))            # right side, exact
        if actual is None or claimed is None:             # a side isn't clean arithmetic
            continue
        ndec = _shown_decimals(lhs + " " + rhs)
        tol = 0.5 * (10 ** -ndec) + 1e-9                  # half-ULP of the shown precision
        out.append({"expr": lhs, "claimed": claimed, "actual": actual,
                    "ok": abs(actual - claimed) <= tol})
    return out


def verdict(text):
    """('wrong'|'checked'|'none', claims).
    wrong  -> at least one asserted equation/constraint is false (HARD escalate).
    checked-> >=1 equation, all hold (trust the local answer's computation + setup).
    none   -> nothing checkable (fall back to self-consistency)."""
    claims = check_claims(text)
    if not claims:
        return "none", claims
    return ("wrong" if any(not c["ok"] for c in claims) else "checked"), claims


def has_constraint(claims):
    """True if any verified claim referenced a variable (a CHECK), not just bare arithmetic
    — lets the caller label the route 'constraint' vs 'arithmetic'."""
    return any(re.search(r"[a-z]", c["expr"], re.I) for c in claims)


def strip_calc(text):
    """Drop the working (VARS/CHECK/CALC lines, or a trailing CALC tail) for a clean
    user-facing answer."""
    text = re.sub(r"(?im)^\s*(?:vars?|check|calc|verify)\s*:.*$", "", text)
    text = re.sub(r"\s*\bcalc\b\s*:.*$", "", text, flags=re.I | re.S)
    return text.strip()


def answer_text(raw):
    """The user-facing answer from a verify response: the `ANSWER:` line if the model
    tagged one, else the prose with the working lines stripped off."""
    m = re.search(r"(?im)^\s*answer\s*:\s*(.+?)\s*$", raw)
    if m and m.group(1).strip():
        return m.group(1).strip()
    return strip_calc(raw)


if __name__ == "__main__":
    import sys
    t = " ".join(sys.argv[1:]) or sys.stdin.read()
    status, claims = verdict(t)
    print("verdict:", status)
    for c in claims:
        mark = "ok " if c["ok"] else "XX "
        print(f"  {mark} {c['expr']} = {c['claimed']}   (your values give: {c['actual']})")
