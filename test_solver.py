#!/usr/bin/env python3
"""
Tests for the deterministic arithmetic solver.

Three things must hold:
  1. CORRECT  — it computes closed-form arithmetic exactly (incl. the cases the
                self-consistency router got confidently wrong).
  2. CONSERVATIVE — it returns None for anything that isn't unambiguously a
                closed-form computation. A false "solved" is worse than escalating.
  3. SAFE     — pathological inputs are refused, not evaluated.

    python test_solver.py
"""
from solver import solve

# (query, expected exact answer) — the solver MUST get these and they must be free.
SOLVES = [
    # the headline confident-wrong case: unanimous-local said 6859; truth is 83521
    ("What is 17 to the power of 4?", "83521"),
    ("17^4", "83521"),
    ("2 to the power of 10", "1024"),
    ("What is 47 times 19?", "893"),
    ("What is 8 factorial?", "40320"),
    ("factorial of 6", "720"),
    ("5!", "120"),
    ("What is the square root of 144?", "12"),
    ("sqrt of 169", "13"),
    ("12 squared", "144"),
    ("3 cubed", "27"),
    ("What is 1,234 plus 5,678?", "6912"),
    ("100 divided by 4", "25"),
    ("(3 + 4) * 5", "35"),
    ("2 cubed plus 3", "11"),
    ("What is 20 percent of 80?", "16"),
    ("15% of 200", "30"),
    ("17 mod 5", "2"),
    ("9 times 9 times 9", "729"),
    ("1000000 times 1000000", "1000000000000"),  # exact big-int, no float error

    # v3.3 — unit conversion (exact Fractions; correct by construction)
    ("How many days are in 3 weeks?", "21"),
    ("How many feet in 3 miles?", "15840"),
    ("Convert 90 minutes to seconds", "5400"),
    ("5 km in meters", "5000"),
    ("How many seconds in a minute?", "60"),         # "a" -> 1
    ("How many ounces in 2 pounds?", "32"),
    # v3.3 — multiples / fractions of a quantity
    ("What is half of 60?", "30"),
    ("double 21", "42"),
    ("triple 7", "21"),
    ("a quarter of 80", "20"),
    # v3.3 — percentage change
    ("What is 20% off 50?", "40"),
    ("20% more than 80", "96"),
    ("increase 200 by 15%", "230"),
    ("reduce 90 by 10%", "81"),
]

# non-arithmetic / ambiguous -> MUST decline (None). Conservativeness.
DECLINES = [
    "What is the capital of Japan?",
    "Define photosynthesis in one sentence.",
    "Rewrite 'hey can u send me that file' more formally.",
    "Prove that the square root of 2 is irrational.",
    "Write a Python function that returns the longest palindromic substring.",
    "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball.",
    "If a chicken and a half lays an egg and a half in a day and a half...",
    "What is the population of China times 2?",   # has 'times 2' but isn't closed-form
    "How many miles in 5 kilograms?",              # cross-dimension -> refuse to convert
    "How many widgets in 3 boxes?",                # unknown units -> decline
    "What time is it?",
    "",
]

# pathological -> MUST refuse (None), never actually evaluate.
GUARDS = [
    "9 ** 9 ** 9 ** 9",        # astronomically large -> refuse
    "100000 factorial",        # over factorial ceiling -> refuse
    "__import__('os')",        # injection attempt -> refuse
    "2 ** 99999",              # over digit ceiling -> refuse
]


def main():
    fails = []

    for q, want in SOLVES:
        got = solve(q)
        ok = got == want
        print(f"{'ok ' if ok else 'XX '} solve   {q[:46]:<46} -> {got!r}  (want {want!r})")
        if not ok:
            fails.append((q, got, want))

    for q in DECLINES:
        got = solve(q)
        ok = got is None
        print(f"{'ok ' if ok else 'XX '} decline {q[:46]:<46} -> {got!r}")
        if not ok:
            fails.append((q, got, None))

    for q in GUARDS:
        got = solve(q)
        ok = got is None
        print(f"{'ok ' if ok else 'XX '} guard   {q[:46]:<46} -> {got!r}")
        if not ok:
            fails.append((q, got, None))

    total = len(SOLVES) + len(DECLINES) + len(GUARDS)
    print("-" * 72)
    if fails:
        print(f"FAIL  {len(fails)}/{total} cases wrong")
        for q, got, want in fails:
            print(f"   {q[:50]!r}  got {got!r}  want {want!r}")
        raise SystemExit(1)
    print(f"PASS  {total}/{total}")


if __name__ == "__main__":
    main()
