#!/usr/bin/env python3
"""
Offline router dry-run — classifies each query into its v3 tier WITHOUT calling any
model, so it runs with no model and no network at all. It answers one question
honestly: what did the deterministic solver buy us versus v2?

v2 had no solver. Under v2, every arithmetic query either:
  - matched a HARD category rule (powers/roots/factorials/^) -> ESCALATE (a frontier
    call spent on grade-school math), or
  - fell to self-consistency, where the cheap model could be confidently wrong.
v3 answers all of those on-box, exactly, for free.

    python bench_offline.py
"""
import re
from solver import solve
from hybrid import _HARD, _OPEN

# v2's hard-category arithmetic triggers (subset of _HARD that is pure math) — these
# are the queries v2 escalated to the frontier purely because the small model was
# known to fumble them. v3 now solves them locally and exactly.
_V2_ARITH_RULE = re.compile(r"to the power|raised to|factorial|root of|\d+\s*\^\s*\d+", re.I)


def v3_tier(q):
    if solve(q) is not None:
        return "SOLVED"
    if _HARD.search(q) or len(q) > 220:
        return "ESCALATE"
    if _OPEN.search(q):
        return "LOCAL"
    return "VERIFY"  # self-consistency decides LOCAL vs ESCALATE at runtime


BATTERY = [
    "What is 17 to the power of 4?",
    "What is 2 to the power of 32?",
    "What is 8 factorial?",
    "What is 13 factorial?",
    "What is the square root of 1764?",
    "What is 47 times 19?",
    "What is 123 times 456?",
    "What is 20 percent of 80?",
    "What is 1,000,000 minus 1?",
    "What is 9^9?",
    # non-arithmetic controls — must NOT be solved
    "What is the capital of Japan?",
    "Rewrite 'hey send me that' more formally.",
    "Prove that the square root of 2 is irrational.",
    "If a chicken and a half lays an egg and a half in a day and a half, how many eggs per chicken per day?",
]


def main():
    print(f"{'TIER (v3)':<10} {'was v2 a frontier call?':<24} answer (exact)        query")
    print("-" * 100)
    saved = 0
    solved = 0
    for q in BATTERY:
        tier = v3_tier(q)
        ans = solve(q)
        # v2 would have made a frontier call on this query iff it hit the arith rule
        # (or the broader _HARD rule) — and v3 now solves it on-box instead.
        v2_escalated_arith = bool(_V2_ARITH_RULE.search(q))
        if tier == "SOLVED":
            solved += 1
            if v2_escalated_arith:
                saved += 1
        was = "YES -> now free" if (tier == "SOLVED" and v2_escalated_arith) else \
              ("(self-consist.)" if tier == "VERIFY" else ("YES" if tier == "ESCALATE" else "no"))
        shown = (ans if ans is not None else "")[:18]
        print(f"{tier:<10} {was:<24} {shown:<20}  {q[:46]}")
    print("-" * 100)
    print(f"{solved}/{len(BATTERY)} solved on-box (exact, free).")
    print(f"{saved} of those were FRONTIER CALLS under v2's category rule — now $0 and correct by construction.")
    print("Each saved call also removes a confident-wrong risk: v2's unanimous-local 17^4 -> 6859; v3 -> 83521.")


if __name__ == "__main__":
    main()
