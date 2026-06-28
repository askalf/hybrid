#!/usr/bin/env python3
"""
Quantify the whole router on a labeled set — the thesis in numbers.

Runs the REAL routing decision over labeled queries and reports the metrics that
matter for "answer the easy majority on-box, safely":
  - ON-BOX RATE   — fraction answered without a frontier call (SOLVED + LOCAL).
  - ON-BOX SAFETY — of those on-box answers, how many are CORRECT. Serving a wrong
                    answer locally is the cardinal sin; this is the number to watch.
  - CATCHES       — confident-wrong answers the verifier intercepted (would-escalate).
  - HONEST LIMIT  — setup traps that slip through local + wrong (the known boundary).

Frontier escalation is STUBBED (we record "would escalate", never call out), so the
bench costs nothing at the frontier. It still calls the local model, so it needs a
local Ollama serving LOCAL_MODEL. Numbers below were measured on qwen2.5:7b; a smaller
model follows the CHECK format less reliably, so expect a lower on-box rate.

    python bench_router.py
"""
import re
import hybrid

# Don't call the frontier — record the decision, return a marker.
hybrid.escalate = lambda q: ("[would escalate -> frontier]", 0.0)

# (query, ground-truth answer or None if it SHOULD escalate, category)
#   category: solved | factual | wordprob | catch | hard | trap
CASES = [
    # closed-form arithmetic + the widened oracle -> SOLVED, exact, free
    ("What is 47 times 19?", "893", "solved"),
    ("What is 17 to the power of 4?", "83521", "solved"),
    ("What is 8 factorial?", "40320", "solved"),
    ("How many feet in 3 miles?", "15840", "solved"),
    ("How many seconds in 90 minutes?", "5400", "solved"),
    ("What is 20% off 50?", "40", "solved"),
    ("What is half of 60?", "30", "solved"),
    # factual / open -> LOCAL
    ("What is the capital of Japan?", "tokyo", "factual"),
    ("What is the capital of France?", "paris", "factual"),
    # word problems the local model solves + self-verifies -> LOCAL (checks hold)
    ("A store sells notebooks at $12.50 each. How much do 7 notebooks cost?", "87.5", "wordprob"),
    ("A bat and a ball cost $1.10; the bat is $1.00 more than the ball. How much is the ball?", "0.05", "wordprob"),
    ("A shirt costs $40 after a 20% discount. What was the original price?", "50", "wordprob"),
    ("A number increased by 30% is 78. What is the number?", "60", "wordprob"),
    # confident-wrong embedded arithmetic -> should be CAUGHT (verify -> escalate)
    ("A factory makes 1,847 widgets per day. How many widgets in 263 days?", "485761", "catch"),
    ("A server processes 3,408 requests per minute. How many in 47 minutes?", "160176", "catch"),
    ("Each shipping container holds 1,728 units. How many units in 56 containers?", "96768", "catch"),
    # known-hard -> ESCALATE by rule
    ("Prove that the square root of 2 is irrational.", None, "hard"),
    ("Write a Python function that returns the longest palindromic substring.", None, "hard"),
    # setup traps -> the honest limit (may slip through local + wrong)
    ("If a chicken and a half lays an egg and a half in a day and a half, how many eggs does one chicken lay in one day?", "0.67", "trap"),
    ("Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have?", "1", "trap"),
]


def _num(s):
    m = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return float(m[-1].replace(",", "")) if m else None


def correct(answer, truth):
    """Did the on-box answer match ground truth? Numeric -> value within 0.5%/0.01;
    textual -> keyword appears."""
    if truth is None:
        return None
    tn = _num(truth)
    if tn is not None:
        an = _num(answer)
        return an is not None and abs(an - tn) <= max(0.01, abs(tn) * 0.005)
    return truth.lower() in answer.lower()


def main():
    onbox = served_wrong = caught = trap_miss = escalated = 0
    for q, truth, cat in CASES:
        r = hybrid.route(q)
        route, ans = r["route"], r["answer"]
        is_onbox = route in ("SOLVED", "LOCAL")
        ok = correct(ans, truth) if is_onbox else None
        if is_onbox:
            onbox += 1
            if truth is not None and ok is False:
                served_wrong += 1
        else:
            escalated += 1
            if cat == "catch":
                caught += 1
        if cat == "trap" and is_onbox and ok is False:
            trap_miss += 1
        flag = {True: "ok", False: "WRONG-SERVED", None: "(escalated)"}[ok]
        print(f"{cat:<9}{route:<9}{flag:<14}{q[:44]:<44} -> {ans[:30].replace(chr(10),' ')}")

    n = len(CASES)
    print("-" * 100)
    print(f"ON-BOX:        {onbox}/{n} ({100*onbox//n}%) answered without a frontier call")
    print(f"ON-BOX SAFETY: {onbox-served_wrong}/{onbox} on-box answers correct"
          f"  ({served_wrong} wrong answer(s) served locally)")
    print(f"CATCHES:       {caught}/3 confident-wrong arithmetic intercepted -> escalated")
    print(f"ESCALATED:     {escalated}/{n} routed to the frontier")
    print(f"HONEST LIMIT:  {trap_miss} setup trap(s) slipped through local + wrong (known boundary)")


if __name__ == "__main__":
    main()
