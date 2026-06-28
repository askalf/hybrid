#!/usr/bin/env python3
"""
Tests for the answer verifier. It must:
  1. CATCH false arithmetic the local model asserts (the whole point).
  2. PASS correct arithmetic, including legitimate rounding.
  3. IGNORE prose that merely contains '=' (no false "wrong" on non-math).

    python test_verify.py
"""
from verify import verdict, strip_calc, answer_text

# (answer text, expected verdict)
CASES = [
    # correct computations -> 'checked'
    ("21 eggs.\nCALC: 3 * 7 = 21", "checked"),
    ("The total is $87.50.\nCALC: 7 * 12.50 = 87.50", "checked"),
    ("It travels 210 miles.\nCALC: 60 * 3.5 = 210", "checked"),
    ("1024.\nCALC: 2 ^ 10 = 1024", "checked"),          # '^' read as power, not xor
    ("About 0.33.\nCALC: 1 / 3 = 0.33", "checked"),     # legitimate rounding passes
    ("1,000,000.\nCALC: 1000 * 1000 = 1,000,000", "checked"),  # commas in result

    # false computations -> 'wrong' (the catches self-consistency would miss)
    ("$85.00.\nCALC: 7 * 12.50 = 85.00", "wrong"),      # true 87.50
    ("813.\nCALC: 47 * 19 = 813", "wrong"),             # true 893
    ("The answer is 408.\nCALC: 17 * 24 = 388", "wrong"),  # true 408 (result line wrong)
    ("6859.\nCALC: 17 ^ 4 = 6859", "wrong"),            # the classic: true 83521

    # no checkable arithmetic -> 'none' (fall back to self-consistency)
    ("Tokyo.", "none"),
    ("The capital is Paris.", "none"),
    ("Photosynthesis converts light into chemical energy.", "none"),
    ("The answer = Tokyo", "none"),                     # prose '=' ignored
    ("Use version 3 = the stable release", "none"),     # not an equation
    ("I recommend option 5 = best value", "none"),      # RHS not a clean number op

    # CONSTRAINT check (v3.2): substitute the answer's VARS into the problem's CHECK
    # relationships. Catches a wrong *solve* of a right *setup* — what CALC can't.
    # The bat-and-ball trap, answered wrong (ball=0.10): no assignment with ball=0.10
    # satisfies both constraints, so a faithful transcription of them is caught.
    ("The ball costs $0.10.\nVARS: ball = 0.10; bat = 1.10\n"
     "CHECK: bat + ball = 1.10\nCHECK: bat - ball = 1.00", "wrong"),
    # the same problem solved correctly (ball=0.05) satisfies both -> stays local
    ("The ball costs $0.05.\nVARS: ball = 0.05; bat = 1.05\n"
     "CHECK: bat + ball = 1.10\nCHECK: bat - ball = 1.00", "checked"),
    # the operation may sit on the right of the '=' too
    ("VARS: x = 4\nCHECK: 12 = 3 * x", "checked"),
    ("VARS: x = 5\nCHECK: 12 = 3 * x", "wrong"),
    # mixed: a true constraint AND a false one -> wrong (the false one wins)
    ("VARS: a = 2; b = 3\nCHECK: a + b = 5\nCHECK: a * b = 7", "wrong"),
    # the HONEST LIMIT: a self-consistent but wrong *setup* passes (we verify the
    # answer against the stated relationships, not the relationships themselves).
    ("Distance 60 km.\nVARS: d = 60\nCHECK: d = 30 * 2", "checked"),
    # a CHECK that doesn't reduce to clean arithmetic (undefined var) is ignored, not 'wrong'
    ("VARS: ball = 0.05\nCHECK: bat + ball = 1.10", "none"),

    # numeric plug-back (how the 7B actually verifies): it substitutes its OWN numbers into
    # the problem's relationships, yielding pure-numeric CHECK lines.
    # the RHS of a CHECK may itself be an expression — it must be parsed whole, never
    # prefix-matched (regression: `= 3 * (10 - 5)` must not be read as `= 3`).
    ("Anna is 10.\nCHECK: 10 + 5 = 3 * (10 - 5)", "checked"),    # 15 == 15
    ("Anna is 8.\nCHECK: 8 + 5 = 3 * (8 - 5)", "wrong"),         # 13 != 9
    # the bat-and-ball CAUGHT via plug-back: the trap answer ($0.10) put into the problem's
    # total constraint fails it — the catch the bare CALC check and self-consistency miss.
    ("The ball is $0.10.\nCHECK: 0.10 + 1.10 = 1.10\nCHECK: 1.10 - 0.10 = 1.00", "wrong"),
    ("The ball is $0.05.\nCHECK: 0.05 + 1.05 = 1.10\nCHECK: 1.05 - 0.05 = 1.00", "checked"),
    # units/letters in a CHECK -> unparseable -> ignored, not a false 'wrong' (the
    # chicken-and-a-half honest limit: the model's check carries units we can't evaluate)
    ("One egg.\nCHECK: (1.5 chickens) * (1.5 eggs) / (1.5 days) = 1 egg", "none"),
]


def main():
    fails = []
    for text, want in CASES:
        got, claims = verdict(text)
        ok = got == want
        oneline = text.replace(chr(10), " ")[:50]
        print(f"{'ok ' if ok else 'XX '} {got:<8} (want {want:<8}) {oneline}")
        if not ok:
            fails.append((text, got, want, claims))

    # strip_calc keeps the answer, drops the working line
    s = strip_calc("The total is $87.50.\nCALC: 7 * 12.50 = 87.50")
    assert s == "The total is $87.50.", f"strip_calc -> {s!r}"
    print(f"ok  strip_calc -> {s!r}")

    # answer_text prefers a tagged ANSWER: line, else falls back to stripped prose
    a = answer_text("ANSWER: 0.05\nCHECK: 0.05 + 1.05 = 1.10")
    assert a == "0.05", f"answer_text(tagged) -> {a!r}"
    a = answer_text("The ball is $0.05.\nCHECK: 0.05 + 1.05 = 1.10")
    assert a == "The ball is $0.05.", f"answer_text(prose) -> {a!r}"
    print("ok  answer_text -> tagged + prose")

    print("-" * 64)
    if fails:
        print(f"FAIL {len(fails)}/{len(CASES)}")
        for text, got, want, claims in fails:
            print(f"   {text!r}  got {got} want {want}  claims={claims}")
        raise SystemExit(1)
    print(f"PASS {len(CASES)}/{len(CASES)} + strip_calc")


if __name__ == "__main__":
    main()
