#!/usr/bin/env python3
"""
Tests for the setup re-derivation oracle. It must:
  1. CATCH an answer the model's OWN transcribed equations contradict (the whole point).
  2. PASS an answer the transcribed system independently re-derives.
  3. Return 'none' — never guess — for anything nonlinear, inconsistent, underdetermined,
     or where we can't tell which quantity the answer names.

    python test_equations.py
"""
from fractions import Fraction

from equations import parse_equation, solve_system, verdict

# (response text, expected verdict)
CASES = [
    # ---- the headline: the model mis-solves its own correct transcription -> mismatch
    ("EQN: bat + ball = 1.10\nEQN: bat = ball + 1.00\nANSWER: ball = 0.10", "mismatch"),
    # same setup solved right -> derived
    ("EQN: bat + ball = 1.10\nEQN: bat = ball + 1.00\nANSWER: ball = 0.05", "derived"),
    # the recorded live symbolic shape (single unknown, self-referencing form)
    ("EQN: x + (x + 1.00) = 1.10\nANSWER: x = 0.10", "mismatch"),
    ("EQN: x + (x + 1.00) = 1.10\nANSWER: x = 0.05", "derived"),

    # ---- rate-shaped transcription (chicken-and-a-half): const*var*const stays linear
    ("EQN: 1.5 * r * 1.5 = 1.5\nANSWER: r = 0.5", "mismatch"),      # true r = 2/3
    ("EQN: 1.5 * r * 1.5 = 1.5\nANSWER: r = 0.67", "derived"),      # 2/3 to shown precision

    # ---- two unknowns, answer names its variable
    ("EQN: x + y = 10\nEQN: x - y = 4\nANSWER: x = 7", "derived"),
    ("EQN: x + y = 10\nEQN: x - y = 4\nANSWER: x = 5", "mismatch"),
    ("EQN: x + y = 10\nEQN: x - y = 4\nANSWER: y = 3", "derived"),
    # redundant third equation doesn't break the solve
    ("EQN: x + y = 10\nEQN: 2*x + 2*y = 20\nEQN: x - y = 4\nANSWER: x = 7", "derived"),

    # ---- surface forms the model actually writes
    ("EQN: bat + ball = $1.10\nEQN: bat = ball + $1.00\nANSWER: ball = $0.10", "mismatch"),
    ("EQN: total = 1,200 + 800\nANSWER: total = 2,000", "derived"),
    ("EQN: price - 20% * price = 40\nANSWER: price = 50", "derived"),   # reverse-% trap, right
    ("EQN: price - 20% * price = 40\nANSWER: price = 48", "mismatch"),  # the classic wrong 48
    ("EQN: 2x + 3 = 7\nANSWER: x = 2", "derived"),                      # implicit multiplication
    # untagged equations (model skipped the EQN: prefix) still parse
    ("bat + ball = 1.10\nbat = ball + 1.00\nANSWER: ball = 0.10", "mismatch"),
    # answer stated in prose, last number wins
    ("EQN: x + (x + 1.00) = 1.10\nANSWER: the ball costs $0.05", "derived"),
    # a restated ANSWER line: the last one is the commitment
    ("EQN: x + 2 = 5\nANSWER: x = 4\nANSWER: x = 3", "derived"),

    # ---- conservative refusals -> 'none'
    ("EQN: x * y = 6\nANSWER: x = 2", "none"),               # nonlinear
    ("EQN: 1.5 chickens * rate * 1.5 = 1.5\nANSWER: rate = 0.5",
     "none"),  # a unit word becomes a variable -> chickens*rate is nonlinear -> refuse, don't guess
    ("EQN: 60 / t = 30\nANSWER: t = 2", "none"),             # divide by variable
    ("EQN: x = 1\nEQN: x = 2\nANSWER: x = 1", "none"),       # self-contradictory transcription
    ("EQN: x + y = 10\nANSWER: x = 3", "none"),              # underdetermined
    ("EQN: x + y = 10\nEQN: x - y = 4\nANSWER: 7", "none"),  # two pinned vars, unnamed answer
    ("EQN: x + 2 = 5\nANSWER: x = 3\nno wait", "derived"),   # trailing prose is fine
    ("ANSWER: x = 3", "none"),                               # no equations at all
    ("EQN: x + 2 = 5", "none"),                              # no answer line
    ("EQN: x + 2 = 5\nANSWER: three", "none"),               # answer has no number
    ("Tokyo is the capital.", "none"),                       # nothing to do
    ("EQN: x ** 20 = 5\nANSWER: x = 1", "none"),             # runaway exponent refused
    ("EQN: profit margin = 0.2\nANSWER: 0.2", "none"),       # two adjacent names don't parse

    # ---- tolerance: half-ULP of the precision the answer shows
    ("EQN: 3 * x = 1\nANSWER: x = 0.33", "derived"),         # 1/3 at two decimals
    ("EQN: 3 * x = 1\nANSWER: x = 0.34", "mismatch"),        # off by more than a half-ULP
    ("EQN: 2 * x = 5\nANSWER: x = 2.5", "derived"),
    # an exact-EXPRESSION answer is matched tightly, no rounding slack
    ("EQN: 3 * x = 2\nANSWER: x = 2/3", "derived"),
    ("EQN: 3 * x = 2\nANSWER: x = 1/2", "mismatch"),
    ("EQN: x = 100 - 50\nANSWER: x = 50%", "derived"),       # '50%' names 50, not 0.5

    # ---- live transcripts, pinned verbatim (qwen2.5:7b on-box, 2026-07-01)
    # chicken-and-a-half: the setup prompt acts as chain-of-thought — the model gets the
    # rate RIGHT (2/3) — and it answers in LaTeX, which must parse, not fall through
    # (naive last-number parsing of \frac{2}{3} would read '3' and false-mismatch).
    ("Let's define the variables first:\n\n"
     "- Let \\( C \\) be the number of eggs one chicken lays in one day.\n\n"
     "EQN: \\( 1.5 \\times C \\times 1.5 = 1.5 \\)\n\n"
     "Now, we solve for \\( C \\):\n\n"
     "\\[ 2.25C = 1.5 \\]\n"
     "ANSWER: \\( C = \\frac{2}{3} \\)", "derived"),
    # same transcription with the classic WRONG answer must be a catch
    ("EQN: \\( 1.5 \\times C \\times 1.5 = 1.5 \\)\nANSWER: \\( C = \\frac{1}{2} \\)",
     "mismatch"),
    # Sally's-sisters: the model's own transcription (S = 3*2) contradicts its answer (3)
    # -> mismatch -> escalate. The trap escalates instead of being served wrong.
    ("EQN: S = 3 * 2\n\nANSWER: S = 3", "mismatch"),
    # controls that must NOT false-mismatch: fully symbolic (no numbers to pin) -> none
    ("EQN: COST = PRICE * QUANTITY\n\nANSWER: COST = 87.50", "none"),
    ("EQN: S = P * (1 - 0.20)\n\nANSWER: P = 50", "none"),   # underdetermined -> none
    # ages control: implicit multiplication '3J', two equations, exact solve
    ("EQN: T = 3J  \nEQN: T + J = 48\n\nANSWER: J = 12", "derived"),
]


def test_cases():
    bad = []
    for text, want in CASES:
        got, _ = verdict(text)
        if got != want:
            bad.append((text.replace("\n", " | "), want, got))
    return bad


def test_exactness():
    """The solve is Fraction-exact, not float-approximate."""
    pairs = [parse_equation("bat + ball = 1.10"), parse_equation("bat = ball + 1.00")]
    sol = solve_system(pairs)
    assert sol["ball"] == Fraction(1, 20), sol
    assert sol["bat"] == Fraction(21, 20), sol
    pairs = [parse_equation("1.5 * r * 1.5 = 1.5")]
    assert solve_system(pairs)["r"] == Fraction(2, 3)


def test_partial_determination():
    """A variable the system pins is reported even when another stays free."""
    pairs = [parse_equation("x + y = 10"), parse_equation("y = 3")]
    sol = solve_system(pairs)
    assert sol == {"x": Fraction(7), "y": Fraction(3)}, sol
    pairs = [parse_equation("x + y + z = 10"), parse_equation("y = 3")]
    sol = solve_system(pairs)
    assert sol == {"y": Fraction(3)}, sol  # x, z free -> absent, not guessed


if __name__ == "__main__":
    failures = test_cases()
    test_exactness()
    test_partial_determination()
    for text, want, got in failures:
        print(f"FAIL  want {want:<9} got {got:<9}  {text[:90]}")
    total = len(CASES) + 2
    print(f"{total - len(failures)}/{total} passed" + ("" if not failures else "  <-- FAILURES"))
    raise SystemExit(1 if failures else 0)
