# The honest part

Back to the [README](../README.md).

## The honest part (what this taught me)

The interesting finding isn't that it works — it's *where the routing fails*, and how far a
free verifier can move that line.

**A cheap router inherits the cheap model's blind spots.** Self-consistency (answer a few
times, escalate on disagreement) catches genuine *uncertainty* but **cannot catch confident
wrongness** — a small model states `17⁴ = 6859` *unanimously* (it's 83,521). The escapes are
category rules for known-weak domains, or **a verifier stronger than the model.** For the
huge closed-form-and-arithmetic slice, the strongest possible verifier is *free*: Python's
exact arithmetic. So the solver answers closed-form math outright, and the verify tier has
the model plug its numbers back into the problem and re-derives them — catching confident-wrong
*embedded* arithmetic (live: 5/6 ugly products) that self-consistency waves through.

**The line moves — v1.1.0 cracked v1.0.0's documented traps.** v1.0.0 shipped with two
setup traps served locally and wrong, kept visible in the benchmark as the honest limit.
The setup re-derivation tier moved both: Sally's-sisters is **caught** (the model's own
transcription `S = 3 * 2` contradicts its answer), and chicken-and-a-half comes back
**right and verified** — the equation prompt doubles as chain-of-thought, so the model
writes the rate correctly and we re-derive `2/3` exactly.

**What still gets through — kept visible, not papered over.** The oracle solves the system
the model *transcribes*; it cannot check the transcription against the *problem*. A
misconception that leaks *into* the equations — a wrong rate written as if the problem
stated it — re-derives the same wrong answer and sails through. And only linear systems are
in reach: set-logic riddles and nonlinear setups fall through (conservative) rather than
guess. So a passed derivation is labelled **"setup re-derived," never "correct."** We even
tested the obvious cheap escape — a *second* small model as an independent vote — and it
shares the classic blind spots (both models miss the same famous traps) while over-escalating
when the weaker one is merely vaguer. A second cheap model is still a cheap-model signal.
Cracking a faithfully-mis-transcribed setup needs a stronger *reasoner* (a frontier call) —
the line keeps moving; it doesn't disappear.
