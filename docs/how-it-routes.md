# How hybrid routes

Back to the [README](../README.md).

## How it routes

```
query → router ─┬─ solve:    arithmetic · unit conversion · %-change? ▶ SOLVED  (python, exact, free)
                ├─ template: a word-problem SHAPE we recognize outright
                │              (rate×qty · bat-and-ball pair · reverse-% ·
                │               shift · price mix)? deterministic
                │              transcription + closed form ──────────▶ SOLVED  (python, exact, free)
                ├─ rule:     hard category (code/proof/puzzle) ───────▶ ESCALATE
                ├─ rule:     open-ended (rewrite/summarize) ──────────▶ LOCAL
                ├─ derive:   quantitative? the model transcribes the
                │              problem's relationships as EQUATIONS;
                │              we solve the linear system ourselves,
                │              exactly ─▶ LOCAL if it re-derives the model's answer,
                │                         ESCALATE if it contradicts it
                ├─ verify:   not derivable? local answers + plugs its
                │              numbers into the problem's relationships;
                │              re-derive each exactly ─▶ LOCAL if every check holds,
                │                                        ESCALATE if any is false
                └─ vote:     local self-consistency ─▶ LOCAL if unanimous, else ESCALATE
```

0. **Deterministic solver** (`solver.py`) answers what a cheap model gets *confidently
   wrong* — closed-form arithmetic, plus exact **unit conversions** (`3 miles → 15840 feet`,
   via `1 in = 25.4 mm` with `Fraction`s, never a float), **percentage-change**
   (`20% off 50 → 40`), and **multiples** (`half of 60 → 30`). Zero frontier calls, correct
   by construction. Strictly conservative — anything that doesn't reduce cleanly falls through.
1. **Template transcriber** (`templates.py`) — the derive tier's lesson,
   inverted. The model's real job on a word problem is *transcription*, and transcription
   is the one surface the exact oracle cannot check — so for the shapes that dominate
   everyday quantitative queries, don't ask the model at all. Five rigid shapes (rate ×
   quantity/time, total + gap pairs, reverse-percentage, plain shifts, two-price mixes) are
   parsed deterministically and solved in closed form over `Fraction`s: **zero tokens, zero
   latency, and the answer cannot be multiplied wrong** — the confident-wrong-product class
   the verifier used to have to *catch* is simply answered, exactly, for free. Stricter
   than any other tier about declining: every number in the query must be consumed by the
   shape, number-words ("half", "twice") anywhere else decline, nouns must agree between
   declaration and question, and set-logic riddles never match. It even out-ranks the
   hard-category rule — a clean exact parse beats a stray keyword.
2. **Category rules** escalate domains a small model is *known* to fail (code, proofs, puzzles).
3. **Open-ended rules** keep creative tasks (rewrite, summarize) local — no single right answer.
4. **Setup re-derivation** (`equations.py`) — for any *quantitative* query (a
   digit, or two number-words: "a chicken and **a half** lays an egg and **a half**…"), the
   model **transcribes the problem's relationships as equations** over named unknowns —
   transcription is an easier skill than solving — and we solve the linear system ourselves,
   by exact Gaussian elimination over `Fraction`s. An answer its own transcription contradicts
   is a **hard escalate** (the model mis-solved its own setup); a re-derived match stays local
   in *one* call where self-consistency needs three. Runs *before* plug-back because a
   derivation produces its **own** value instead of grading the model's checks — so a
   tautology can't fool it (live: the 7B "verified" its wrong Sally's-sisters answer with
   `CHECK: 3 + 3 - 1 = 5 / 2 * 2` — true, and disconnected from the problem). Strictly
   conservative: nonlinear, inconsistent, or underdetermined systems fall through rather
   than guess.
5. **Verify-the-local-answer** (`verify.py`) — when nothing was derivable, the local model
   answers and **plugs its own numbers back into the problem's relationships**, writing
   pure-numeric checks we re-derive exactly. A false check is a **hard escalate** (the answer
   is provably inconsistent with the problem); all-checks-hold stays local. Strictly stronger
   than self-consistency, which at temperature 0 just repeats the same wrong number.
6. **Self-consistency** for the rest: answer a few times — concurrently, so a batching
   server (Ollama with `OLLAMA_NUM_PARALLEL` ≥ 3) streams the weights once for all
   samples and the vote costs about one sample's wall time; unanimous → keep local,
   else escalate.
