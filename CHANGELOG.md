# Changelog

All notable changes to hybrid are documented here. This project adheres to
[Semantic Versioning](https://semver.org).

## v1.2.0 — 2026-07-02

Production hardening, part 1: the router now fails predictably, and the routing logic
itself — not just the oracles — is under test and CI.

### Failure policy

- `ollama()` and `escalate()` raise a typed `BackendError` after retries instead of
  crashing the query (local transport now retries once) — and an escalation failure is
  no longer returned *as the answer string*: an error is always a structured
  `route: ERROR` result, never answer-shaped text a caller could mistake for content.
- `route()` never raises for a dead backend. Policy via env, read per-call:
  - `HYBRID_ON_LOCAL_FAIL` — `escalate` (default): local model unreachable → send the
    query to the frontier; or `error`.
  - `HYBRID_ON_FRONTIER_FAIL` — `error` (default, honest): a query the router decided
    needs the frontier fails explicitly rather than getting a silent local answer; or
    `local`: availability-over-correctness, a plain local answer labelled `DEGRADED`
    (including for queries whose local answer the verifier just refuted — documented,
    opt-in).
- `--demo` reports backend errors in their own `ERRORS` bucket instead of folding them
  into escalations.

### Tests + CI

- `test_route.py` (new, 16 tests) — the router *plumbing* finally has coverage: tier
  order (solve short-circuits; rules before oracles; derive before plug-back; vote
  last), the quantitative gate (factual queries skip the oracle tiers, number-word
  setups reach them), verdict→route mapping (mismatch/false-check escalate, exact
  values served), and the full failure-policy matrix — all offline via scripted fake
  backends.
- GitHub Actions CI (`.github/workflows/test.yml`): compile + all 142 offline tests on
  Python 3.10–3.13, every push and PR. Actions pinned by commit SHA. Badge in README.

## v1.1.1 — 2026-07-02

Conservatism fix in the conversion oracle, found by adversarial stress testing against
the live router.

- **Mixed-unit quantities now decline.** "Convert 5 feet 4 inches to centimeters" was
  answered `10.16` by the SOLVED tier — the conversion pattern matched the `4 inches`
  pair and silently dropped the `5 feet`. A wrong answer served with *correct by
  construction* confidence is the exact failure the deterministic tier exists to prevent.
  Any query with more than one number-carrying known unit now returns None and falls
  through — where, live, the derive tier picks it up: the model transcribes
  `5*30.48 + 4*2.54` and the system re-derives **162.56** exactly. The layered design
  turned a wrong answer into a verified correct one.
- `test_solver.py` 50/50 → **53/53** with the mixed-unit regressions pinned
  (`_CONV1` had the same flaw; both patterns are behind the guard).

## v1.1.0 — 2026-07-02

The setup re-derivation tier — v1.0.0's documented open problem, moved.

### Router

- **DERIVE** (`equations.py`) — for quantitative queries, the local model transcribes the
  problem's *stated relationships* as equations over named unknowns (transcription is an
  easier skill than solving), and hybrid solves the linear system itself — exact Gaussian
  elimination over `Fraction`s, no floats, no model — then compares the derived value
  against the answer the model committed to. A contradiction is a **hard escalate**: the
  model mis-solved its own setup. A re-derived match is served locally in *one* call where
  self-consistency needs three. Runs before plug-back because a derivation produces its
  own value instead of grading the model's checks, so a true-but-disconnected
  (tautological) check can't fool it.
- The quantitative gate widened from "has a digit" to *digits, or two number-words* — so
  worded setups ("a chicken and **a half** lays an egg and **a half**…") reach the oracle
  tiers instead of falling to the vote.

### Measured (qwen2.5:7b, same 20-query labeled set, `bench_router.py`)

- **On-box safety 14/14 — zero wrong answers served locally.** v1.0.0 was 13/15: both
  documented setup traps were served locally and wrong.
- **Sally's-sisters is caught** — the model's own transcription contradicts its answer —
  and escalated. **Chicken-and-a-half now comes back right *and* verified**: the equation
  prompt doubles as chain-of-thought, the model writes the rate correctly, and the system
  re-derives `2/3` exactly.
- On-box rate 14/20 (70%, was 75%): the one extra escalation is a query v1.0.0 answered
  *wrong*. Trading one on-box point for zero-wrong-served is the point of the router.
- 3/3 confident-wrong products still caught and escalated.

### Tooling

- `test_equations.py` — 45/45 covering catch/pass/decline, LaTeX and symbolic shapes,
  inconsistent and underdetermined systems; like every oracle test, no model or network
  needed.

### The limit that remains

The oracle solves the system the model *transcribes*; it cannot check the transcription
against the *problem*. A misconception written into the equations as if the problem stated
it re-derives the same wrong answer. Only linear systems are in reach — set-logic riddles
and nonlinear setups fall through (conservative) rather than guess. A passed derivation is
labelled "setup re-derived," never "correct."

## v1.0.0 — 2026-06-28

First public release: a local-first LLM router with a free verifier stronger than the
model — Python's exact arithmetic — applied at two depths.

### Router

- **SOLVED** — a deterministic solver (`solver.py`) answers closed-form arithmetic, plus
  exact unit conversions (`Fraction`s, so `1 in = 25.4 mm` never drifts), percentage-change,
  and multiples — on-box, free, correct by construction. Conservative: cross-dimension or
  unknown-unit requests decline rather than guess.
- **VERIFY** — for any query with a number, the local model answers and plugs its own
  numbers back into the problem's relationships (`verify.py`); we re-derive each one exactly.
  A false check is a hard escalate — the answer is provably inconsistent with the problem,
  not merely out-voted.
- **Category / open-ended rules** and **self-consistency** handle the rest, escalating to any
  OpenAI-compatible frontier endpoint only when a query genuinely earns it.

### Measured (qwen2.5:7b, 20-query labeled set, `bench_router.py`)

- 75% of queries answered on-box, with no frontier call.
- 13/15 on-box answers correct — the only two wrong are documented setup traps.
- 3/3 confident-wrong multiplications caught and escalated.

### Tooling

- `solver.py` / `verify.py` with full test suites (`test_solver.py` 50/50,
  `test_verify.py` 28/28) — no model or network required.
- `bench_offline.py` (what the solver buys) and `bench_router.py` (full-router benchmark).
- `server.py` — OpenAI-compatible endpoint exposing the router as model `hybrid`.

### How it got here

hybrid went through three design generations before this first release, each kept honest
about its own limit:

1. **Category rules** — escalate the domains a small model is known to fail (code, proofs,
   puzzles); answer nothing risky locally.
2. **+ self-consistency** — sample the local model a few times and escalate when it disagrees
   with itself. Catches genuine *uncertainty* — but cannot catch *confident wrongness* (a small
   model states `17⁴ = 6859` unanimously; it's 83,521).
3. **+ a verifier stronger than the model** — the deterministic solver and the plug-back check.
   This is what catches confident-wrong arithmetic that self-consistency waves through, and it
   is the heart of v1.0.0.

The limit that remains, kept visible in `--demo` and the benchmark: the oracle checks the
answer against the relationships the model *transcribes*, so a self-consistently-wrong *setup*
still needs the frontier. A second cheap model as an independent vote was tested and rejected —
it shares the classic blind spots.
