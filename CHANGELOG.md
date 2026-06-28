# Changelog

All notable changes to hybrid are documented here. This project adheres to
[Semantic Versioning](https://semver.org).

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
