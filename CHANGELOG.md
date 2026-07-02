# Changelog

All notable changes to hybrid are documented here. This project adheres to
[Semantic Versioning](https://semver.org).

## v1.5.0 — 2026-07-02

The template transcriber — the fastest token is the one you never generate.

### Why

Two live findings drove this tier. First, latency: CPU decode is memory-bandwidth-bound
(measured ~8 tok/s for a 7B on an 8-core box), so a routed word problem costs 5–40 s in
generated tokens no CPU trick can speed up. Second, safety: an experiment swapping the
local model for a smaller one (llama3.2:3b) tripled wrong-served answers — the weaker
model writes *wrong but internally consistent* equation systems, and the oracle faithfully
re-derives garbage. **Transcription is the one surface the exact oracle cannot check.**
Both problems have the same fix: for the shapes that dominate everyday quantitative
queries, take the model out of transcription entirely.

### Router

- **TEMPLATE** (`templates.py`) — five rigid word-problem shapes parsed deterministically
  and solved in closed form over `Fraction`s: rate × quantity/time ("2,417 pages per hour
  → 94 hours"), total + gap pairs (bat-and-ball), reverse-percentage ("costs $68 after a
  15% discount"), plain shifts ("a number decreased by 12 is 39"), and two-price mixes
  ("$9 kids / $14 adults → 3 kids + 2 adults"). Zero model calls, zero tokens, zero
  latency; the answer is exact by construction. The confident-wrong-product class the
  verifier used to have to *catch* (v1.0.0's headline) is now simply answered — a
  recognized rate shape cannot be multiplied wrong.
- Runs after `solver.py`, **before the hard-category rule**: a clean exact parse
  out-ranks a stray rule keyword ("...13.9 liters..." no longer escalates a unit-rate
  query the transcriber answers exactly).
- **Ruthlessly conservative, by contract**: every number in the query must be consumed
  by the shape's slots (the v1.1.1 mixed-unit lesson, promoted to a rule); number-words
  ("half", "twice") outside a slot decline; declaration and question nouns must agree
  (stemmed); money markers must be consistent; negatives and >4-number queries decline.
  Set-logic riddles (Emma/Sally/Tom), work-rate traps, and exponential growth never
  match — they fall through to the model tiers exactly as before.

### Tests + bench

- `test_templates.py` (new, 58 tests) — every shape against the live bench/holdout/stress
  queries it retires, plus 30 must-decline traps and near-misses. Offline, like every
  oracle suite. `test_route.py` 16 → 19 (template short-circuit, template-beats-hard-rule,
  template-declines-fall-through). **Suite: 160 → 221, all offline.**
- `bench_router.py` — the three v1.1 "catch" products are recategorized `template` (they
  are now answered exactly on-box instead of escalated); two new catch cases carry an
  extra quantity so the transcriber declines and the verify tier stays honestly measured;
  new `TEMPLATE` summary metric; dynamic denominators.

### Measured (qwen2.5:7b, 8-core CPU box, frontier stubbed)

- Labeled bench (now 22 cases): **17/22 on-box (77%), 17/17 correct — zero wrong
  served**, 7/7 template shapes exact with zero model calls, 2/2 off-template
  confident-wrong products still caught by the verifier, both setup traps still handled
  (chicken solved + verified, Sally caught).
- Fresh 24-query holdout (never seen by any tier): **22/24 on-box (92%)**, **11/24
  answered in 0 ms** (solver + templates), wall time **228 s → 106 s** on the same box
  and mix. The one wrong-served answer remains the documented transcription-leak trap
  (a sisters-riddle variant), which the templates correctly decline — the model-side
  limit is unchanged, just reached less often.
- The movie-tickets case shows the ordering paying off twice: v1.4.0 caught the local
  model's wrong total (60) by derive-mismatch and escalated — correct, but slow and a
  frontier call. v1.5.0 answers 55 exactly, for free.

## v1.4.0 — 2026-07-02

Production hardening, part 3: installable, deployable, publishable.

### Packaging

- **`pip install hybrid-router`** (or `pipx install hybrid-router`) — console commands
  **`hybrid`** and **`hybrid-server`**. Zero runtime dependencies: the wheel is the five
  flat modules, installed exactly as they read in the repo. Version single-sourced from
  `hybrid.__version__`; `hybrid --version` reports it.
- **PyPI Trusted Publishing** (`publish.yml`) — every GitHub release builds and publishes
  via OIDC; no API tokens anywhere. A sanity step refuses to publish when the tag doesn't
  match `hybrid.__version__`. First publish uses PyPI's *pending publisher* flow (a
  one-time web-UI registration; verified against the current PyPI docs).

### Deploy

- **`Dockerfile`** — python:3.12-slim + the five modules (~50 MB), `/health` healthcheck,
  stdout = the JSONL decision log.
- **`deploy/docker-compose.yml`** — the whole local tier in two containers (ollama +
  hybrid), port published to loopback only; escalation works the moment
  `FRONTIER_API_KEY` lands in the environment.
- **`deploy/hybrid.service`** — hardened systemd unit (`DynamicUser`,
  `ProtectSystem=strict`, `NoNewPrivileges`); `journalctl -u hybrid` is the decision log.

### CI

- New **package job**: build sdist + wheel, install the wheel, then smoke the installed
  console scripts *away from the checkout* — `hybrid --version`, a SOLVED query with no
  model, and a `hybrid-server` boot + `/health` probe.

## v1.3.0 — 2026-07-02

Production hardening, part 2: the server grows the surface a real OpenAI client
expects, and every request leaves an observable trail.

### Server

- **`stream: true` works (SSE).** OpenAI SDKs, Cursor, and Cline default to streaming;
  the flag was previously ignored and clients got a body they weren't parsing. Now:
  role delta → one content delta (the answer arrives whole — routing has to finish
  before an answer exists; the verify tiers see it complete) → a stop chunk carrying
  `x_hybrid` + `usage` → `data: [DONE]`.
- **JSONL decision log.** One line per request — ts, route, why, backend, latency,
  wall time, status, stream flag, sha256 prefix + length of the query — to stdout
  (banner moved to stderr so stdout is pure JSONL) or `HYBRID_LOG` file. Query text
  only with `HYBRID_LOG_QUERIES=1`: observability without logging user content by
  default.
- **Limits + honest errors.** Request-body cap (`HYBRID_MAX_BODY`, default 1 MiB →
  413), content-length required (411), and a route `ERROR` (v1.2.0 failure policy)
  maps to **502 with an OpenAI-shaped error object** + `x_hybrid` — an outage is never
  a 200 with error-shaped content.
- **Optional bearer auth.** `HYBRID_API_KEY` gates everything except `/health`
  (constant-time compare); `HYBRID_HOST` binds beyond loopback deliberately.
- **Protocol polish.** `usage` chars/4 estimates flagged `usage_estimated`, `/health`
  with version, handler timeout so a stalled client releases its thread, version
  headers scrubbed, multi-turn conversations accepted.
- **Fix:** `SOLVED` answers were labelled `model: hybrid:frontier` in responses —
  now `hybrid:local` (the solver is the most on-box tier there is).

### Router

- `route(query, messages=None)` — multi-turn passthrough: routing and the local tiers
  always work on the last user message; an **escalated call carries the whole
  conversation** to the frontier. The 1-arg `escalate()` stub contract (bench,
  measure, tests) is preserved via an internal dispatch helper.
- `__version__` — reported by `/health`, the server banner, and (soon) packaging.

### Tests

- `test_server.py` (new, 18 tests) — a real `ThreadingHTTPServer` on an ephemeral
  loopback port against a faked `route()`: protocol round-trip, SSE shape (delta
  chunks, `[DONE]`, stop reason, `x_hybrid` on the final chunk), multi-turn
  passthrough, body cap, auth matrix, 502 mapping, and decision-log opt-in.
  **Suite: 142 → 160, all offline.**

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
