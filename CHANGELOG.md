# Changelog

All notable changes to hybrid are documented here. This project adheres to
[Semantic Versioning](https://semver.org).

## v1.12.0 — 2026-07-11

**Logit-read classification — read the posterior, don't sample it.** The labelled
vote asked the model the same question k times at temperature 0.6 and checked
agreement: a noisy, 3-forward-pass approximation of a number the model computes in
its FIRST forward pass — the probability distribution over the label set. When the
output space is enumerable, confidence is directly readable.

- On the llamacpp transport, labelled classification now makes ONE call
  (`n_predict: 1`, `n_probs`) and reads the first-token distribution: token mass is
  mapped onto labels (BPE splits like `autom`→automate and leading-space variants
  are summed; a prefix matching several labels is skipped, not guessed), and the
  argmax is served behind a probability-and-margin gate (`HYBRID_LABEL_MIN_P`,
  default 0.4; `HYBRID_LABEL_MARGIN`, default 2.0). A soft posterior escalates —
  after the model slot is released, so a frontier round-trip never holds local
  capacity.
- **Measured on a real 0.5B** (same six forge-shaped requests, warm): p50 505 →
  374 ms, on-box 5/6 → 6/6, and two consecutive logit passes were IDENTICAL —
  the vote's pass-to-pass swing (temperature sampling) is gone entirely.
- **The honest finding the read exposes:** the small model's known bias class
  (two items mislabelled `analyze`) serves at posteriors 0.66/0.69 — inside the
  range of its CORRECT answers (0.56–0.87). Confidently-wrong is calibrated-
  looking; no consensus vote and no fixed threshold separates it. What the logit
  read adds is the NUMBER: every decision now logs `label posterior p1 vs p2`,
  so escalation thresholds can be tuned per family from real traffic — and the
  logged (margin, frontier-verdict) pairs are exactly the training signal for a
  distilled student that kills the bias with data.
- The sampling vote remains as the automatic fallback: ollama transport, servers
  without `top_logprobs`, unreadable responses, or `HYBRID_LABEL_LOGITS=0` — any
  of these degrade to exactly the k-sample behavior.
- `test_backend.py` fake server gains `top_logprobs`; new pins: posterior mapping
  (BPE/space variants, ambiguity skip), one-forward-pass serve, soft-margin
  escalate with probabilities in the why-string, missing-probs and env-off
  fallbacks. Suite total 342.

## v1.11.0 — 2026-07-11

**Slot pinning — prompt families keep their prefill.** `cache_prompt` only reuses a
prefill when the matching KV sits in the slot a request happens to land on. llama-server
spreads unpinned requests across slots, so a family of requests sharing a long fixed
prefix (a classifier's system prompt) keeps re-prefilling that prefix — worst in the
k-sample vote, where k IDENTICAL prompts land on k slots and prefill k times
simultaneously.

- The llamacpp transport now pins a request's **prompt family** to one server slot:
  `ollama(..., family=...)` sets `id_slot` to a stable hash of the family name modulo
  the server's slot count (discovered once via `GET /slots`). Same family → same slot,
  every process, every restart. The labelled-classification vote passes its family
  (system prompt + label set), so sample 1 prefills and samples 2..k reuse the whole
  prompt; the prefix then stays hot for the classifier's next request.
- **Measured** on a real llama-server (3B, `--parallel 3`, fresh server per mode, same
  six forge-shaped requests): cold classify 9434ms → 3571ms (**2.6×**), warm p50
  3175ms → 1623ms (**2.0×**). Labels chosen were identical in both modes — pinning
  changes where work lands, never what is answered.
- Pinned only where decode is tiny and the prefix dominates (grammar-locked labels:
  1–4 tokens). Long-decode votes stay unpinned on purpose — they gain more from
  batching across slots than they lose re-prefilling.
- Degrades to exactly the old behavior when the server does not expose `GET /slots`
  (`--no-slots`), and `HYBRID_SLOT_PIN=0` is the escape hatch. No new required config.
- `test_backend.py` grows the fake server a `/slots` endpoint and pins the behavior:
  stable slot per family, one probe per server, escape hatch, no-endpoint degradation,
  and the labelled vote riding one slot. Suite total 334.

## v1.10.0 — 2026-07-10

**Labelled classification on the Anthropic endpoint — constrain and verify, not vote
and hope.** Most cheap Anthropic calls are classification (pick one of these labels).
The plain instruction vote handles it, but votes on raw text — a rambly local answer
breaks unanimity and nothing stops the model inventing an out-of-set label.

- A request declares its label set via `metadata.hybrid_labels: ["build", "research",
  ...]` (a custom key real Anthropic ignores). hybrid then GRAMMAR-LOCKS the local model
  to emit exactly one of those labels (GBNF, on the llama.cpp transport), samples it k
  times, and normalizes each sample to the label it contains before voting. A served
  answer is both self-consistent AND provably in-set; disagreement, or a sample with no
  in-set label, escalates. The verifier discipline — constrain the output, verify it is
  valid — applied to labels instead of arithmetic.
- **Measured live**, grammar-locked, on a real 3B over `build`/`research`/`monitor`/
  `security`: four dispatcher-style requests each classified on-box, unanimous, and
  guaranteed in-set — the model cannot return a category you did not ask for.
- New helpers: `_clean_labels`, `_labels_grammar` (returns None on an unsafe label →
  degrade to unconstrained-sample-then-extract, never a broken grammar), `_extract_label`
  (whole-word, tolerates preamble). A malformed `hybrid_labels` payload falls back to the
  plain instruction-following path.
- `test_messages.py` grows to 32 with the label unit tests, the routing table
  (unanimous / split / no-label / malformed), and the end-to-end `metadata` path. Suite
  total 325.

## v1.9.0 — 2026-07-10

**An Anthropic `/v1/messages` front door** — so the Anthropic-shaped callers a fleet
is full of (inline `@anthropic-ai/sdk` `messages.create`, a base-URL-configurable
client) can point at hybrid and get local-first routing with frontier escalation. The
OpenAI `/v1/chat/completions` surface is unchanged.

- **Instruction-following aware routing** (`route_messages`): Anthropic cheap calls put
  the task in the `system` prompt (classify/extract/judge), not the user turn — so the
  arithmetic tiers (which impose their own prompt) do NOT apply. When a `system`
  instruction is present, hybrid votes with the local model on the CALLER'S OWN prompt;
  unanimous serves on-box, otherwise the whole conversation escalates to the frontier.
  No system prompt -> the user turn is a self-contained question and gets the full
  router (verifier and all). A low-confidence local answer is never served.
- **Server**: `POST /v1/messages` (non-streaming, text-only) + `/v1/messages/count_tokens`;
  `x-api-key` auth accepted alongside `Authorization: Bearer`; Anthropic message/error
  response shape; the decision log tags `api:"anthropic"`. Streaming and the tool-using
  agent path are out of scope (they need the real model).
- **Measured live** against a real 3B: dispatcher-style classification (build / research /
  monitor) came back on-box, unanimous, and correct — an uncertain one would escalate.
- Refactor: the failure policy (`_apply_failure_policy`) and the vote engine (`_vote`)
  are factored out and shared by both surfaces — the OpenAI path is byte-identical
  (full suite green before the new tests). New `test_messages.py` (19). Suite total 312.

## v1.8.0 — 2026-07-10

**Load shedding — the tier that makes "production" honest.** On a CPU box the model
tiers cost seconds and decode is memory-bandwidth-bound, so concurrent model requests
queue on the same bus rather than parallelizing. Under load, hybrid now escalates the
expensive local work to the frontier instead of making a caller wait. Both signals are
**off by default** (behavior unchanged); the deterministic tiers never shed.

- **`HYBRID_MODEL_MAX_INFLIGHT=N`** — run at most N model-tier requests at once; beyond
  the cap, shed to the frontier. Backed by a shared, thread-safe in-flight gauge, taken
  atomically at a gate placed after the free tiers and before any model call, and held
  for the whole model portion via `try/finally`. `N=1` is the honest one-box CPU
  setting. Exposed as `model_inflight` on `/health`.
- **`HYBRID_LATENCY_BUDGET_MS=ms`** (+ `HYBRID_MODEL_TIER_MS`, default 8000) — a
  per-request wall-clock budget; if elapsed + projected model cost (scaled by queue
  depth ahead) would exceed it, shed. "Answer what fits your SLA, escalate the rest."
- A shed is an ordinary escalation: carries the conversation, obeys the frontier failure
  policy, logs `route: ESCALATE, why: "load shed: …"`.
- **Measured live** on the 2013 box with `MAX_INFLIGHT=1`: two concurrent model-path
  queries — the first re-derived locally on the 7B, the second (arriving while that slot
  was held) shed straight to the frontier instead of queueing behind it.
- Tests: `test_route.py` grows to 39 with the gate unit tests, a cap/deterministic-tier
  matrix, and a **threaded** integration test that genuinely holds a slot open while a
  concurrent request sheds. `/health` gauge pinned in `test_server.py`. Suite total 293.
- Refactor: the model-tier portion of `_route` is extracted to `_route_model` so the
  slot brackets it cleanly — behavior byte-identical with shedding off (full suite green
  before any new test was added).

## v1.7.0 — 2026-07-10

The GPU-less fast path: a native **llama.cpp transport** for the local tier
(`HYBRID_LOCAL_BACKEND=llamacpp`), built from measured CPU physics — prefill is
compute-bound and decode is bandwidth-bound, so the wins come from not re-prefilling
and not over-generating, not from a faster kernel.

- **Prefix caching**: tier instructions ride in the template's system slot and
  `cache_prompt: true` re-uses their prefill across calls — a transcription call's
  prefill drops from ~128 tokens to ~24 after the first (measured ~5 s/call back on a
  2013 Haswell; the preamble is the majority of most tier prompts).
- **GBNF grammars** lock the transcription tiers to exactly the `EQN:`/`ANSWER:`/
  `CHECK:` shapes the oracles parse. Kills the ramble class (measured worst live case:
  210 tokens of unparseable LaTeX in 54 s → 23 clean tokens in 6.5 s) and the
  units-inside-CHECK fall-through class. `HYBRID_GRAMMAR=0` opts out. llama-server
  silently ignores malformed grammars, so the grammar strings are pinned by tests.
- **Fused transcription tier — EXPERIMENTAL, off by default** (`HYBRID_FUSE=1` opts
  in): ONE model call for equations + answer + substitution checks, read
  strongest-signal-first with two-call precedence (derive out-ranks a sloppy false
  CHECK — the self-referential brick case, pinned by a test). Measured ~2.2× end to
  end, and then measured why it stays off: transcribe-AND-self-check in one call
  degrades the transcription itself (a mixed-unit conversion the setup tier gets
  exactly right came back mangled, a percent answer lost its unit), and plug-back then
  grades the mangled answer's true-but-disconnected arithmetic as "checked".
- The grammars carry a **bounded THINK block** (up to 6 × 220-char lines, invisible to
  the parsers): the first cut had no think room and transcription quality collapsed on
  exactly the trap classes the derive tier exists for — the prose the prompts used to
  permit WAS the model's chain of thought.
- **Honest finding, promoted to the README:** the classic setup traps are
  runtime-FRAGILE at temperature 0 — the same model + prompts flip between cracked /
  caught / wrong-served across llama.cpp builds and transports, independent of
  grammar, temperature, and prompt wrap. "Zero wrong served" on that class is a fact
  about a runtime build; the runtime-stable safety is the deterministic tiers. Bench
  tables now state the runtime.
- New `test_backend.py` (44 checks): the transport against a real loopback fake —
  ChatML wrap, system-slot instructions, cache/stop/n_predict fields, grammar
  attach/opt-out, retry-once + BackendError, fusion-off default — plus grammar sanity
  pins (GBNF has no `\-` class escape; a literal dash sits last; llama-server silently
  ignores a malformed grammar, so a typo would quietly disarm it). Route suite grows
  to 30 with the fused decision table.
- Ollama transport behavior is byte-identical to v1.6.1 by default (the `grammar`
  argument is ignored by the generate API; fusion is opt-in everywhere).
- **Split-server fast tier** (`LLAMACPP_URL_FAST`): llama-server loads one model, so
  the split-model policy gets a second server — vote/creative calls (they pass
  `model=LOCAL_MODEL_FAST`) route to it, transcription never does. Same pinned
  safety rule as the Ollama transport, enforced by URL. Measured, and OPT-IN for a reason: on llama.cpp b9660 the 3B voted 'Brazil'
  for the 2014 World Cup — 3/3 self-consistent, served in 1.5 s. A factual
  wrong-served the 7B does not make on this runtime, and a direct demonstration
  that the '3B is safe on votes' result was itself runtime-specific. The
  mechanism ships; the trade is yours, with the receipt attached.
- **Vote budget trimmed** to 56 tokens (CONCISE asks for a number, a word, or a
  sentence or two; past that is decode spent on a ramble no key extraction reads),
  and `HYBRID_VOTE_FAST=1` (opt-in) votes 2-of-2 instead of 3-of-3 — one fewer
  decode, strictly more escalation-prone on disagreement. Measured: bench wall 2 m 03 s -> 1 m 28 s (-28%), but a 2/2 vote agreed-wrong on a
  trap the 3/3 vote escalated. Off by default.

## v1.6.1 — 2026-07-03

- **The self-consistency vote fires its k samples concurrently** (stdlib
  `ThreadPoolExecutor`). CPU decode is memory-bandwidth-bound, so a server that
  batches concurrent requests — Ollama with `OLLAMA_NUM_PARALLEL >= 3` — streams the
  weights once per token-step for all samples: the vote costs roughly ONE sample's
  wall time instead of three. Against a serial server the requests simply queue (same
  behavior and total time as the old loop), so there is no configuration to get wrong.
  Verified live on the box after raising `OLLAMA_NUM_PARALLEL` to 4: three concurrent
  decodes overlap instead of serializing. Errors re-raise exactly as before — the
  failure-policy surface is unchanged, and the whole 227-test suite passes untouched.

## v1.6.0 — 2026-07-02

Split local models + an answer cache — the other two latency levers that need no GPU.

### Router

- **`LOCAL_MODEL_FAST`** (default: `LOCAL_MODEL`, so nothing changes until you set it) —
  a smaller model for the **vote and creative tiers only**. The live 3B experiment cut
  both ways: llama3.2:3b was flawless on factual votes and rewrites at ~2x the speed
  (measured 17.0 vs 8.3 tok/s — CPU decode scales with model bytes), but **tripled
  wrong-served answers when allowed to transcribe** — it writes wrong-but-internally-
  consistent equation systems the exact oracle then faithfully re-derives. So the split
  is enforced in the router, not left to configuration discipline: the derive/verify
  transcription tiers always use `LOCAL_MODEL`, and results report which model answered.

### Server

- **Answer cache** (`HYBRID_CACHE_TTL` seconds, default 0 = off; `HYBRID_CACHE_MAX`
  entries, default 512, LRU) — a repeated single-turn query is served from memory in
  ~0 ms with `x_hybrid.cached: true` and a `cached` mark in the decision log. Real
  traffic repeats; a hit costs neither tokens nor memory bandwidth. Deliberately narrow:
  multi-turn requests always re-route (the same last message can mean something else
  mid-conversation), and `ERROR` / `DEGRADED` results are never cached.

### Tests

- `test_route.py` 19 → 21 (fast model on the vote tier; transcription pinned to
  `LOCAL_MODEL`); `test_server.py` 18 → 22 (cache hit skips routing, multi-turn
  bypass, ERROR never cached, off by default). **Suite: 221 → 227, all offline.**

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
