# hybrid — local-first LLM routing with frontier escalation

[![tests](https://github.com/askalf/hybrid/actions/workflows/test.yml/badge.svg)](https://github.com/askalf/hybrid/actions/workflows/test.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/askalf/hybrid/badge)](https://scorecard.dev/viewer/?uri=github.com/askalf/hybrid)

<p align="center">
  <img src="https://raw.githubusercontent.com/askalf/hybrid/main/og.png" alt="hybrid: answer the easy majority locally, escalate the few that earn it" width="840">
</p>

> _hybrid — **own your inference**. Part of **[Own Your Stack](https://github.com/askalf)** — own your AI infrastructure instead of renting it by the token._

Answer the easy majority of your LLM queries on your **own machine** — free, private,
fast — and escalate only the genuinely hard ones to a **frontier model**. Most of what
you ask an LLM is easy (facts, rewrites, simple Q&A, arithmetic); the rare hard query (a
proof, real code, multi-step reasoning) goes to the frontier. Frontier quality where it
matters; nothing paid or sent off your machine for the rest.

Dependency-free Python (stdlib only). Built and measured on a **GPU-less 2013 desktop** —
the writeup, with all the numbers, is here:
[**Your CPU isn't bad at LLMs — it's bandwidth-starved**](https://sprayberrylabs.com/blog/own-your-inference).

The hard part isn't routing the easy queries home — it's knowing when the cheap model is
**confidently wrong**. A router built on the cheap model's own signals (classification,
self-consistency) inherits its blind spots: it can't tell confident-and-right from
confident-and-wrong. hybrid's answer is a **free verifier that is stronger than the
model** — Python's exact arithmetic — applied at every depth it can reach: solve the
closed forms outright, transcribe the *shaped* word problems deterministically, and
re-derive the model's own working on everything else.

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
1. **Template transcriber** (`templates.py`, new in v1.5.0) — the derive tier's lesson,
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
4. **Setup re-derivation** (`equations.py`, new in v1.1.0) — for any *quantitative* query (a
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

## Measured (`bench_router.py`, 22-query labeled set, qwen2.5:7b)

The real router over a labeled mix — closed-form, conversions, shaped word problems,
factual, off-template confident-wrong arithmetic, hard, and setup traps. Frontier
escalation is stubbed, so the benchmark is free (measured live on an 8-core CPU box):

```
ON-BOX:        17/22 (77%) answered without a frontier call
ON-BOX SAFETY: 17/17 on-box answers correct        (ZERO wrong answers served)
TEMPLATE:      7/7 shaped word problems answered exact, zero model calls
CATCHES:       2/2 confident-wrong arithmetic intercepted -> escalated
ESCALATED:     5/22 routed to the frontier
HONEST LIMIT:  0 setup traps slipped through local + wrong
```

The line has moved three times now. v1.0.0: 15/20 on-box but **13/15 correct** — both
documented setup traps served locally and *wrong*. v1.1.0: zero wrong served — the setup
re-derivation tier caught Sally's-sisters and solved chicken-and-a-half, trading on-box
points for safety. v1.5.0 moves it again in the other direction: the shaped word problems
— *including the confident-wrong products the verifier used to have to intercept* — are
now answered exactly with **zero model calls and zero latency**, so the on-box rate goes
back UP without giving back any safety. A few of the rows, verbatim:

```text
SOLVED     How many feet in 3 miles?                  -> 15840    (exact, free)
SOLVED     1,847 widgets/day for 263 days             -> 485761   (template: rate — v1.1 had to CATCH the model flubbing this; now it is answered, exactly, in 0 ms)
SOLVED     bat and ball, bat $1 more                  -> 0.05     (template: sum-diff — the $0.10 trap answer is unproducible)
LOCAL      chicken-and-a-half                         -> 0.67     (setup re-derived)
ESCALATE   56 crates arrive, 3 damaged — units left?  -> caught: the extra quantity makes the template decline; the verifier catches the model's flubbed product
ESCALATE   Sally's-sisters                            -> caught: its own equations contradict its answer
```

On a fresh **24-query holdout** (never seen by any tier — new numbers, new phrasings, new
traps), the same build measured **22/24 on-box (92%)** with **11/24 answered in 0 ms**
(solver + templates) and total wall time halved on the same box and mix (228 s → 106 s).
The one wrong-served answer is the documented transcription-leak trap, which the
templates correctly decline — the model-side limit is unchanged, just reached less often.
**Where a small model stays confidently wrong even when it reasons well is exactly where
a free exact oracle wins — and the strongest form of winning is never asking it.**

## Measured economics (`measure_routing.py`, v1.0.0 routing mix)

On-box *query share* and *dollar share* are not the same number. The queries that escalate
are the token-heavy ones — a proof, a code-gen — so routing saves less than the 75% on-box
rate suggests. `measure_routing.py` prices every query's frontier cost: escalations at what
they really cost, on-box answers at the counterfactual cost they avoided.

```
ON-BOX:               15/20 (75%) answered without a frontier call
$ SAVED:              ~52% of frontier spend avoided  (not 75%)
PER 1000 (this mix):  ~$2.86 all-frontier  ->  ~$1.37 hybrid
```

Three-quarters of *queries* stay home, but only about *half the dollars*: escalation
correctly sends the few token-expensive hard problems to the frontier — which is the whole
point, and the reason query-share overstates the savings. Measured against claude-sonnet-4-6
list pricing ($3 / $15 per 1M); set `PRICE_IN_PER_M` / `PRICE_OUT_PER_M` for your own frontier.

## Install

```bash
pipx install hybrid-router       # console commands: hybrid, hybrid-server
# or: pip install hybrid-router
# or straight from the repo: pipx install git+https://github.com/askalf/hybrid
```

Zero runtime dependencies — the wheel is the six modules you can read above, installed
exactly as they read. Published to PyPI from CI on every release via **Trusted
Publishing** (OIDC — no tokens anywhere). `hybrid --version` tells you what you got.

## Run

```bash
# local tier — Ollama with a small model
ollama pull qwen2.5:7b           # the measured default — the TRANSCRIPTION model
ollama pull llama3.2:3b          # optional: LOCAL_MODEL_FAST=llama3.2:3b makes the
                                 # vote/creative tiers ~2x faster. Measured live: a 3B is
                                 # safe there — but NEVER as LOCAL_MODEL; allowed to
                                 # transcribe, it tripled wrong-served answers

# frontier tier — any OpenAI-compatible endpoint
export FRONTIER_API_KEY=sk-...                                    # OpenAI, or your own proxy
export FRONTIER_URL=https://api.openai.com/v1/chat/completions    # default; point anywhere OpenAI-compatible
export FRONTIER_MODEL=gpt-4o                                      # default

python solver.py "how many feet in 3 miles"   # the deterministic tier alone -> 15840
python templates.py "A printer prints 2,417 pages per hour. How many pages in 94 hours?"
                                               # the template transcriber alone -> ('227198', 'rate')
python test_solver.py                          # solver tests (53/53, no model needed)
python test_templates.py                       # template transcriber tests (58/58, no model needed)
python test_verify.py                          # verifier tests (28/28, no model needed)
python test_equations.py                       # setup re-derivation tests (45/45, no model needed)
python test_route.py                           # router plumbing + failure policy (21/21, no model needed)
python test_server.py                          # server surface: SSE, auth, limits, cache (22/22, no model needed)
python bench_router.py                         # full-router benchmark: on-box %, safety, catches
python measure_routing.py                      # router economics: $ saved vs all-frontier (needs FRONTIER_API_KEY)
python hybrid.py "your question"               # route one query
python hybrid.py --demo                        # mixed test set + summary
python server.py                               # OpenAI-compatible server on :8080 (model "hybrid", stream ok)
```

The oracle tiers and both harnesses (router + server tests) need **nothing** — no model,
no network — so all 227 tests run anywhere, including CI.

### The server, as a service

`server.py` speaks enough OpenAI protocol for real clients: **`stream: true` works**
(SSE — role delta, one content delta, a stop chunk, `[DONE]`; the content arrives whole
because routing has to finish before an answer exists), multi-turn conversations route
on the last user message while an **escalated call carries the whole conversation**, and
every response has `x_hybrid` (route / why / backend / latency) plus a chars/4 `usage`
estimate (flagged `usage_estimated` — the local tier isn't token-metered).

Every request writes one **JSONL decision line** — route, why, backend, latency, status,
a sha256 prefix of the query — to stdout (the banner goes to stderr) or to `HYBRID_LOG`.
Query text stays out of the log unless `HYBRID_LOG_QUERIES=1`. A backend failure is a
**502 with an OpenAI-shaped error object** (see failure policy), never error text
disguised as an answer. `/health` reports liveness + version without auth; set
`HYBRID_API_KEY` to require a bearer token on everything else, and `HYBRID_HOST` if you
deliberately bind beyond loopback.

**Repeats are free.** Set `HYBRID_CACHE_TTL=300` and a repeated single-turn query is
served from memory in ~0 ms with `x_hybrid.cached: true` — real traffic repeats, and a
cache hit costs neither tokens nor bandwidth. Multi-turn requests, `ERROR` results, and
`DEGRADED` answers are never cached; `HYBRID_CACHE_MAX` (default 512) caps entries, LRU.

Capacity honesty: on a CPU box the *model* tiers run **seconds-to-a-minute per query**
and effectively serially — that's memory bandwidth, not a bug. The solver and template
tiers answer in ~0 ms regardless, and `LOCAL_MODEL_FAST` roughly halves the vote/creative
tiers. Size expectations (and any reverse proxy timeouts) for the residual model-path
queries accordingly.

Deploying it: the **`Dockerfile`** is python-slim plus the five modules (with a
`/health` healthcheck); **`deploy/docker-compose.yml`** runs the whole local tier —
ollama + hybrid — with the port published to loopback only; **`deploy/hybrid.service`**
is a hardened systemd unit (`DynamicUser`, `ProtectSystem=strict`) where
`journalctl -u hybrid` *is* the decision log.

### Config (env)

| var | default | |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | local Ollama endpoint |
| `LOCAL_MODEL` | `qwen2.5:7b` | the **transcription** model (derive/verify tiers) |
| `LOCAL_MODEL_FAST` | = `LOCAL_MODEL` | smaller model for the **vote/creative** tiers only — safe there, measured; never for transcription |
| `FRONTIER_URL` | `https://api.openai.com/v1/chat/completions` | any OpenAI-compatible endpoint |
| `FRONTIER_API_KEY` | — | required for escalation |
| `FRONTIER_MODEL` | `gpt-4o` | frontier model id |
| `PORT` | `8080` | server.py listen port |
| `HYBRID_ON_LOCAL_FAIL` | `escalate` | local backend down → `escalate` to the frontier, or `error` |
| `HYBRID_ON_FRONTIER_FAIL` | `error` | frontier down → honest `error`, or `local` (degraded, unverified) |
| `HYBRID_HOST` | `127.0.0.1` | server bind address — set with intent, pair with auth |
| `HYBRID_API_KEY` | — | if set, server requires `Authorization: Bearer <key>` (except `/health`) |
| `HYBRID_MAX_BODY` | `1048576` | server request-body cap, bytes |
| `HYBRID_LOG` | stdout | decision-log JSONL file (append) |
| `HYBRID_LOG_QUERIES` | off | `1` = include query text in the decision log |
| `HYBRID_CACHE_TTL` | `0` (off) | seconds to serve repeated single-turn queries from memory (~0 ms hits) |
| `HYBRID_CACHE_MAX` | `512` | answer-cache entry cap, LRU-evicted |

`FRONTIER_URL` is just an OpenAI-compatible chat endpoint — OpenAI, a local proxy, or your
own gateway. The key only ever leaves your machine on an *escalated* query.

### Failure policy

A dead backend degrades predictably. If the **local model** is unreachable, queries
escalate to the frontier (set `HYBRID_ON_LOCAL_FAIL=error` to fail them instead). If the
**frontier** is unreachable, a query that earned it returns an explicit error — never a
silently-substituted local answer. `HYBRID_ON_FRONTIER_FAIL=local` opts into
availability-over-correctness: a plain local answer labelled `DEGRADED`, *including* for
queries whose local answer the verifier just refuted — opt in knowingly. Either way a
failure is a structured result (`route: ERROR`), never an answer-shaped string.

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

## Files

- `hybrid.py` — router + dispatch + `--demo`
- `solver.py` — deterministic arithmetic + exact unit/percentage/multiple conversion (the SOLVED tier)
- `templates.py` — deterministic word-problem transcriber: five rigid shapes parsed and
  solved in closed form over `Fraction`s, no model; ruthlessly conservative
- `equations.py` — setup re-derivation: solve the model's transcribed equation system exactly
  (linear systems, Gaussian elimination over `Fraction`s); conservative
- `verify.py` — verify-the-local-answer: re-derive the model's plugged-in checks exactly
- `test_solver.py` / `test_templates.py` / `test_verify.py` / `test_equations.py` /
  `test_route.py` / `test_server.py` — 227 tests (oracles + transcriber + router
  plumbing + failure policy + server surface + cache); all offline, no model needed
- `bench_offline.py` — what the solver buys versus a no-solver router (no model needed)
- `bench_router.py` — full-router benchmark: on-box rate, on-box safety, catches (frontier stubbed)
- `measure_routing.py` — router economics: prices every query's frontier cost to show real $ saved
- `server.py` — OpenAI-compatible front end: SSE streaming, JSONL decision log, body
  caps, optional bearer auth
- `pyproject.toml` / `Dockerfile` / `deploy/` — pip/pipx packaging (console commands
  `hybrid` + `hybrid-server`), container image, compose + systemd examples

## License

MIT

## Own Your Stack

Part of **[Own Your Stack](https://github.com/askalf)** — open tools for owning your AI infrastructure instead of renting it by the token. One subscription. Your box. Your terms.

- **[dario](https://github.com/askalf/dario)** — own your routing
- **[hybrid](https://github.com/askalf/hybrid)** — own your inference _(you are here)_
- **[deepdive](https://github.com/askalf/deepdive)** — own your research
- **[hands](https://github.com/askalf/hands)** — own your computer-use
- **[browser-bridge](https://github.com/askalf/browser-bridge)** — own your browser
- **[warden](https://github.com/askalf/warden)** — own your agent security
- **[canon](https://github.com/askalf/canon)** — own your agent skills
- **[keeper](https://github.com/askalf/keeper)** — own your agent secrets
- **[cordon](https://github.com/askalf/cordon)** — own your prompts
- **[picket](https://github.com/askalf/picket)** — own your agent browser
- **[amnesia](https://github.com/askalf/amnesia)** — own your search
- **[askalf](https://askalf.org)** — own your operation: the AI operation that runs Sprayberry Labs

---
Part of **[Own Your Stack](https://github.com/askalf)** — own your AI infrastructure instead of renting it. Built by Thomas Sprayberry.
