# hybrid — local-first LLM routing with frontier escalation

<p align="center">
  <img src="og.png" alt="hybrid: answer the easy majority locally, escalate the few that earn it" width="840">
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
model** — Python's exact arithmetic — applied at two depths.

## How it routes

```
query → router ─┬─ solve:   arithmetic · unit conversion · %-change? ▶ SOLVED  (python, exact, free)
                ├─ rule:    hard category (code/proof/puzzle) ────────▶ ESCALATE
                ├─ rule:    open-ended (rewrite/summarize) ───────────▶ LOCAL
                ├─ verify:  has a number? local answers + plugs its
                │             numbers into the problem's relationships;
                │             re-derive each exactly ─▶ LOCAL if every check holds,
                │                                       ESCALATE if any is false
                └─ vote:    local self-consistency ─▶ LOCAL if unanimous, else ESCALATE
```

0. **Deterministic solver** (`solver.py`) answers what a cheap model gets *confidently
   wrong* — closed-form arithmetic, plus exact **unit conversions** (`3 miles → 15840 feet`,
   via `1 in = 25.4 mm` with `Fraction`s, never a float), **percentage-change**
   (`20% off 50 → 40`), and **multiples** (`half of 60 → 30`). Zero frontier calls, correct
   by construction. Strictly conservative — anything that doesn't reduce cleanly falls through.
1. **Category rules** escalate domains a small model is *known* to fail (code, proofs, puzzles).
2. **Open-ended rules** keep creative tasks (rewrite, summarize) local — no single right answer.
3. **Verify-the-local-answer** (`verify.py`) — for any query with a number, the local model
   answers and **plugs its own numbers back into the problem's relationships**, writing
   pure-numeric checks we re-derive exactly. A false check is a **hard escalate** (the answer
   is provably inconsistent with the problem); all-checks-hold stays local. Strictly stronger
   than self-consistency, which at temperature 0 just repeats the same wrong number.
4. **Self-consistency** for the rest: answer a few times; unanimous → keep local, else escalate.

## Measured (`bench_router.py`, 20-query labeled set, qwen2.5:7b)

The real router over a labeled mix — closed-form, conversions, factual, word problems,
confident-wrong arithmetic, hard, and setup traps. Frontier escalation is stubbed, so the
benchmark is free:

```
ON-BOX:        15/20 (75%) answered without a frontier call
ON-BOX SAFETY: 13/15 on-box answers correct        (the 2 wrong are the setup traps below)
CATCHES:       3/3 confident-wrong products intercepted -> escalated
ESCALATED:     5/20 routed to the frontier
HONEST LIMIT:  2 setup traps slipped through local + wrong (known boundary)
```

Three-quarters answered free, and **the only wrong answers served on-box are two documented
setup traps** (chicken-and-a-half, Sally's-sisters). Every other on-box answer is correct,
and all three confident-wrong multiplications were caught and escalated. A few of the rows,
verbatim:

```text
SOLVED     How many feet in 3 miles?                 -> 15840          (exact, free)
SOLVED     What is 20% off 50?                       -> 40             (exact, free)
LOCAL      7 notebooks at $12.50                      -> $87.50         (checks hold)
LOCAL      bat and ball, bat $1 more                  -> 0.05           (0.05+1.00=1.05; 1.05-0.05=1.00)
ESCALATE   1,847 widgets/day for 263 days             -> caught: 1847*263 = 485061 ≠ 485761
LOCAL      chicken-and-a-half                         -> "½ egg/day"    (WRONG — the honest limit)
```

That ESCALATE row is the point: at temperature 0 the model states `485061` on every
sample, so self-consistency would call it *unanimously confident*. The verifier re-derives
`1847 * 263` and escalates instead. **Arithmetic execution is where a small model stays
wrong even when it reasons well — and exactly where a free exact oracle wins.**

## Run

```bash
# local tier — Ollama with a small model
ollama pull qwen2.5:7b           # the measured default; qwen2.5:3b is faster but follows
                                 # the verify-CHECK format less reliably

# frontier tier — any OpenAI-compatible endpoint
export FRONTIER_API_KEY=sk-...                                    # OpenAI, or your own proxy
export FRONTIER_URL=https://api.openai.com/v1/chat/completions    # default; point anywhere OpenAI-compatible
export FRONTIER_MODEL=gpt-4o                                      # default

python solver.py "how many feet in 3 miles"   # the deterministic tier alone -> 15840
python test_solver.py                          # solver tests (50/50, no model needed)
python test_verify.py                          # verifier tests (28/28, no model needed)
python bench_router.py                         # full-router benchmark: on-box %, safety, catches
python hybrid.py "your question"               # route one query
python hybrid.py --demo                        # mixed test set + summary
python server.py                               # OpenAI-compatible server on :8080 (model "hybrid")
```

The solver and verifier tiers (and their tests) need **nothing** — no model, no network —
so they run and test anywhere. The server returns an `x_hybrid` field (route / why /
backend / latency), so any OpenAI client (Cursor, Cline, scripts) gets local-first +
escalation transparently and can see which tier answered.

### Config (env)

| var | default | |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | local Ollama endpoint |
| `LOCAL_MODEL` | `qwen2.5:7b` | local model tag |
| `FRONTIER_URL` | `https://api.openai.com/v1/chat/completions` | any OpenAI-compatible endpoint |
| `FRONTIER_API_KEY` | — | required for escalation |
| `FRONTIER_MODEL` | `gpt-4o` | frontier model id |
| `PORT` | `8080` | server.py listen port |

`FRONTIER_URL` is just an OpenAI-compatible chat endpoint — OpenAI, a local proxy, or your
own gateway. The key only ever leaves your machine on an *escalated* query.

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

**What still gets through — kept visible, not papered over.** The oracle checks the answer
against the relationships the model *transcribes*; it cannot check the *setup*. A
self-consistently-wrong setup (the chicken-and-a-half rate trap) slips through local and
wrong, because the model's own check restates its own misreading. We even tested the obvious
cheap escape — a *second* small model as an independent vote — and it shares the classic blind
spots (both models miss the same famous traps) while over-escalating when the weaker one is
merely vaguer. A second cheap model is still a cheap-model signal. Cracking wrong-*setup*
without a frontier call remains open; `--demo` and `bench_router.py` keep the limit on screen.

## Files

- `hybrid.py` — router + dispatch + `--demo`
- `solver.py` — deterministic arithmetic + exact unit/percentage/multiple conversion (the SOLVED tier)
- `verify.py` — verify-the-local-answer: re-derive the model's plugged-in checks exactly
- `test_solver.py` / `test_verify.py` — solver (50/50) and verifier (28/28) tests; no model needed
- `bench_offline.py` — what the solver buys versus a no-solver router (no model needed)
- `bench_router.py` — full-router benchmark: on-box rate, on-box safety, catches (frontier stubbed)
- `server.py` — OpenAI-compatible front end

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
- **[askalf platform](https://askalf.org)** — own your operation

---
Part of **[Own Your Stack](https://github.com/askalf)** — own your AI infrastructure instead of renting it. Built by Thomas Sprayberry.
