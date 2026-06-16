# hybrid — local-first LLM routing with frontier escalation

Answer the easy majority of your LLM queries on a **small local model** (free,
private, fast); escalate only the genuinely hard ones to a **frontier model**.
Most of what you ask an LLM is easy — facts, rewrites, simple Q&A — and a small
local model nails those. The rare hard query (a proof, real code, multi-step
reasoning) goes to the frontier. Frontier quality where it matters; nothing paid
or sent off your machine for the rest.

~160 lines of dependency-free Python (stdlib only). Built and measured on a
**GPU-less 2013 desktop** — the writeup, with all the numbers, is here:
[**Your CPU isn't bad at LLMs — it's bandwidth-starved**](https://sprayberrylabs.com/blog/own-your-inference).

## How it routes

```
query → router ─┬─ rule: hard category (code/proof/puzzle/powers) ─▶ ESCALATE
                ├─ rule: open-ended (rewrite/summarize)            ─▶ LOCAL
                └─ verify: local self-consistency  ─▶ LOCAL if unanimous, else ESCALATE
```

1. **Category rules** escalate domains a small model is *known* to fail (code,
   proofs, puzzles, powers/roots/factorials) — no point trying locally.
2. **Open-ended rules** keep creative tasks (rewrite, summarize) local — there's
   no single right answer, so the verify step would mis-fire on valid variation.
3. **Verify-then-escalate** for everything else: the local model answers a few
   times; *unanimous* → confident (keep local), otherwise → uncertain (escalate).

## Run

```bash
# local tier — Ollama with a small model
ollama pull qwen2.5:3b

# frontier tier — any OpenAI-compatible endpoint
export FRONTIER_API_KEY=sk-...                                    # OpenAI, or your own proxy
export FRONTIER_URL=https://api.openai.com/v1/chat/completions    # default; point anywhere OpenAI-compatible
export FRONTIER_MODEL=gpt-4o                                      # default

python hybrid.py "your question"   # route one query
python hybrid.py --demo            # mixed test set + summary
python server.py                   # OpenAI-compatible server on :8080 (model "hybrid")
```

The server returns an `x_hybrid` field (route / why / backend / latency), so any
OpenAI client (Cursor, Cline, scripts) gets local-first + escalation transparently
and can see which tier answered.

### Config (env)

| var | default | |
|---|---|---|
| `OLLAMA_URL` | `http://127.0.0.1:11434/api/generate` | local Ollama endpoint |
| `LOCAL_MODEL` | `qwen2.5:3b` | local model tag |
| `FRONTIER_URL` | `https://api.openai.com/v1/chat/completions` | any OpenAI-compatible endpoint |
| `FRONTIER_API_KEY` | — | required for escalation |
| `FRONTIER_MODEL` | `gpt-4o` | frontier model id |
| `PORT` | `8080` | server.py listen port |

`FRONTIER_URL` is just an OpenAI-compatible chat endpoint — OpenAI, a local proxy,
or your own gateway. The key only ever leaves your machine on an *escalated* query.

## The honest part (what this taught me)

The interesting finding isn't that it works — it's *where the routing fails*, and
why that's hard:

**A cheap router inherits the cheap model's blind spots — even with verification.**
Self-consistency (answer a few times, escalate on disagreement) catches genuine
*uncertainty*, but it **cannot catch *confident wrongness*.** A 3B model answered
"17⁴ = 6859" *unanimously* (it's 83,521) and walked into a classic rate-trap with
3/3 agreement on the wrong answer — confident, consistent, and wrong. A router
built on the cheap model's *own* signals (classification, self-consistency,
self-assessment) inherits its blind spots; it can't tell confident-and-right from
confident-and-wrong. The only escapes are **category rules** for known-weak domains
(what the rules above do) or a **verifier stronger than the model** (≈ a frontier
call, which defeats the savings for that query). There is no free lunch — which is
exactly why router quality is the open problem in hybrid systems.

This repo keeps that limit *visible* rather than papering over it. `--demo`
includes the trap.

## Files

- `hybrid.py` — router + dispatch + `--demo`
- `server.py` — OpenAI-compatible front end

## License

MIT
