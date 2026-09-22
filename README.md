<div align="center">

# hybrid

**Own your inference: answer the easy majority of LLM queries on your own machine — free, private, exact — and escalate only the few that earn a frontier call.**

Dependency-free Python. Built and measured on a GPU-less 2013 desktop.

[![tests](https://github.com/askalf/hybrid/actions/workflows/test.yml/badge.svg)](https://github.com/askalf/hybrid/actions/workflows/test.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/askalf/hybrid/badge)](https://scorecard.dev/viewer/?uri=github.com/askalf/hybrid)

[Quickstart](#quickstart) · [How it routes](#how-it-routes) · [Measured](#measured) · [Reference](#reference)

<img src="https://raw.githubusercontent.com/askalf/hybrid/main/og.png" alt="hybrid: answer the easy majority locally, escalate the few that earn it" width="840">

</div>

---

## The result that reframes the problem

*"Each crate weighs 23.7 kg. What do 41 crates weigh?"* — truth: **971.7**.

| answerer | answer | cost |
|---|---|---|
| local `qwen2.5:7b` | 981.7 ✗ — caught by the verifier → escalated (working as designed) | 1 local call |
| `claude-haiku-4-5` (a real frontier tier) | **972.17 ✗ — deterministically: 5/5 identical wrong answers** across resamples | 5 frontier calls |
| **hybrid's template tier** | **971.7 ✓ exact** | **0 ms · zero model calls** |

Reproduce the free row yourself: `python templates.py "Each crate weighs 23.7 kg. What do 41 crates weigh?"` → `('971.7', 'rate')`.

*Scope, honestly:* one query class (rate × decimal quantity), one query × 5 samples, measured 2026-08-10, and the frontier tier was haiku — the cheap one — not a top-end model. Don't read it as "frontier models are bad at math." Read it as the thesis: **arithmetic execution is unreliable in language models generally — including the model you'd escalate to — so for the shapes you can recognize, don't ask a model at all. Compute it.** Escalation is a quality lever, not a guarantee.

That's what hybrid is: a router whose strongest tiers are **not models** — exact arithmetic, deterministic transcription, and a free verifier stronger than any model on the slice it covers — with a local model for the middle and a frontier only for what genuinely earns it. Most of what you ask an LLM is easy; the writeup with all the numbers is [**Your CPU isn't bad at LLMs — it's bandwidth-starved**](https://sprayberrylabs.com/blog/own-your-inference).

The hard part isn't routing the easy queries home — it's knowing when the cheap model is **confidently wrong**. A router built on the cheap model's own signals (classification, self-consistency) inherits its blind spots: it can't tell confident-and-right from confident-and-wrong. hybrid's answer is a **free verifier that is stronger than the model** — Python's exact arithmetic — applied at every depth it can reach: solve the closed forms outright, transcribe the *shaped* word problems deterministically, and re-derive the model's own working on everything else.

## Quickstart

```bash
pipx install git+https://github.com/askalf/hybrid        # zero runtime dependencies
ollama pull qwen2.5:7b                                    # the local tier
hybrid "A printer prints 2,417 pages per hour. How many pages in 94 hours?"
```

The oracle tiers need no model at all: `python solver.py "how many feet in 3 miles"` → `15840`. Set `FRONTIER_API_KEY` for the escalation tier, or run `hybrid-server` for an OpenAI- and Anthropic-compatible endpoint. Full setup, the llama.cpp fast path and the benchmarks: [install and run](https://github.com/askalf/hybrid/blob/main/docs/install-and-run.md).

## How it routes

```
query → router ─┬─ solve:    arithmetic · unit conversion · %-change? ▶ SOLVED  (python, exact, free)
                ├─ template: a word-problem SHAPE we recognize outright ▶ SOLVED  (python, exact, free)
                ├─ rule:     hard category (code/proof/puzzle) ───────▶ ESCALATE
                ├─ rule:     open-ended (rewrite/summarize) ──────────▶ LOCAL
                ├─ derive:   the model writes EQUATIONS; we solve them exactly
                │              ─▶ LOCAL if they re-derive its answer, ESCALATE if not
                ├─ verify:   the model's numbers plugged back into the problem
                │              ─▶ LOCAL if every check holds, ESCALATE if any is false
                └─ vote:     local self-consistency ─▶ LOCAL if unanimous, else ESCALATE
```

Every tier, with the reasoning and the failure cases it exists for: [how it routes](https://github.com/askalf/hybrid/blob/main/docs/how-it-routes.md).

## Measured

`bench_router.py` over a 22-query labeled set, qwen2.5:7b on an 8-core CPU box, run at v1.5.0:

```
ON-BOX:        17/22 (77%) answered without a frontier call
ON-BOX SAFETY: 17/17 on-box answers correct        (ZERO wrong answers served)
TEMPLATE:      7/7 shaped word problems answered exact, zero model calls
CATCHES:       2/2 confident-wrong arithmetic intercepted -> escalated
```

On a fresh 24-query holdout the same build kept **22/24 on-box (92%)**. Priced against a frontier model, roughly half the dollars stay home, not three-quarters, because the queries that escalate are the token-heavy ones. The full blocks, the holdout, the economics and the vintage of each number: [measurements](https://github.com/askalf/hybrid/blob/main/docs/measurements.md).

## Reference

- **[How it routes](https://github.com/askalf/hybrid/blob/main/docs/how-it-routes.md)**: the deterministic solver, templates, equation derivation, verifier and vote, tier by tier.
- **[Measurements](https://github.com/askalf/hybrid/blob/main/docs/measurements.md)**: the router benchmark, the holdout, and routing economics.
- **[Install and run](https://github.com/askalf/hybrid/blob/main/docs/install-and-run.md)**: install, every command, the test suites, and the llama.cpp transport for GPU-less boxes.
- **[The server](https://github.com/askalf/hybrid/blob/main/docs/server.md)**: the OpenAI-compatible and Anthropic `/v1/messages` front doors, config, failure policy and load shedding.
- **[The honest part](https://github.com/askalf/hybrid/blob/main/docs/lessons.md)**: what building this taught, including what didn't work.
- **[Files](https://github.com/askalf/hybrid/blob/main/docs/files.md)**: what each module does.

## Own Your Stack

Part of **[Own Your Stack](https://github.com/askalf)** — open tools for owning your AI infrastructure instead of renting it by the token. One subscription. Your box. Your terms.

- **[dario](https://github.com/askalf/dario)** — own your routing
- **[hybrid](https://github.com/askalf/hybrid)** — own your inference _(you are here)_
- **[browser-bridge](https://github.com/askalf/browser-bridge)** — own your browser
- **[redstamp](https://github.com/askalf/redstamp)** — own your agent security
- **[truecopy](https://github.com/askalf/truecopy)** — own your agent skills
- **[cordon](https://github.com/askalf/cordon)** — own your prompts
- **[fieldpass](https://github.com/askalf/browser-bridge/tree/master/policy)** — own your agent browser
- **[amnesia](https://github.com/askalf/amnesia)** — own your search
- **[askalf](https://askalf.org)** — own your operation: the AI operation that runs Sprayberry Labs

---
Part of **[Own Your Stack](https://github.com/askalf)** — own your AI infrastructure instead of renting it by the token. Built by Thomas Sprayberry · MIT.
