# hybrid measurements

Back to the [README](../README.md).

Router benchmark and routing economics, with the version each was run at.

## Measured (`bench_router.py`, 22-query labeled set, qwen2.5:7b — run at v1.5.0)

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

The line moved three times across v1.0.0 → v1.5.0. v1.0.0: 15/20 on-box but **13/15 correct** — both
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

> **Vintage.** Both benchmark blocks above were last run against v1.5.0. Releases
> since then changed the transport and the classifier rather than the tier
> boundaries — llama.cpp transport (v1.7.0), load shedding (v1.8.0), labelled and
> then logit-read classification (v1.10.0, v1.12.0), slot pinning (v1.11.0),
> startup warmup and token accounting (v1.13.0). Re-run `bench_router.py` and
> `measure_routing.py` yourself for current numbers; the routing *shape* above is
> what these figures are here to show.

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
