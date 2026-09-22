# The hybrid server

Back to the [README](../README.md).

## The server, as a service

`server.py` speaks enough OpenAI protocol for real clients: **`stream: true` works**
(SSE — role delta, one content delta, a stop chunk, `[DONE]`; the content arrives whole
because routing has to finish before an answer exists), multi-turn conversations route
on the last user message while an **escalated call carries the whole conversation**, and
every response has `x_hybrid` (route / why / backend / latency) plus a chars/4 `usage`
estimate (flagged `usage_estimated` — the local tier isn't token-metered).

Every request writes one **JSONL decision line** — route, why, backend, latency, status,
a sha256 prefix of the query, and `tokens`: the request's **measured token spend per
tier** (`local_in/out/calls`, `frontier_in/out/calls`), read from the backends' own
responses rather than estimated — a SOLVED route logs zeros, which is the point. The
line goes to stdout (the banner goes to stderr) or to `HYBRID_LOG`.
Query text stays out of the log unless `HYBRID_LOG_QUERIES=1`. A backend failure is a
**502 with an OpenAI-shaped error object** (see failure policy), never error text
disguised as an answer. `/health` reports liveness + version without auth; set
`HYBRID_API_KEY` to require a bearer token on everything else, and `HYBRID_HOST` if you
deliberately bind beyond loopback.

**Repeats are free.** Set `HYBRID_CACHE_TTL=300` and a repeated single-turn query is
served from memory in ~0 ms with `x_hybrid.cached: true` — real traffic repeats, and a
cache hit costs neither tokens nor bandwidth. Multi-turn requests, `ERROR` results, and
`DEGRADED` answers are never cached; `HYBRID_CACHE_MAX` (default 512) caps entries, LRU.

**Cold starts are optional.** On a CPU box, prefill is the compute-bound wall, so a
freshly restarted server answers its opening traffic slowly until the prefix cache
fills. `HYBRID_WARMUP=1` primes each fixed local tier's instruction preamble with one
throwaway forward pass at boot — no routing, no frontier spend, and a cold backend
never blocks startup. (The labelled-classifier preamble is caller-dependent and
deliberately not warmed.)

### The Anthropic front door — `POST /v1/messages`

Most fleets are **Anthropic-shaped** — inline `@anthropic-ai/sdk` `messages.create`
calls, or the Claude CLI, all speaking `/v1/messages`. So the server has a second front
door in that shape (non-streaming, text-only), and `x-api-key` auth works alongside
`Authorization: Bearer`. Point an Anthropic client's base URL at hybrid and its cheap
calls route local-first.

The catch it handles: those callers are usually **instruction-following** — the task is
in the `system` prompt (*classify this / extract that / judge this*), not a self-contained
question. hybrid's arithmetic tiers impose their own prompt, so running the solver on the
user text would answer the wrong thing. `route_messages` branches on it:

- **No `system`** → the user turn is a self-contained question → the full router (solver,
  templates, verifier, vote, escalate), verifier and all.
- **A `system` instruction present** → self-consistency on the **caller's own prompt**
  (system + turns) with the local model; unanimous → serve on-box (free), otherwise
  escalate the whole conversation to the frontier. The instruction is respected, and a
  low-confidence local answer is never served — the frontier catches the hard ones.

Measured live against a real 3B, classifying the way a dispatcher does — `build`,
`research`, `monitor` — all three came back **on-box, unanimous, and correct**; an
uncertain one would have escalated. `POST /v1/messages/count_tokens` returns the same
chars/4 estimate for clients that probe it. Streaming and the tool-using agent path are
out of scope here — those need the real model; this door is for the cheap, text-only,
Anthropic-shaped calls a fleet makes by the thousand.

#### Labelled classification — constrain and verify, not vote and hope

Most cheap Anthropic calls are *classification*: "pick one of these labels." The plain
instruction-following vote handles it, but it votes on the raw text, so a rambly local
answer breaks unanimity, and nothing stops the model inventing a label that isn't in
your set. So a request can declare its label set — `metadata.hybrid_labels: ["build",
"research", ...]` (a custom key real Anthropic ignores) — and hybrid switches to a
**constrained-and-verified** path: it grammar-locks the local model to emit *exactly*
one of your labels (GBNF, on the llama.cpp transport), samples it a few times, and
normalizes each sample to the label it contains before voting. A served answer is then
both **self-consistent** and **provably one you declared**; disagreement, or a sample
with no in-set label, escalates. It's the verifier discipline — constrain the output,
verify it's valid — applied to labels instead of arithmetic.

Measured live, grammar-locked, on a real 3B over `["build","research","monitor",
"security"]`: *harden our API against injection* → `security`, *set up a CI/CD pipeline*
→ `build`, *track p99 latency and page me* → `monitor`, *compare vector databases* →
`research` — every one on-box, unanimous, and guaranteed in-set. The model **cannot**
return a category you didn't ask for.

**Enumerate your labels in the system prompt too — this is not optional.**
`metadata.hybrid_labels` grammar-locks the *output* and enables the posterior read,
but the model still has to know what it is choosing between: that comes from your
prompt, not from the metadata. Measured on the deployed gateway, same label set in
both arms, only the system prompt differing:

| system prompt | first-token posterior | routed |
|---|---|---|
| `"Classify into ONE category."` | 0.03 – 0.11 | **0/4 on-box — everything escalated** |
| labels enumerated + *"reply with ONLY the category word"* | 0.91 – 0.98 | **4/4 on-box, ~0.2 s** |

The failure is silent and it is expensive: a diffuse posterior is *correctly* treated
as low confidence, so every request escalates and you pay frontier prices for the
whole classification workload — the exact opposite of the point. If your on-box rate
for a labelled workload is near zero, check the prompt before the model.

**The in-set guarantee is an ON-BOX guarantee.** A locally served label is provably one
you declared — grammar-locked and read off the posterior. An *escalated* one is not:
escalation forwards your conversation to the frontier, which answers under your prompt
like any other request. With the prompt above it returns the category word; with a weak
prompt it returned 1,951 characters of prose in the same test. If you parse the reply,
parse defensively and treat an unrecognized answer as "no classification."

**Ownership labels are facts, not semantics — declare them.** Some label sets encode
*who owns a surface* ("discord things go to the Discord manager"), and a general
model cannot infer that roster fact: measured live, a tuned classifier routed *"the
discord bot stopped responding to slash commands"* to `monitor` at 0.98 posterior —
confidently wrong, with the answer named in the message. So a request may also
declare `metadata.hybrid_label_hints: {"discord": ["discord", "slash command"],
"cloudflare": ["dns", "cdn", "cloudflare"]}`. When exactly one label's lexicon
matches the message (case-insensitive, whole-word, phrases allowed), hybrid serves
that label **deterministically** — `SOLVED`, zero model calls, 0 ms, correct by
construction. Zero hits, or a hit on two labels, falls through to the model path
unchanged: ambiguity degrades to measured behavior, never to a guess. It's the same
rule the arithmetic tiers are built on — never ask a model something you can decide
deterministically — applied to the label set itself.

**Read the posterior, don't sample it.** On the llama.cpp transport the vote itself is
now the fallback: the label set is enumerable, so hybrid reads the model's OWN
first-token probability distribution over it (one forward pass, `n_probs`) and serves
the argmax behind a probability-and-margin gate — `HYBRID_LABEL_MIN_P` (default 0.4)
and `HYBRID_LABEL_MARGIN` (default 2.0); a soft posterior escalates. That is strictly
more information than "k samples at temperature 0.6 agreed," at a third of the forward
passes, and it is deterministic — measured on a real 0.5B, two consecutive passes were
identical, p50 dropped 505→374 ms, and on-box went 5/6→6/6. Every decision logs
`label posterior p1 vs p2`, so the gate can be tuned per family from real traffic.
One honest caveat the read *exposes* rather than causes: a small model's bias class
(mislabeling toward a favorite label) is **calibrated-looking** — wrong at the same
posterior as right — so no fixed threshold removes it; the logged margins against
frontier verdicts are exactly the data that a distilled student needs to remove it
with training instead. `HYBRID_LABEL_LOGITS=0` restores the pure sampling vote.

Capacity honesty: on a CPU box the *model* tiers run **seconds-to-a-minute per query**
and effectively serially — that's memory bandwidth, not a bug. The solver and template
tiers answer in ~0 ms regardless, and `LOCAL_MODEL_FAST` roughly halves the vote/creative
tiers. Size expectations (and any reverse proxy timeouts) for the residual model-path
queries accordingly.

Deploying it: the **`Dockerfile`** is python-slim plus the modules (with a
`/health` healthcheck); **`deploy/docker-compose.yml`** runs the whole local tier —
ollama + hybrid — with the port published to loopback only; **`deploy/hybrid.service`**
is a hardened systemd unit (`DynamicUser`, `ProtectSystem=strict`) where
`journalctl -u hybrid` *is* the decision log. This compose shape is how it ran on our
own box from June to September 2026 (a 2013-era CPU host, gateway + llama.cpp fast tier
under compose, 400 logged decisions) before we retired that deployment — the only
caller was a chat-message classifier that never saw real traffic. The measurements in
this README come from that period.

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
| `HYBRID_WARMUP` | off | `1` = prime each fixed local tier's instruction preamble at boot (one throwaway pass; fills the prefix cache) |
| `HYBRID_SLOT_PIN` | `1` | llamacpp transport: pin prompt families to a server slot (needs `GET /slots`); `0` disables |
| `HYBRID_LABEL_LOGITS` | `1` | labelled classification reads the first-token posterior (one pass) instead of voting; `0` = sampling vote |
| `HYBRID_LABEL_MIN_P` | `0.4` | minimum posterior mass on the winning label to serve locally |
| `HYBRID_LABEL_MARGIN` | `2.0` | winning label must carry ≥ this × the runner-up's mass |

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

### Load shedding — the production tier

On a CPU box the model tiers cost seconds, and decode is memory-bandwidth-bound, so
**two model requests at once don't run twice as fast — they queue on the same memory
bus.** Making the second caller wait 40 seconds is worse than escalating them. So under
load, hybrid sheds the expensive local work to the frontier instead of queueing it. Both
signals are **off by default** (behavior unchanged), and the deterministic tiers
(`solve`/`template`) *never* shed — they cost nothing and answer regardless:

- **`HYBRID_MODEL_MAX_INFLIGHT=N`** — run at most `N` model-tier requests at once; past
  the cap, escalate immediately. `N=1` is the honest setting for a one-box CPU deploy:
  serve one model query locally, send the rest to the frontier. A shared, thread-safe
  in-flight gauge (exposed as `model_inflight` on `/health`) is the capacity signal — a
  concurrent request checks it and sheds *before* it can queue.
- **`HYBRID_LATENCY_BUDGET_MS=ms`** — a per-request wall-clock budget. If the time
  already spent plus the estimated cost of a model tier (`HYBRID_MODEL_TIER_MS`, default
  8000, scaled by how many calls are queued ahead) would blow it, shed. This turns "the
  box is slow" into "the box answers what it can inside your SLA and escalates the rest."

A shed is an ordinary escalation — it carries the whole conversation, obeys the frontier
failure policy, and logs its reason (`route: ESCALATE`, `why: "load shed: 1 model call(s)
in flight, cap 1 -> frontier"`). Measured live on the 2013 box with `MAX_INFLIGHT=1`: two
concurrent model-path queries — the first re-derived its answer on the 7B locally, the
second, arriving while that slot was held, went straight to the frontier instead of
waiting behind it. That is the difference between a demo and something you can put real
concurrent traffic through.
