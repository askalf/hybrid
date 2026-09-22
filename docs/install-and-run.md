# Installing and running hybrid

Back to the [README](../README.md).

## Install

**Not yet on PyPI.** `publish.yml` ships a release via Trusted Publishing (OIDC — no
tokens anywhere) the moment PyPI has a pending-publisher binding registered for this
repo; until then it fails closed on every release rather than claim success it didn't
earn. Until that's done, install straight from the repo — same zero-dependency wheel,
same console commands:

```bash
pipx install git+https://github.com/askalf/hybrid   # console commands: hybrid, hybrid-server
# or: pip install git+https://github.com/askalf/hybrid
```

Zero runtime dependencies — the wheel is the modules you can read in this repo, installed
exactly as they read. `hybrid --version` tells you what you got.

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

python solver.py "how many feet in 3 miles"    # the deterministic tier alone -> 15840
python templates.py "A printer prints 2,417 pages per hour. How many pages in 94 hours?"
                                               # the template transcriber alone -> ('227198', 'rate')
python bench_router.py                         # full-router benchmark: on-box %, safety, catches
python measure_routing.py                      # router economics: $ saved vs all-frontier (needs FRONTIER_API_KEY)
python hybrid.py "your question"               # route one query
python hybrid.py --demo                        # mixed test set + summary
python server.py                               # OpenAI-compatible server on :8080 (model "hybrid", stream ok)
```

The oracle tiers and every test harness need **nothing** — no model, no network — so all
**365 tests** run anywhere, including CI: `test_solver.py` 53 · `test_templates.py` 58 ·
`test_verify.py` 28 · `test_equations.py` 45 · `test_route.py` 39 · `test_server.py` 23 ·
`test_backend.py` 64 · `test_messages.py` 32 · `test_tokens.py` 14 · `test_warmup.py` 9.

## The llama.cpp transport — the GPU-less fast path

Ollama is the friendly default; llama.cpp's own server is the fast one. Point hybrid at
a [`llama-server`](https://github.com/ggml-org/llama.cpp) and the router turns on four
things Ollama's generate API can't express, all aimed at the two places CPU inference
actually hurts:

```bash
llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf -c 12288 --parallel 3 --port 8080
export HYBRID_LOCAL_BACKEND=llamacpp        # LLAMACPP_URL if not :8080/completion
python server.py

# optional second server: the split-model policy, on llama.cpp. The vote and
# creative tiers decode on a 3B (~2.8x the 7B at the same memory bandwidth);
# transcription stays on the 7B - the pinned safety rule, enforced by URL:
llama-server -m qwen2.5-3b-instruct-q4_k_m.gguf -c 8192 --parallel 3 --port 8081
export LOCAL_MODEL_FAST=qwen2.5:3b LLAMACPP_URL_FAST=http://127.0.0.1:8081/completion
# ^ measured before you trust it: on one llama.cpp build the 3B voted a wrong
#   World-Cup winner 3/3 — the fast-model trade is runtime-specific. See below.
```

- **Prefix caching** (`cache_prompt`). Each tier's instruction preamble is fixed; only
  the question changes. The transport puts the instructions in the system slot so
  llama-server prefills them ONCE — after the first call, a transcription call prefills
  ~24 tokens instead of ~128. On a CPU, prefill is the *compute*-bound wall that no
  quantization fixes; measured on a 2013 Haswell this is ~5 s back **per call**.
- **GBNF grammars.** The transcription tiers are sampled under a grammar that can only
  produce the line shapes the oracles parse (`EQN:` / `ANSWER:` / `CHECK:`). This kills
  the ramble class outright — the same 7B that answers a rate problem in 23 tokens
  will, unconstrained, write 210 tokens of LaTeX the parsers can't read (measured:
  54 s → 6.5 s on the worst live case) and then cost a SECOND call as the tier falls
  through unparsed. A grammar also cannot write units inside a `CHECK:` line, which
  was a documented fall-through class. `HYBRID_GRAMMAR=0` turns it off.
- **Slot pinning** (prompt families). `cache_prompt` only helps when the matching KV is
  in the slot a request lands on — and llama-server spreads unpinned requests across
  slots, so a classifier that always sends the same system prompt keeps re-prefilling
  it in whichever slot each request hits. Worst case is the k-sample vote: k IDENTICAL
  prompts land on k slots and prefill k times at once. The transport pins each **prompt
  family** (labelled classification: system prompt + label set) to one slot, so sample
  1 prefills, samples 2..k reuse the whole prompt, and the prefix stays hot for the
  family's next request. Measured (3B, `--parallel 3`, six dispatcher-shaped classify
  requests): cold 9434 → 3571 ms (**2.6×**), warm p50 3175 → 1623 ms (**2.0×**), with
  identical labels chosen — it moves work, it never changes answers. Applied only where
  decode is tiny (grammar-locked labels); long-decode votes still batch across slots.
  Needs `GET /slots` exposed (llama-server's default); `HYBRID_SLOT_PIN=0` disables.

#### Same box, same GGUF, two transports (i7-4770 8-thread, qwen2.5:7b Q4_K_M, frontier stubbed)

| transport | bench 22q on-box | bench safety | bench wall | stress 26q on-box | stress wrong-served | stress wall |
|---|---|---|---|---|---|---|
| Ollama (standalone, Jun '26 build) | 18/22 | 17/18 | 3 m 23 s | 23/26 | 1 (documented class) | 9 m 59 s |
| llama.cpp b9660, this transport | 18/22 | 16/18 | **2 m 03 s** | 23/26 | 1 (documented class) | **4 m 23 s** |

Same day, same hardware, same weights: **1.65–2.3× end-to-end** from the prefix cache
plus grammar-shortened outputs, at safety parity on the stress set — every wrong-served
answer in every leg is the documented runtime-fragile trap class (see below), and WHICH
member of that class slips varies by runtime build, not by transport feature.

There is also an **experimental fused tier** (`HYBRID_FUSE=1`): equations + answer +
substitution checks in ONE call, read strongest-signal-first with the same precedence
as the two-call flow. It measured ~2.2× end-to-end — and then measured *why it stays
off by default*: asking one call to transcribe AND self-check degrades the
transcription itself. A mixed-unit conversion the setup tier transcribes correctly
(5 ft 4 in → 162.56 cm) came back mangled (5.33), a percent answer lost its unit, and
the plug-back tier then graded the mangled answer's true-but-disconnected arithmetic
as "checked". One call is only cheaper if its answers stay worth serving.

Two honest wrinkles worth knowing. llama-server **silently ignores** a grammar it
can't parse (it logs and generates unconstrained) — so hybrid's grammars are pinned by
tests (`test_backend.py`) rather than trusted at runtime. And the raw `/completion`
endpoint bypasses the GGUF's chat template, so the transport wraps prompts itself —
`HYBRID_PROMPT_WRAP` (default ChatML, the Qwen family) if your model speaks another
dialect.

**And one honest finding that outranks both:** the classic setup traps
(chicken-and-a-half, Sally's-sisters) turn out to be **runtime-fragile** at
temperature 0 — the same model, same prompts, flipped between cracked / caught /
wrong-served across llama.cpp builds and transports, with or without grammars, at any
temperature, under every prompt wrap we tried. A "zero wrong served" number on that
trap class is a fact about one runtime build, not about the router. The tiers that are
runtime-STABLE are exactly the deterministic ones — the solver and the template
transcriber, which answer the shaped majority at 0 ms with no model in the loop — and
that, not benchmark luck, is the durable safety story. Bench tables should state the
runtime; ours do.
