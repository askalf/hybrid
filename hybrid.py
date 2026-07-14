#!/usr/bin/env python3
"""
hybrid — local-first LLM router with frontier escalation.

  EASY queries  -> a small local model (Ollama)                  free, private, fast
  EXACT queries -> a deterministic Python oracle                 free, instant, correct
  HARD queries  -> a frontier model (any OpenAI-compatible API)  quality where it counts

The hard part of local-first routing isn't sending the easy queries home — it's knowing
when the cheap model is *confidently wrong*. A router built on the cheap model's own
signals (classification, self-consistency) inherits its blind spots. hybrid's answer is a
free verifier that is *stronger* than the model — Python's exact arithmetic — applied at
three depths:

  - SOLVE  closed-form arithmetic / unit conversions / %-change exactly, on-box (solver.py).
  - TEMPLATE a word problem whose SHAPE we recognize outright (rate x quantity, bat-and-ball
    pairs, reverse-percentage, ...): deterministic transcription + exact closed form — no
    model, no tokens, no latency, correct by construction (templates.py).
  - DERIVE a word problem's answer independently: the model transcribes the problem's
    relationships as equations, we solve the linear system exactly; a contradiction with
    the model's own answer is a hard escalate (equations.py).
  - VERIFY a numeric answer by having the model plug its numbers back into the problem's
    relationships and re-deriving each exactly; a false check is a hard escalate (verify.py).

Everything the oracle can't settle falls back to self-consistency, then the frontier.

  python hybrid.py "your question"   # route one query
  python hybrid.py --demo            # mixed test set + summary

Config (env):
  OLLAMA_URL        default http://127.0.0.1:11434/api/generate
  LOCAL_MODEL       default qwen2.5:7b — the TRANSCRIPTION model (derive/verify tiers)
  LOCAL_MODEL_FAST  default LOCAL_MODEL — a smaller model for the vote/creative tiers
                    only. Measured live: a 3B is safe (and ~2x faster) on factual votes
                    and rewrites, but tripled wrong-served answers when allowed to
                    TRANSCRIBE — it writes wrong-but-consistent equation systems the
                    exact oracle then faithfully re-derives. Transcription stays on
                    LOCAL_MODEL; never point LOCAL_MODEL_FAST at the derive/verify path.
  FRONTIER_URL      default https://api.openai.com/v1/chat/completions  (any OpenAI-compatible endpoint)
  FRONTIER_API_KEY  required for escalation
  FRONTIER_MODEL    default gpt-4o

Failure policy (env) — a dead backend degrades predictably instead of crashing the query:
  HYBRID_ON_LOCAL_FAIL     escalate (default) | error
                           "escalate": the local model is unreachable -> send the query to
                           the frontier rather than failing it.
  HYBRID_ON_FRONTIER_FAIL  error (default) | local
                           "error" is the honest default: a query the router decided needs
                           the frontier gets an ERROR, never a silent local answer.
                           "local" trades correctness for availability — it serves a plain
                           local answer labelled degraded, INCLUDING for queries whose local
                           answer the verifier just caught as wrong. Opt in knowingly.

Load shedding (env) — under load, ESCALATE the expensive local work instead of queueing
it. Both OFF by default (behavior unchanged); the deterministic tiers (solve/template)
never shed — they cost nothing and answer regardless.
  HYBRID_MODEL_MAX_INFLIGHT  0 (off) | N
                           Run at most N model-tier requests at once. On a memory-
                           bandwidth-bound CPU a single decode saturates the bus, so the
                           N+1th request would not run faster — it would queue behind the
                           others. Past the cap, shed to the frontier now. N=1 is the
                           honest setting for a one-box CPU deploy: serve one model query
                           locally, send the rest up.
  HYBRID_LATENCY_BUDGET_MS   0 (off) | ms
                           A per-request wall-clock budget. If the time already spent plus
                           the estimated cost of a model tier (HYBRID_MODEL_TIER_MS, scaled
                           by how many calls are queued ahead) would exceed it, shed. Turns
                           "the box is slow" into "the box answers what it can inside your
                           SLA and escalates the rest".
  HYBRID_MODEL_TIER_MS       8000 — the estimated wall cost of one model tier, for the
                           budget projection. Set it to your box's measured p50.

Slot pinning (llamacpp transport) — treat a request's stable instruction prefix as a
PROMPT FAMILY and pin the family to one server slot, so its prefill survives in that
slot's KV cache instead of being redone per request (and per vote sample). Used only
where decode is tiny and the prefix dominates (labelled classification); long-decode
votes stay unpinned — they benefit more from batching across slots.
  HYBRID_SLOT_PIN            1 (default) | 0
                           0 disables pinning entirely. With 1, pinning still engages
                           only when the server exposes GET /slots (llama-server default;
                           --no-slots turns it off) — otherwise behavior is unchanged.

Logit-read classification (llamacpp transport) — the label set is enumerable, so
labelled classification reads the model's first-token posterior over it in ONE forward
pass (n_probs) instead of sampling k times and voting. Deterministic, a third of the
passes, and the decision log carries the two probabilities (calibration data). The
sampling vote remains the automatic fallback (ollama transport, no top_logprobs, or
any malformed response).
  HYBRID_LABEL_LOGITS        1 (default) | 0 — 0 restores the pure sampling vote
  HYBRID_LABEL_MIN_P         0.4 — minimum posterior mass on the winner to serve
  HYBRID_LABEL_MARGIN        2.0 — winner must carry >= this x the runner-up's mass
"""
import sys, os, time, json, re, threading, urllib.request, hashlib, math
import contextvars
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from solver import solve
import equations
import templates
import verify

__version__ = "1.13.0"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 chokes on non-ASCII
except Exception:
    pass  # best-effort console tweak; stdout may be replaced/unreconfigurable (tests, pipes)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
# Local transport: "ollama" (default, /api/generate) or "llamacpp" (llama-server's native
# /completion). llama.cpp buys three things Ollama's generate API can't give the router:
#   - cache_prompt: each tier's fixed instruction preamble is prefilled ONCE and reused —
#     on a CPU, prefill is the compute-bound wall, and this measured ~5s/call on a 2013
#     Haswell (128-token preamble -> 24-token warm prefill).
#   - grammar: GBNF-locked EQN/CHECK/ANSWER output. The failure it kills is the RAMBLE:
#     the same 7B that answers a rate problem in 23 tokens will, unconstrained, write 210
#     tokens of LaTeX the parsers can't read (measured 54s -> 6.5s) and then cost a SECOND
#     call when the tier falls through.
# (A fourth idea — FUSING setup+verify into one call — is implemented but
# experimental and OFF by default: measured, the double duty degrades the
# transcription itself. See _fused().)
LOCAL_BACKEND = os.environ.get("HYBRID_LOCAL_BACKEND", "ollama")
LLAMACPP_URL = os.environ.get("LLAMACPP_URL", "http://127.0.0.1:8080/completion")
# llama-server loads ONE model, so the split-model policy (LOCAL_MODEL_FAST for the
# vote/creative tiers, never for transcription) needs a SECOND server: point
# LLAMACPP_URL_FAST at a llama-server holding the fast model and every call the tiers
# make with model=LOCAL_MODEL_FAST routes there. Decode is bandwidth-bound, so a 3B
# answers a vote sample ~2.8x faster than the 7B for the same bytes/sec. Unset, all
# calls go to LLAMACPP_URL — single-server behavior, unchanged.
LLAMACPP_URL_FAST = os.environ.get("LLAMACPP_URL_FAST", "")
# Chat template for the llamacpp transport (raw /completion bypasses the GGUF's built-in
# template). Default is ChatML (the Qwen family). {sys} = tier instructions -> a stable,
# cacheable prefix; {user} = the query.
PROMPT_WRAP = os.environ.get(
    "HYBRID_PROMPT_WRAP",
    "<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:7b")
# The vote/creative tiers tolerate a smaller, faster model: their outputs are either
# voted on or have no single right answer. The transcription tiers do NOT (a weaker
# model's garbled-but-consistent equations sail through the oracle), so they always
# use LOCAL_MODEL.
LOCAL_MODEL_FAST = os.environ.get("LOCAL_MODEL_FAST") or LOCAL_MODEL
FRONTIER_URL = os.environ.get("FRONTIER_URL", "https://api.openai.com/v1/chat/completions")
FRONTIER_KEY = os.environ.get("FRONTIER_API_KEY", "")
FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "gpt-4o")


class BackendError(RuntimeError):
    """A model backend failed after retries. `tier` is "local" or "frontier". Raised by
    ollama()/escalate(); route() catches it and applies the failure policy, so callers
    of route() never see an exception — they see a routed result or an ERROR result."""

    def __init__(self, tier, detail):
        super().__init__(detail)
        self.tier = tier

CONCISE = "Answer directly and concisely - a number, a word, or one or two sentences, no working shown.\nQuestion: {q}"

# Verify-the-local-answer prompt. The small model, asked for "VARS/CHECK", tends to write
# symbolic algebra (`x + 1.00 = 1.10`) that never reduces to a number — so instead we ask it
# to PLUG ITS OWN NUMBERS into the problem's relationships and write pure-numeric CHECK lines
# the exact oracle can re-derive. "Transcribe the problem, no letters/units" is load-bearing:
# the catch only works when a CHECK is the PROBLEM's constraint with the answer plugged in.
VERIFY_PROMPT = (
    "Answer the question, then verify your own answer by substitution.\n"
    "Put the answer on its own line:\n"
    "    ANSWER: <a number or one short sentence>\n"
    "Then substitute YOUR numbers into the relationships the PROBLEM states and write each "
    "as a line containing ONLY digits and + - * / ( ) — no letters, no variables, no units:\n"
    "    CHECK: <numbers only> = <number>\n"
    "    CHECK: <numbers only> = <number>\n"
    "Each CHECK must restate a fact the PROBLEM gives, with your numbers plugged in. If the "
    "answer is a single calculation, one CHECK containing it is enough.\n"
    "Question: {q}")

# Setup re-derivation prompt (see equations.py): when the numeric plug-back had nothing
# checkable, ask the model to TRANSCRIBE the problem's relationships as equations over
# named unknowns — transcription is an easier skill than solving — and we solve the
# linear system ourselves, exactly, and compare against the model's answer. "From the
# problem itself, not your solution" is load-bearing: an equation invented from the
# model's own (possibly wrong) reasoning just re-derives the same mistake.
SETUP_PROMPT = (
    "Set the problem up as equations, then solve it.\n"
    "First write each relationship the PROBLEM states as one equation line, using a short "
    "one-word name for each unknown quantity and * for multiplication:\n"
    "    EQN: <equation>\n"
    "Every EQN must restate a fact from the problem itself, not a step of your solution.\n"
    "Then give the answer, naming the unknown it is the value of:\n"
    "    ANSWER: <name> = <number>\n"
    "Question: {q}")

# One transcription call instead of two: EQN lines + ANSWER + CHECK lines in a single
# response. The router reads the strongest signal first — a solvable EQN system is a
# re-derivation (equations.py); only when nothing derives do the CHECK lines get graded
# (verify.py); only when nothing checks does the vote run. Same precedence as the
# two-call flow, one prefill+decode cheaper every time the setup tier used to fall
# through. Live-measured: the 7B produces all three sections cleanly in ~30-60 tokens.
FUSED_PROMPT = (
    "Set the problem up as equations, solve it, then verify by substitution.\n"
    "You may reason briefly first, each thought on its own line starting with THINK: .\n"
    "Then write each relationship the PROBLEM states as one equation line, using a short "
    "one-word name for each unknown quantity and * for multiplication:\n"
    "    EQN: <equation>\n"
    "Every EQN must restate a fact from the problem itself, not a step of your solution.\n"
    "Then give the answer, naming the unknown it is the value of:\n"
    "    ANSWER: <name> = <number>\n"
    "Then substitute YOUR numbers into the problem's relationships and write each as a "
    "line containing ONLY digits and + - * / ( ):\n"
    "    CHECK: <numbers only> = <number>\n"
    "Question: {q}")

# GBNF grammars (llamacpp transport only). They lock the transcription tiers to exactly
# the line shapes the oracles parse — killing the ramble class and the units-inside-CHECK
# class at the sampler. Written to fight the model as little as possible: capitals
# allowed (equations.py parses case-insensitively), optional blank lines between
# sections, and — load-bearing — a BOUNDED think block up front. The first grammar cut
# had no think room and transcription quality collapsed on exactly the trap classes the
# derive tier exists for (chicken cracked -> wrong-served, Sally caught -> wrong-served,
# feet-and-inches correct -> mangled): the prose these prompts used to permit WAS the
# model's chain of thought. THINK: lines give it back — capped at 6 x 220 chars, and
# invisible to the parsers. The ANSWER number keeps an optional % (a percent answer
# forced unitless came back as 0.99, served in place of 99%). ⚠ GBNF character classes
# have NO `\-` escape — a dash goes LAST in the class, unescaped. llama-server SILENTLY
# ignores an unparseable grammar (it logs and generates unconstrained), so these
# strings are pinned by tests: a typo here would not crash anything, it would quietly
# disarm the constraint.
GRAMMAR_SETUP = (
    'root ::= think{0,6} eqn{1,6} answer\n'
    'think ::= "THINK: " [^\\n]{3,220} "\\n" "\\n"?\n'
    'eqn ::= "EQN: " side " = " side "\\n" "\\n"?\n'
    'answer ::= "ANSWER: " name " = " num "\\n"?\n'
    'side ::= [A-Za-z0-9+*/(). _-]+\n'
    'name ::= [A-Za-z_]+\n'
    'num ::= "-"? [0-9]+ ("." [0-9]+)? ("/" [0-9]+)? "%"?\n')
GRAMMAR_FUSED = (
    'root ::= think{0,6} eqn{1,6} answer check{1,4}\n'
    'think ::= "THINK: " [^\\n]{3,220} "\\n" "\\n"?\n'
    'eqn ::= "EQN: " side " = " side "\\n" "\\n"?\n'
    'answer ::= "ANSWER: " name " = " num "\\n" "\\n"?\n'
    'check ::= "CHECK: " dexpr " = " num "\\n"? "\\n"?\n'
    'side ::= [A-Za-z0-9+*/(). _-]+\n'
    'name ::= [A-Za-z_]+\n'
    'num ::= "-"? [0-9]+ ("." [0-9]+)? ("/" [0-9]+)? "%"?\n'
    'dexpr ::= [0-9+*/(). -]+\n')


def _fused():
    """Is the one-call fused transcription tier on? EXPERIMENTAL, default OFF
    (HYBRID_FUSE=1 opts in). Measured live before demotion: asking one call to
    transcribe AND self-check interferes with the transcription itself — mixed-unit
    conversions the setup tier transcribes correctly (5 ft 4 in -> 162.56 cm) came
    back mangled (5.33), and a percent answer lost its unit — and the plug-back
    tier then graded the mangled answer's true-but-disconnected arithmetic as
    'checked'. One call is only cheaper if its answers stay worth serving."""
    return os.environ.get("HYBRID_FUSE", "") == "1"


def _fmt(x):
    return str(int(x)) if float(x).is_integer() else f"{x:.4g}"


# Known-hard categories: a small model is unreliable here, so escalate by rule.
_HARD = re.compile(
    r"\b(prove|proof|derive|theorem|algorithm|complexity|optimi[sz]e|debug|implement|"
    r"function|code|program|script|regex|puzzle|riddle|jug|liters?|measure exactly|"
    r"step[- ]by[- ]step|why (does|is|are|do)|explain why|trade[- ]?offs?|analy[sz]e|"
    r"to the power|raised to|factorial|root of|\d+\s*\^\s*\d+)\b", re.I)

# Open-ended / creative: no single right answer, so self-consistency mis-fires. Keep local.
_OPEN = re.compile(
    r"\b(rewrite|reword|rephrase|paraphrase|summari[sz]e|shorten|polish|"
    r"more formal|less formal|formally|casual|draft (a|an|me)|compose|"
    r"brainstorm|suggest|give me ideas)\b", re.I)

_NUMBER_WORD = re.compile(
    r"\d|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"twenty|thirty|forty|fifty|hundred|thousand|half|third|quarter|dozen|"
    r"twice|double|triple)\b", re.I)


def _quantitative(q):
    """Should the derive/verify tiers fire? A digit is enough; number-WORDS need two
    mentions ('a chicken and a half ... an egg and a half') so a lone 'in one sentence'
    doesn't send a factual query through two extra local calls."""
    return any(ch.isdigit() for ch in q) or len(_NUMBER_WORD.findall(q)) >= 2


def _llamacpp_body(prompt, num_predict, temperature, grammar):
    """Request body for llama-server's native /completion. Every tier prompt ends with
    '\\nQuestion: <q>', so the split below puts the FIXED instructions in the template's
    {sys} slot and only the query in {user} — that makes the instructions a stable
    prefix, which is what cache_prompt amortizes across calls."""
    instr, _, q = prompt.rpartition("\nQuestion: ")
    wrapped = (PROMPT_WRAP.format(sys=instr, user="Question: " + q) if instr
               else PROMPT_WRAP.format(sys="", user=prompt))
    body = {"prompt": wrapped, "n_predict": num_predict, "temperature": temperature,
            "cache_prompt": True, "stop": ["<|im_end|>"]}
    if grammar and os.environ.get("HYBRID_GRAMMAR", "1") != "0":
        body["grammar"] = grammar
    return body


# ── Slot pinning ─────────────────────────────────────────────────────────────
# cache_prompt reuses a prefill only when the matching KV is in the SLOT the
# request lands on. llama-server distributes unpinned requests across slots, so
# a family of requests sharing a long instruction prefix (a classifier's system
# prompt) keeps re-prefilling that prefix in whichever slot each request happens
# to hit — worst in the k-sample vote, where k IDENTICAL prompts land on k slots
# and prefill k times. Pinning the family to one slot (id_slot) makes the server
# queue them there instead: the first request prefills, the rest reuse. Decode is
# what serializes, so this is only worth it where decode is tiny (grammar-locked
# labels: 1-4 tokens); long-decode votes are better off batching across slots.
_SLOTS_LOCK = threading.Lock()
_SLOTS_CACHE = {}  # server root URL -> slot count (0 = endpoint absent/disabled)


def _llamacpp_slots(url):
    """Slot count of the llama-server behind `url` (a .../completion endpoint),
    probed once via GET /slots and cached. 0 means unknown — callers skip pinning.
    Never raises: a server without the endpoint just gets unpinned behavior."""
    root = url.rsplit("/completion", 1)[0]
    with _SLOTS_LOCK:
        if root in _SLOTS_CACHE:
            return _SLOTS_CACHE[root]
    n = 0
    try:
        r = json.loads(urllib.request.urlopen(root + "/slots", timeout=5).read())
        if isinstance(r, list):
            n = len(r)
    except Exception:
        n = 0
    with _SLOTS_LOCK:
        _SLOTS_CACHE[root] = n
    return n


def _pin_slot(body, url, family):
    """Add id_slot to a /completion body: a stable hash of the family name, modulo
    the server's slot count. Same family -> same slot, every process, every run."""
    if not family or os.environ.get("HYBRID_SLOT_PIN", "1") == "0":
        return
    n = _llamacpp_slots(url)
    if n > 0:
        h = int(hashlib.sha1(family.encode("utf-8", "replace")).hexdigest()[:8], 16)
        body["id_slot"] = h % n


# ── Token accounting ─────────────────────────────────────────────────────────
# Per-request tally of real token counts, read from the backends' own responses
# (Ollama's eval counts, llama-server's tokens_evaluated/predicted, the frontier's
# usage object) — the decision log gets measured spend, not a chars/4 estimate.
# A ContextVar carries the tally down the call tree: server threads each handle
# one request, so concurrent requests keep separate tallies, and the vote tiers
# copy their context into worker threads so k samples add to the SAME request.
_TOKENS = contextvars.ContextVar("hybrid_tokens", default=None)
_TOKENS_LOCK = threading.Lock()


def _tokens_reset():
    """Start a fresh tally for the current request and return the (mutable) dict.
    Outside a request — warmup, bare transport calls in tests — no tally exists
    and _tokens_add is a no-op."""
    t = {"local_in": 0, "local_out": 0, "local_calls": 0,
         "frontier_in": 0, "frontier_out": 0, "frontier_calls": 0}
    _TOKENS.set(t)
    return t


def _tokens_add(tier, n_in, n_out):
    """Add one backend call's token counts to the current request's tally.
    Backends that omit a count contribute 0 for it — the call is still counted."""
    t = _TOKENS.get()
    if t is None:
        return
    try:  # accounting must never break routing — garbage counts read as 0
        n_in, n_out = int(n_in or 0), int(n_out or 0)
    except (TypeError, ValueError):
        n_in, n_out = 0, 0
    with _TOKENS_LOCK:
        t[tier + "_in"] += n_in
        t[tier + "_out"] += n_out
        t[tier + "_calls"] += 1


def ollama(prompt, num_predict=256, temperature=0.0, model=None, grammar=None,
           family=None):
    """One local-model call (default LOCAL_MODEL; the vote/creative tiers pass
    LOCAL_MODEL_FAST). Retries once (transports flake), then raises BackendError —
    an answer string is ALWAYS a real model answer, never an error in disguise.
    With HYBRID_LOCAL_BACKEND=llamacpp the same call goes to llama-server's native
    /completion instead (prefix cache + optional GBNF grammar; `model` is whatever
    the server loaded). `grammar` is ignored on the Ollama transport — the generate
    API has no grammar parameter. `family` (llamacpp only) names the request's
    prompt family for slot pinning — see _pin_slot; None (default) pins nothing."""
    if LOCAL_BACKEND == "llamacpp":
        r, dt = _llamacpp_call(prompt, num_predict, temperature, grammar, model, family)
        return r.get("content", "").strip(), dt
    url = OLLAMA_URL
    body = json.dumps({
        "model": model or LOCAL_MODEL, "prompt": prompt, "stream": False, "keep_alive": "5m",
        "options": {"num_predict": num_predict, "temperature": temperature},
    }).encode()
    t0 = time.time()
    last = None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"content-type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            _tokens_add("local", r.get("prompt_eval_count"), r.get("eval_count"))
            return r.get("response", "").strip(), time.time() - t0
        except Exception as e:
            last = e
            if attempt == 1:
                time.sleep(1.0)
    raise BackendError("local", f"{model or LOCAL_MODEL} at {url}: {last}")


def _llamacpp_call(prompt, num_predict, temperature, grammar, model, family, extra=None):
    """One llama-server /completion round-trip, returning the PARSED RESPONSE dict
    (not just the text) plus elapsed seconds — so callers that need more than
    `content` (the label-posterior read wants completion_probabilities) share the
    same URL routing, slot pinning, and retry/BackendError contract as ollama().
    `extra` merges additional request fields (e.g. n_probs)."""
    # the fast-tier calls (they pass model=LOCAL_MODEL_FAST) go to the fast server
    # when one is configured; transcription calls never do — same pinned policy as
    # the Ollama transport, enforced by URL instead of model name.
    fast = (LLAMACPP_URL_FAST and model is not None
            and model == LOCAL_MODEL_FAST and model != LOCAL_MODEL)
    url = LLAMACPP_URL_FAST if fast else LLAMACPP_URL
    body_dict = _llamacpp_body(prompt, num_predict, temperature, grammar)
    if extra:
        body_dict.update(extra)
    _pin_slot(body_dict, url, family)
    body = json.dumps(body_dict).encode()
    t0 = time.time()
    last = None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(url, data=body,
                                         headers={"content-type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            _tokens_add("local", r.get("tokens_evaluated"), r.get("tokens_predicted"))
            return r, time.time() - t0
        except Exception as e:
            last = e
            if attempt == 1:
                time.sleep(1.0)
    raise BackendError("local", f"{model or LOCAL_MODEL} at {url}: {last}")


def escalate(query, messages=None):
    """Send to any OpenAI-compatible frontier endpoint (set FRONTIER_URL / _API_KEY /
    _MODEL). Sends `messages` (a full OpenAI-style conversation) when given, else a
    single user turn. Raises BackendError on failure — same contract as ollama()."""
    body = json.dumps({
        "model": FRONTIER_MODEL,
        "messages": messages or [{"role": "user", "content": query}],
        "max_tokens": 512,
    }).encode()
    headers = {"content-type": "application/json"}
    if FRONTIER_KEY:
        headers["authorization"] = "Bearer " + FRONTIER_KEY
    req = urllib.request.Request(FRONTIER_URL, data=body, headers=headers)
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        ans = r["choices"][0]["message"]["content"].strip()
        u = r.get("usage") or {}
        _tokens_add("frontier", u.get("prompt_tokens"), u.get("completion_tokens"))
        return ans, time.time() - t0
    except Exception as e:
        raise BackendError("frontier", f"{FRONTIER_MODEL} at {FRONTIER_URL}: {e}")


def _escalate(query, messages):
    """Escalate, passing the conversation only when there is one. The 1-arg call
    shape is a contract: bench_router/measure_routing/tests replace escalate() with
    single-argument stubs, and every no-conversation path must keep working there."""
    return escalate(query, messages=messages) if messages else escalate(query)


def _key(ans):
    """Normalize an answer to a comparable key for self-consistency voting."""
    s = ans.lower().strip()
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", s)
    if nums:
        return re.sub(r"[^\d.]", "", nums[-1]).rstrip(".") or s[:30]
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split()[:6])


def local_consistency(query, k=3):
    """Sample the fast local model k times CONCURRENTLY; 'confident' iff all answers
    agree. Concurrency is a pure wall-time win: CPU decode is memory-bandwidth-bound,
    so a server that batches (Ollama with OLLAMA_NUM_PARALLEL >= k) streams the weights
    once per token-step for all k samples — the vote costs roughly ONE sample's wall
    time instead of k. Against a serial server the requests just queue: same behavior,
    same total time as the old loop. A BackendError in any sample re-raises here, so
    the failure policy sees exactly what it saw before.

    HYBRID_VOTE_FAST=1 (opt-in) samples 2 instead of 3 and requires 2/2 — one fewer
    decode, and STRICTLY more escalation-prone on disagreement (there is no third
    sample to complete a unanimity). The trade is confidence-evidence for wall time:
    2/2 at temperature 0.6 is weaker evidence than 3/3. Measured before shipping;
    see the README table. The vote budget is 56 tokens — CONCISE asks for a number,
    a word, or a sentence or two, and everything past that is decode spent on a
    ramble no key extraction reads."""
    if os.environ.get("HYBRID_VOTE_FAST", "") == "1":
        k = 2
    return _vote(CONCISE.format(q=query), k, LOCAL_MODEL_FAST, 56)


def _vote(prompt, k, model, num_predict):
    """Sample `model` on `prompt` k times concurrently at temperature 0.6; return
    (unanimous, best_sample, "n/k", secs). The shared engine behind both the
    self-contained-question vote (local_consistency, hybrid's own CONCISE prompt)
    and the instruction-following vote (route_messages, the CALLER's own prompt)."""
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=k) as ex:
        # copy_context per worker: the samples' token counts land on the tally of
        # the REQUEST that voted (ContextVars don't cross threads by themselves).
        futures = [ex.submit(contextvars.copy_context().run,
                             ollama, prompt, num_predict, 0.6, model)
                   for _ in range(k)]
        samples = [f.result()[0] for f in futures]
    t = time.time() - t0
    keys = [_key(s) for s in samples]
    top, n = Counter(keys).most_common(1)[0]
    best = next(s for s, kk in zip(samples, keys) if kk == top)
    return (n >= k), best, f"{n}/{k}", t  # unanimous: escalate unless the model fully agrees with itself


# ── Load shedding ──────────────────────────────────────────────────────────
# One shared gauge of model-tier requests currently executing. It is the load
# signal the shed gate reads, and it is shared across threads on purpose: a
# server (server.py) handles requests on threads, and on a bandwidth-bound box
# every one of them contends for the same memory bus, so "how many model calls
# are running right now" is the real capacity number — not CPU %, not a queue.
_MODEL_LOCK = threading.Lock()
_MODEL_INFLIGHT = 0


def _int_env(name):
    """Env var as a non-negative int; malformed or unset -> 0 (the 'off' value)."""
    try:
        return max(0, int(os.environ.get(name, "0") or 0))
    except ValueError:
        return 0


def model_inflight():
    """Model-tier requests currently executing (for /health, dashboards, tests)."""
    with _MODEL_LOCK:
        return _MODEL_INFLIGHT


def _enter_model_or_shed(elapsed_s):
    """Decide — atomically — whether to run model-tier work locally or shed it to
    the frontier. Returns a shed-reason string (NO slot taken), or None meaning a
    slot was taken and the caller MUST call _leave_model() when the model work ends.
    Check-and-take under one lock so two arrivals can't both slip past a cap of 1."""
    global _MODEL_INFLIGHT
    cap = _int_env("HYBRID_MODEL_MAX_INFLIGHT")
    budget_ms = _int_env("HYBRID_LATENCY_BUDGET_MS")
    with _MODEL_LOCK:
        inflight = _MODEL_INFLIGHT
        if cap and inflight >= cap:
            return f"load shed: {inflight} model call(s) in flight, cap {cap} -> frontier"
        if budget_ms:
            tier_ms = _int_env("HYBRID_MODEL_TIER_MS") or 8000
            # a new call waits behind the ones already running (bandwidth-bound, so
            # roughly additive), then costs one tier itself.
            projected = elapsed_s * 1000 + tier_ms * (inflight + 1)
            if projected > budget_ms:
                return (f"latency budget: ~{int(projected)}ms projected "
                        f"({inflight} ahead) > {budget_ms}ms -> frontier")
        _MODEL_INFLIGHT += 1  # slot taken; caller owns the matching _leave_model()
        return None


def _leave_model():
    global _MODEL_INFLIGHT
    with _MODEL_LOCK:
        _MODEL_INFLIGHT = max(0, _MODEL_INFLIGHT - 1)


def _apply_failure_policy(e, query, messages):
    """Turn a BackendError into a degraded route or an ERROR result per the
    HYBRID_ON_*_FAIL env policy. Shared by route() and route_messages() so both
    surfaces degrade identically instead of raising."""
    if e.tier == "local" and os.environ.get("HYBRID_ON_LOCAL_FAIL", "escalate") == "escalate":
        try:
            ans, dt = _escalate(query, messages)
            return {"route": "ESCALATE", "why": "local backend down -> frontier",
                    "backend": FRONTIER_MODEL, "answer": ans,
                    "router_s": 0.0, "answer_s": round(dt, 2)}
        except BackendError as e2:
            e = e2                        # frontier is down too -> ERROR below
    elif e.tier == "frontier" and os.environ.get("HYBRID_ON_FRONTIER_FAIL", "error") == "local":
        try:
            ans, dt = ollama(CONCISE.format(q=query), num_predict=200,
                             model=LOCAL_MODEL_FAST)
            return {"route": "LOCAL", "why": "DEGRADED: frontier down, unverified local",
                    "backend": LOCAL_MODEL_FAST, "answer": ans,
                    "router_s": 0.0, "answer_s": round(dt, 2)}
        except BackendError as e2:
            e = e2
    return {"route": "ERROR", "why": f"{e.tier} backend unavailable",
            "backend": e.tier, "answer": f"[{e.tier} backend unavailable: {e}]",
            "router_s": 0.0, "answer_s": 0.0, "error": True}


# ─────────────────────────────────────────────────────────────────────────
# Startup warmup (HYBRID_WARMUP=1, see server.py). On a CPU, prefill is the
# compute-bound wall (~seconds per uncached instruction preamble), so a freshly
# started / redeployed container serves its opening traffic slowly until the KV
# cache fills. Warmup carries each FIXED local tier's preamble through one
# throwaway forward pass up front so the prefill (and, for pinned families, the
# slot) is hot before real traffic arrives. In-memory only — re-run per restart.
# ─────────────────────────────────────────────────────────────────────────

# The preamble — not the query — is what cache_prompt/prefill amortizes, so the
# trivial query is irrelevant and num_predict=1 keeps decode negligible. The
# caller-supplied LABELLED-classifier preamble (route_messages) depends on the
# runtime system+labels, so it can't be primed blind at boot and is not warmed
# here — only the router's own fixed-template tiers are.
_WARMUP_TIERS = (
    ("concise", lambda: CONCISE.format(q="hello"), LOCAL_MODEL_FAST),
    ("derive",  lambda: FUSED_PROMPT.format(q="What is 2 plus 2?"), LOCAL_MODEL),
)


def warmup():
    """Prime the local backend before serving. Sends one throwaway forward pass per
    fixed tier DIRECTLY to the local model — no routing, no escalation, no frontier
    call — so each tier's preamble lands in the prefill cache and the model loads.
    Never raises: a cold or unreachable backend records an error marker and startup
    proceeds unblocked. Returns {tier: seconds} (or {tier: "err: <Type>"})."""
    out = {}
    for name, make_prompt, model in _WARMUP_TIERS:
        t0 = time.time()
        try:
            ollama(make_prompt(), num_predict=1, temperature=0.0, model=model)
            out[name] = round(time.time() - t0, 2)
        except Exception as e:  # BackendError / transport flake — warmup is best-effort
            out[name] = f"err: {type(e).__name__}"
    return out


def route(query, messages=None):
    """Route one query. `messages` (optional) is the full OpenAI-style conversation:
    routing decisions and the LOCAL tiers always work on `query` — the last user
    message — but an escalated call carries the whole conversation to the frontier.
    Never raises for a dead backend — the failure policy (env, read per-call so a
    live service can be re-tuned) turns a BackendError into either a degraded route
    or an explicit ERROR result the caller can surface.

    The result carries `tokens` — the request's measured backend spend (see
    _tokens_reset). SOLVED routes show all zeros: that IS the datapoint."""
    t = _tokens_reset()
    try:
        r = _route(query, messages)
    except BackendError as e:
        r = _apply_failure_policy(e, query, messages)
    r["tokens"] = dict(t)
    return r


def _route(query, messages=None):
    t0 = time.time()
    # 0. exact oracle — closed-form arithmetic / conversions / %-change, free + correct.
    exact = solve(query)
    if exact is not None:
        return {"route": "SOLVED", "why": "deterministic arithmetic",
                "backend": "python (exact)", "answer": exact,
                "router_s": 0.0, "answer_s": 0.0}
    # 0.5. template transcriber — a word problem whose shape we recognize outright is
    # transcribed deterministically and solved in closed form: no model, no tokens.
    # Runs BEFORE the hard-category rule on purpose: a clean template parse is exact,
    # so a stray rule keyword in the query ("...13.9 liters...") must not out-rank it.
    tmpl = templates.solve(query)
    if tmpl is not None:
        val, shape = tmpl
        return {"route": "SOLVED", "why": f"template: {shape}",
                "backend": "python (exact)", "answer": val,
                "router_s": 0.0, "answer_s": 0.0}
    # 1. known-hard categories -> escalate by rule. No model call, so it is never
    #    gated or slotted — a hard query already goes to the frontier.
    if _HARD.search(query) or len(query) > 220:
        ans, dt = _escalate(query, messages)
        return {"route": "ESCALATE", "why": "rule: hard category", "backend": FRONTIER_MODEL,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    # --- LOAD-SHED GATE --------------------------------------------------------
    # Everything past here does model-tier work. If the box is over its concurrency
    # cap or the request's latency budget, escalate NOW instead of queueing a slow
    # local call (both signals off by default -> no gating). On "proceed", a model
    # slot is held for the whole model portion via the try/finally below.
    shed = _enter_model_or_shed(time.time() - t0)
    if shed:
        ans, dt = _escalate(query, messages)
        return {"route": "ESCALATE", "why": shed, "backend": FRONTIER_MODEL,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    try:
        return _route_model(query, messages)
    finally:
        _leave_model()


def _route_model(query, messages=None):
    """The model-tier portion of the route, run while a model slot is held. Every
    path here makes at least one local model call: _OPEN creative, the quantitative
    oracle tiers (derive/plug-back, or _route_two_call), then the vote."""
    # 2. open-ended / creative -> keep local (no single right answer, fast model fine).
    if _OPEN.search(query):
        ans, dt = ollama(CONCISE.format(q=query), num_predict=200, model=LOCAL_MODEL_FAST)
        return {"route": "LOCAL", "why": "open-ended (local ok)", "backend": LOCAL_MODEL_FAST,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    # 3. quantitative queries get the exact oracle, strongest signal first:
    #    derive (independent re-derivation) > plug-back (consistency) > vote (agreement).
    if _quantitative(query):
        if _fused():
            # One transcription call carries every signal. Precedence is IDENTICAL to
            # the two-call flow below: a derive verdict (strongest) wins outright — a
            # sloppy CHECK next to a confirmed derivation must not escalate an answer
            # the exact solver just re-derived (live case: the self-referential brick).
            raw, dt = ollama(FUSED_PROMPT.format(q=query), num_predict=380,
                             temperature=0.0, grammar=GRAMMAR_FUSED)
            st, info = equations.verdict(raw)
            if st == "mismatch":
                esc, et = _escalate(query, messages)
                return {"route": "ESCALATE",
                        "why": (f"setup derives {info['var']}="
                                f"{equations.fmt(info['derived'])}≠{_fmt(info['claimed'])}"),
                        "backend": FRONTIER_MODEL, "answer": esc,
                        "router_s": round(dt, 2), "answer_s": round(et, 2)}
            if st == "derived":
                return {"route": "LOCAL", "why": f"setup re-derived ({info['eqns']} eqn, fused)",
                        "backend": LOCAL_MODEL, "answer": equations.fmt(info["derived"]),
                        "router_s": 0.0, "answer_s": round(dt, 2)}
            status, claims = verify.verdict(raw)
            if status == "wrong":
                bad = next(c for c in claims if not c["ok"])
                kind = "constraint violated" if re.search(r"[a-z]", bad["expr"], re.I) else "local math wrong"
                esc, et = _escalate(query, messages)
                return {"route": "ESCALATE",
                        "why": f"{kind} ({bad['expr']}={_fmt(bad['claimed'])}≠{_fmt(bad['actual'])})",
                        "backend": FRONTIER_MODEL, "answer": esc,
                        "router_s": round(dt, 2), "answer_s": round(et, 2)}
            if status == "checked":
                answer = verify.answer_text(raw) or _fmt(claims[-1]["actual"])
                kind = "constraints hold" if verify.has_constraint(claims) else "arithmetic checks"
                return {"route": "LOCAL", "why": f"{kind} ({len(claims)} eqn, fused)",
                        "backend": LOCAL_MODEL, "answer": answer,
                        "router_s": 0.0, "answer_s": round(dt, 2)}
            # neither derivable nor checkable -> fall through to the vote, having spent
            # ONE call where the two-call flow would by now have spent two.
        else:
            return _route_two_call(query, messages)
    # 4. everything else -> self-consistency decides.
    confident, best, agree, ct = local_consistency(query)
    if confident:
        return {"route": "LOCAL", "why": f"self-consistent {agree}",
                "backend": LOCAL_MODEL_FAST,
                "answer": best, "router_s": round(ct, 2), "answer_s": 0.0}
    ans, dt = _escalate(query, messages)
    return {"route": "ESCALATE", "why": "uncertain (self-inconsistent)", "backend": FRONTIER_MODEL,
            "answer": ans, "router_s": round(ct, 2), "answer_s": round(dt, 2)}


def _route_two_call(query, messages):
    """The pre-fusion quantitative path: setup call, then (on fall-through) a separate
    plug-back call, then the vote. Byte-identical behavior to v1.6.x — the Ollama
    transport's measured default until fusion is re-benched there."""
    # setup re-derivation (equations.py): the model transcribes the problem's
    # relationships as equations, we solve the linear system OURSELVES (exact
    # Fractions, free) and compare to its answer. A mismatch means the model
    # mis-solved its own transcription -> HARD escalate. This runs FIRST because
    # plug-back can be fooled by a tautology (a true-but-disconnected check like
    # `(1.5/1.5)*1 = 1` reads as "checked"); a derivation can't — it produces its
    # own value for the answer instead of grading the model's checks.
    raw, dt = ollama(SETUP_PROMPT.format(q=query), num_predict=220, temperature=0.0,
                     grammar=GRAMMAR_SETUP)
    st, info = equations.verdict(raw)
    if st == "mismatch":
        esc, et = _escalate(query, messages)
        return {"route": "ESCALATE",
                "why": (f"setup derives {info['var']}="
                        f"{equations.fmt(info['derived'])}≠{_fmt(info['claimed'])}"),
                "backend": FRONTIER_MODEL, "answer": esc,
                "router_s": round(dt, 2), "answer_s": round(et, 2)}
    if st == "derived":
        # serve the value we RE-DERIVED (== the model's answer, but exact and clean —
        # the model may have phrased its ANSWER line in LaTeX or prose)
        return {"route": "LOCAL", "why": f"setup re-derived ({info['eqns']} eqn)",
                "backend": LOCAL_MODEL, "answer": equations.fmt(info["derived"]),
                "router_s": 0.0, "answer_s": round(dt, 2)}

    # nothing derivable (no clean linear system) -> plug-back verify: the model
    # answers and states its calculation; we re-check that arithmetic exactly. A
    # false equation is a HARD escalate signal (the model's own stated math is
    # wrong), not a vote. Correct arithmetic we trust. Nothing checkable -> vote.
    raw2, dt2 = ollama(VERIFY_PROMPT.format(q=query), num_predict=220, temperature=0.0)
    status, claims = verify.verdict(raw2)
    if status == "wrong":
        bad = next(c for c in claims if not c["ok"])
        # a variable in the failed expr means it was a CHECK (the answer is inconsistent
        # with a problem constraint); a bare expr means the model's own arithmetic is off.
        kind = "constraint violated" if re.search(r"[a-z]", bad["expr"], re.I) else "local math wrong"
        esc, et = _escalate(query, messages)
        return {"route": "ESCALATE",
                "why": f"{kind} ({bad['expr']}={_fmt(bad['claimed'])}≠{_fmt(bad['actual'])})",
                "backend": FRONTIER_MODEL, "answer": esc,
                "router_s": round(dt + dt2, 2), "answer_s": round(et, 2)}
    if status == "checked":
        # the model may have put the value only in the working lines; show the answer
        answer = verify.answer_text(raw2) or _fmt(claims[-1]["actual"])
        kind = "constraints hold" if verify.has_constraint(claims) else "arithmetic checks"
        return {"route": "LOCAL", "why": f"{kind} ({len(claims)} eqn)",
                "backend": LOCAL_MODEL, "answer": answer,
                "router_s": round(dt, 2), "answer_s": round(dt2, 2)}
    # 4. everything else -> self-consistency decides.
    confident, best, agree, ct = local_consistency(query)
    if confident:
        return {"route": "LOCAL", "why": f"self-consistent {agree}",
                "backend": LOCAL_MODEL_FAST,
                "answer": best, "router_s": round(ct, 2), "answer_s": 0.0}
    ans, dt = _escalate(query, messages)
    return {"route": "ESCALATE", "why": "uncertain (self-inconsistent)", "backend": FRONTIER_MODEL,
            "answer": ans, "router_s": round(ct, 2), "answer_s": round(dt, 2)}


# ── Anthropic Messages surface ───────────────────────────────────────────────
# A second front door that speaks the Anthropic /v1/messages shape, so the
# Anthropic-shaped callers a fleet is full of (inline @anthropic-ai/sdk
# `messages.create`, a base-URL-configurable client) can point at hybrid and get
# local-first routing with frontier escalation. The KEY difference from the
# OpenAI surface: those callers are usually *instruction-following* — the task
# lives in the `system` prompt (classify / extract / judge), not in the user
# turn. hybrid's arithmetic tiers impose their OWN prompt and so would ignore
# that instruction; running the deterministic solver on the user text would even
# answer the wrong question. So route_messages() branches on whether a system
# instruction is present (see below). Text-only, non-streaming — the cheap
# `messages.create` calls this targets don't stream, and the full agent/CLI
# path (tools, streaming) needs the real model anyway.

def _block_text(content):
    """Anthropic message content -> plain text. A turn's content is either a
    string or a list of blocks ([{type:'text',text:...}, ...]); non-text blocks
    (images, tool_use) are skipped — this surface is text-only."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content
                         if isinstance(b, dict) and b.get("type", "text") == "text")
    return ""


def _anthropic_user_text(messages):
    """The last user turn's text — the 'query' hybrid's self-contained tiers see."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return _block_text(m.get("content", "")).strip()
    return ""


def _anthropic_to_openai(system, messages):
    """Anthropic (system + messages) -> an OpenAI-style message list for the
    escalation payload, so an escalated call carries the whole conversation AND
    the system instruction to the frontier."""
    out = []
    if system and str(system).strip():
        out.append({"role": "system", "content": str(system)})
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
            out.append({"role": m["role"], "content": _block_text(m.get("content", ""))})
    return out


def _render_prompt(system, messages):
    """Render (system + conversation) into ONE prompt string for the local model —
    used only by the instruction-following vote, so the local model sees the
    caller's ACTUAL task (its system instruction), not hybrid's CONCISE rewrite.
    ollama()'s llamacpp transport splits on '\\nQuestion: ' into the template's
    system/user slots; the ollama transport sends it verbatim."""
    parts = []
    if system and str(system).strip():
        parts.append(str(system).strip())
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant"):
            continue
        who = "Assistant" if m["role"] == "assistant" else "User"
        parts.append(f"{who}: {_block_text(m.get('content', '')).strip()}")
    return "\n".join(parts) + "\nQuestion: (respond to the last User turn following the instructions above)"


def _clean_labels(labels):
    """A request's declared label set (metadata.hybrid_labels) -> a clean list of
    distinct non-empty single-line strings, capped. Anything malformed -> [] (the
    request just isn't treated as a labelled classification)."""
    if not isinstance(labels, (list, tuple)):
        return []
    out = []
    for l in labels:
        if isinstance(l, str):
            l = l.strip()
            if l and "\n" not in l and len(l) <= 64 and l not in out:
                out.append(l)
    return out[:32]


def _labels_grammar(labels):
    """GBNF forcing the local model to emit EXACTLY one of the labels (llamacpp
    transport only — the Ollama generate API ignores it, and we extract the label
    instead). Returns None if a label isn't safe for a GBNF string literal, so a
    weird label degrades to unconstrained-sample-then-extract rather than a broken
    grammar (which llama-server would silently ignore anyway)."""
    if any('"' in l or "\\" in l for l in labels):
        return None
    return "root ::= " + " | ".join(f'"{l}"' for l in labels) + "\n"


def _extract_label(text, labels):
    """The first declared label appearing as a whole word in `text` (case-
    insensitive), or None — so a rambly 'The category is build.' still yields
    'build', and 'I am not sure' yields None (-> escalate)."""
    low = text.lower()
    best = None
    for l in labels:
        m = re.search(r"\b" + re.escape(l.lower()) + r"\b", low)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), l)
    return best[1] if best else None


def _label_posterior(entries, labels):
    """Map a first-token top_logprobs list onto the label set and return
    {label: probability mass}. An entry counts toward a label when its token
    (stripped, case-folded) is a non-empty prefix of EXACTLY ONE label — that
    absorbs BPE splits ('autom' -> automate, 'analy' -> analyze) and the
    leading-space variants, while an ambiguous prefix ('a' matches several)
    is skipped outright rather than guessed."""
    mass = {}
    for e in entries or []:
        tok = str(e.get("token", "")).strip().lower()
        lp = e.get("logprob")
        if not tok or lp is None:
            continue
        hits = [l for l in labels if l.lower().startswith(tok)]
        if len(hits) == 1:
            mass[hits[0]] = mass.get(hits[0], 0.0) + math.exp(lp)
    return mass


def _route_messages_labeled_logits(labels, grammar, fam, prompt):
    """Read the classifier's answer off the model's OWN first-token distribution —
    one forward pass — instead of sampling it k times and voting. The label set is
    enumerable, so the posterior over it is directly readable (n_probs); serving
    the argmax behind a probability-and-margin gate is strictly more information
    than 'k samples at temperature 0.6 agreed', at a third of the forward passes
    and with no sampling noise. The why-string carries the two probabilities, so
    the decision log accumulates calibration data (local margin vs the frontier's
    later verdict) for tuning the gate per family.

    Returns a LOCAL result dict; or a soft-posterior marker {p1, p2, router_s}
    (the caller escalates AFTER releasing the model slot — a frontier round-trip
    must not hold local capacity); or None to fall back to the sampling vote —
    on ANY problem (old server without top_logprobs, malformed response, no
    readable mass): the vote is the safety net, never an exception."""
    try:
        r, dt = _llamacpp_call(prompt, 1, 0.0, grammar, LOCAL_MODEL_FAST, fam,
                               extra={"n_probs": 25})
        cp = r.get("completion_probabilities") or []
        entries = (cp[0] or {}).get("top_logprobs") if cp else None
        mass = _label_posterior(entries, labels)
        if not mass:
            return None
        ranked = sorted(mass.items(), key=lambda kv: -kv[1])
        best, p1 = ranked[0]
        p2 = ranked[1][1] if len(ranked) > 1 else 0.0
        min_p = float(os.environ.get("HYBRID_LABEL_MIN_P", "0.4"))
        margin = float(os.environ.get("HYBRID_LABEL_MARGIN", "2.0"))
        if p1 >= min_p and (p2 == 0.0 or p1 >= margin * p2):
            return {"route": "LOCAL",
                    "why": f"label posterior {p1:.2f} vs {p2:.2f} of {len(labels)}",
                    "backend": LOCAL_MODEL_FAST, "answer": best,
                    "router_s": round(dt, 2), "answer_s": 0.0}
        return {"p1": p1, "p2": p2, "router_s": round(dt, 2)}
    except BackendError:
        raise  # a dead backend is the failure policy's business, same as the vote
    except Exception:
        return None  # anything unexpected -> the sampling vote takes over


def _route_messages_labeled(system, messages, user_text, oai, labels):
    """Labelled classification — the 'constrain and verify' answer to 'vote and
    hope'. The local model is grammar-constrained to emit ONE of the caller's
    labels (llamacpp) and sampled k times; each sample is normalized to the label
    it contains (so generative preamble can't break unanimity). A unanimous, valid
    label is served on-box — guaranteed to be one the caller declared. Anything
    else — disagreement, or a sample with no valid label — escalates. So a served
    label is both self-consistent AND provably in-set; hybrid's verifier discipline,
    applied to classification."""
    shed = _enter_model_or_shed(0.0)
    if shed:
        ans, dt = _escalate(user_text, oai)
        return {"route": "ESCALATE", "why": shed, "backend": FRONTIER_MODEL,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    soft = None
    picks = ct = None
    try:
        grammar = _labels_grammar(labels)
        prompt = _render_prompt(system, messages)
        # One classifier = one family: same system prompt + same label set. Pinning
        # to that family's slot keeps the prompt's prefill hot across requests (and
        # across the vote's samples, when the vote runs). See _pin_slot.
        fam = "labels\x1f" + str(system or "") + "\x1f" + "\x1f".join(labels)
        # First choice: read the posterior (one forward pass). Falls back to the
        # k-sample vote when the transport/server can't report it.
        if LOCAL_BACKEND == "llamacpp" and os.environ.get("HYBRID_LABEL_LOGITS", "1") != "0":
            res = _route_messages_labeled_logits(labels, grammar, fam, prompt)
            if res is not None and res.get("route") == "LOCAL":
                return res
            soft = res  # soft marker (escalate below, slot released) or None (vote)
        if soft is None:
            k = 2 if os.environ.get("HYBRID_VOTE_FAST", "") == "1" else 3
            t0 = time.time()
            with ThreadPoolExecutor(max_workers=k) as ex:
                # copy_context per worker — see _vote
                futures = [ex.submit(contextvars.copy_context().run,
                                     ollama, prompt, 16, 0.6, LOCAL_MODEL_FAST,
                                     grammar, fam)
                           for _ in range(k)]
                picks = [_extract_label(f.result()[0], labels) for f in futures]
            ct = time.time() - t0
    finally:
        _leave_model()
    if soft is not None:
        ans, dt = _escalate(user_text, oai)
        return {"route": "ESCALATE",
                "why": f"label posterior soft ({soft['p1']:.2f} vs {soft['p2']:.2f})",
                "backend": FRONTIER_MODEL, "answer": ans,
                "router_s": soft["router_s"], "answer_s": round(dt, 2)}
    if picks[0] is not None and all(p == picks[0] for p in picks):
        return {"route": "LOCAL", "why": f"label self-consistent {k}/{k} of {len(labels)}",
                "backend": LOCAL_MODEL_FAST, "answer": picks[0],
                "router_s": round(ct, 2), "answer_s": 0.0}
    why = "label uncertain (self-inconsistent)" if None not in picks else "no valid label on-box"
    ans, dt = _escalate(user_text, oai)
    return {"route": "ESCALATE", "why": why, "backend": FRONTIER_MODEL,
            "answer": ans, "router_s": round(ct, 2), "answer_s": round(dt, 2)}


def route_messages(system, messages, max_tokens=512, labels=None):
    """Route an Anthropic-style request. Three modes:

      - LABELLED (metadata.hybrid_labels declares an allowed label set): a
        constrained-and-verified classification — grammar-lock the local model to
        the labels, vote on the extracted label, serve a unanimous in-set label or
        escalate (see _route_messages_labeled). The reliable path for classifiers.

    ...and, when no labels are declared:

    Two modes, chosen by whether a system instruction is present:

      - NO system instruction: the last user turn is a self-contained question, so
        run the full router (solve / template / verify / vote / escalate) — the
        arithmetic verifier and all — on it.
      - A system instruction IS present: this is instruction-following (classify /
        extract / judge). The deterministic solver does NOT apply (the task is not
        'answer the user text'), so we vote with the local model on the CALLER'S
        OWN prompt; unanimous -> serve local (free), otherwise escalate the whole
        conversation to the frontier. This respects the instruction and never
        serves a low-confidence local answer — the frontier catches the hard ones.

    Same dict shape and failure policy as route(), including `tokens`."""
    user_text = _anthropic_user_text(messages)
    oai = _anthropic_to_openai(system, messages)
    labels = _clean_labels(labels)
    if not labels and not (system and str(system).strip()):
        return route(user_text, oai)          # route() keeps its own tally
    t = _tokens_reset()
    try:
        if labels:
            r = _route_messages_labeled(system, messages, user_text, oai, labels)
        else:
            r = _route_messages_instructed(system, messages, user_text, oai, max_tokens)
    except BackendError as e:
        r = _apply_failure_policy(e, user_text, oai)
    r["tokens"] = dict(t)
    return r


def _route_messages_instructed(system, messages, user_text, oai, max_tokens):
    # everything here does model work -> obey the load-shed gate, like _route
    shed = _enter_model_or_shed(0.0)
    if shed:
        ans, dt = _escalate(user_text, oai)
        return {"route": "ESCALATE", "why": shed, "backend": FRONTIER_MODEL,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    try:
        k = 2 if os.environ.get("HYBRID_VOTE_FAST", "") == "1" else 3
        budget = max(48, min(int(max_tokens or 256), 256))  # vote answers stay short enough to compare
        confident, best, agree, ct = _vote(_render_prompt(system, messages), k, LOCAL_MODEL_FAST, budget)
    finally:
        _leave_model()
    if confident:
        return {"route": "LOCAL", "why": f"instruction self-consistent {agree}",
                "backend": LOCAL_MODEL_FAST, "answer": best,
                "router_s": round(ct, 2), "answer_s": 0.0}
    ans, dt = _escalate(user_text, oai)
    return {"route": "ESCALATE", "why": "instruction uncertain (self-inconsistent)",
            "backend": FRONTIER_MODEL, "answer": ans,
            "router_s": round(ct, 2), "answer_s": round(dt, 2)}


DEMO = [
    "What is the capital of Japan?",
    "What is 47 times 19?",
    "Define photosynthesis in one sentence.",
    "Rewrite 'hey can u send me that file' more formally.",
    "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball?",
    "A store sells notebooks at $12.50 each. How much do 7 notebooks cost?",
    "Each crate weighs 23.7 kg. What do 41 crates weigh?",
    "A factory makes 1,847 widgets per day. How many widgets in 263 days?",
    "A shirt costs $40 after a 20% discount. What was the original price?",
    "If a chicken and a half lays an egg and a half in a day and a half, how many eggs does one chicken lay in one day?",
    "What is 17 to the power of 4?",
    "Prove that the square root of 2 is irrational.",
    "Write a Python function that returns the longest palindromic substring of a string.",
    "I have a 3-liter and a 5-liter jug and unlimited water. How do I measure exactly 4 liters?",
]


def demo():
    print(f"{'#':>2}  {'ROUTE':<9} {'why':<24} {'lat':>6}  query")
    print("-" * 92)
    solved = local = esc = err = 0
    tsolved = tlocal = tesc = 0.0
    for i, q in enumerate(DEMO, 1):
        r = route(q)
        tot = r["router_s"] + r["answer_s"]
        if r["route"] == "SOLVED":
            solved += 1; tsolved += tot
        elif r["route"] == "LOCAL":
            local += 1; tlocal += tot
        elif r["route"] == "ESCALATE":
            esc += 1; tesc += tot
        else:
            err += 1
        print(f"{i:>2}  {r['route']:<9} {r['why']:<24} {tot:>5.1f}s  {q[:40]}")
        print(f"     -> {r['answer'][:108].replace(chr(10), ' ')}")
    n = len(DEMO)
    on_box = solved + local
    print("-" * 92)
    print(f"SOLVED:    {solved}/{n} ({100*solved//n}%)  avg {tsolved/max(solved,1):.2f}s   python exact / free")
    print(f"LOCAL:     {local}/{n} ({100*local//n}%)  avg {tlocal/max(local,1):.1f}s   {LOCAL_MODEL} / free / private")
    print(f"ESCALATED: {esc}/{n} ({100*esc//n}%)  avg {tesc/max(esc,1):.1f}s   frontier ({FRONTIER_MODEL})")
    if err:
        print(f"ERRORS:    {err}/{n}  (backend unavailable — check OLLAMA_URL / FRONTIER_* config)")
    print(f"ON-BOX:    {on_box}/{n} ({100*on_box//n}%)  answered with no frontier call")


def main():
    """Console entry point (installed as `hybrid`)."""
    if len(sys.argv) > 1 and sys.argv[1] in ("--version", "-V"):
        print(f"hybrid {__version__}")
        return
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
        return
    q = " ".join(sys.argv[1:]) or sys.stdin.read().strip()
    r = route(q)
    print(f"[{r['route']}: {r['why']} -> {r['backend']}  ({r['router_s']+r['answer_s']:.1f}s)]")
    print(r["answer"])


if __name__ == "__main__":
    main()
