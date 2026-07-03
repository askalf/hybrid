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
"""
import sys, os, time, json, re, urllib.request
from collections import Counter
from solver import solve
import equations
import templates
import verify

__version__ = "1.6.0"

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 chokes on non-ASCII
except Exception:
    pass

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
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


def ollama(prompt, num_predict=256, temperature=0.0, model=None):
    """One local-model call (default LOCAL_MODEL; the vote/creative tiers pass
    LOCAL_MODEL_FAST). Retries once (transports flake), then raises BackendError —
    an answer string is ALWAYS a real model answer, never an error in disguise."""
    body = json.dumps({
        "model": model or LOCAL_MODEL, "prompt": prompt, "stream": False, "keep_alive": "5m",
        "options": {"num_predict": num_predict, "temperature": temperature},
    }).encode()
    t0 = time.time()
    last = None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(OLLAMA_URL, data=body,
                                         headers={"content-type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=300).read())
            return r.get("response", "").strip(), time.time() - t0
        except Exception as e:
            last = e
            if attempt == 1:
                time.sleep(1.0)
    raise BackendError("local", f"{model or LOCAL_MODEL} at {OLLAMA_URL}: {last}")


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
        return r["choices"][0]["message"]["content"].strip(), time.time() - t0
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
    """Sample the fast local model k times; 'confident' iff all answers agree."""
    samples, t = [], 0.0
    for _ in range(k):
        a, dt = ollama(CONCISE.format(q=query), num_predict=80, temperature=0.6,
                       model=LOCAL_MODEL_FAST)
        samples.append(a); t += dt
    keys = [_key(s) for s in samples]
    top, n = Counter(keys).most_common(1)[0]
    best = next(s for s, kk in zip(samples, keys) if kk == top)
    return (n >= k), best, n, t  # unanimous: escalate unless the model fully agrees with itself


def route(query, messages=None):
    """Route one query. `messages` (optional) is the full OpenAI-style conversation:
    routing decisions and the LOCAL tiers always work on `query` — the last user
    message — but an escalated call carries the whole conversation to the frontier.
    Never raises for a dead backend — the failure policy (env, read per-call so a
    live service can be re-tuned) turns a BackendError into either a degraded route
    or an explicit ERROR result the caller can surface."""
    try:
        return _route(query, messages)
    except BackendError as e:
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


def _route(query, messages=None):
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
    # 1. known-hard categories -> escalate by rule.
    if _HARD.search(query) or len(query) > 220:
        ans, dt = _escalate(query, messages)
        return {"route": "ESCALATE", "why": "rule: hard category", "backend": FRONTIER_MODEL,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    # 2. open-ended / creative -> keep local (no single right answer, fast model fine).
    if _OPEN.search(query):
        ans, dt = ollama(CONCISE.format(q=query), num_predict=200, model=LOCAL_MODEL_FAST)
        return {"route": "LOCAL", "why": "open-ended (local ok)", "backend": LOCAL_MODEL_FAST,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    # 3. quantitative queries get the exact oracle, strongest signal first:
    #    derive (independent re-derivation) > plug-back (consistency) > vote (agreement).
    if _quantitative(query):
        # setup re-derivation (equations.py): the model transcribes the problem's
        # relationships as equations, we solve the linear system OURSELVES (exact
        # Fractions, free) and compare to its answer. A mismatch means the model
        # mis-solved its own transcription -> HARD escalate. This runs FIRST because
        # plug-back can be fooled by a tautology (a true-but-disconnected check like
        # `(1.5/1.5)*1 = 1` reads as "checked"); a derivation can't — it produces its
        # own value for the answer instead of grading the model's checks.
        raw, dt = ollama(SETUP_PROMPT.format(q=query), num_predict=220, temperature=0.0)
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
        return {"route": "LOCAL", "why": f"self-consistent {agree}/3",
                "backend": LOCAL_MODEL_FAST,
                "answer": best, "router_s": round(ct, 2), "answer_s": 0.0}
    ans, dt = _escalate(query, messages)
    return {"route": "ESCALATE", "why": "uncertain (self-inconsistent)", "backend": FRONTIER_MODEL,
            "answer": ans, "router_s": round(ct, 2), "answer_s": round(dt, 2)}


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
