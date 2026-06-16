#!/usr/bin/env python3
"""
hybrid — local-first LLM router with frontier escalation.

  EASY queries -> a small local model (Ollama)                  free, private, fast
  HARD queries -> a frontier model (any OpenAI-compatible API)  quality where it counts

The router decides per query: known-hard categories (code / proofs / puzzles /
big math) escalate by rule; open-ended / creative tasks stay local; everything
else runs through a self-consistency check and escalates only when the local
model disagrees with itself.

  python hybrid.py "your question"   # route one query
  python hybrid.py --demo            # mixed test set + summary

Config (env):
  OLLAMA_URL        default http://127.0.0.1:11434/api/generate
  LOCAL_MODEL       default qwen2.5:3b
  FRONTIER_URL      default https://api.openai.com/v1/chat/completions  (any OpenAI-compatible endpoint)
  FRONTIER_API_KEY  required for escalation
  FRONTIER_MODEL    default gpt-4o
"""
import sys, os, time, json, re, urllib.request
from collections import Counter

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 chokes on non-ASCII
except Exception:
    pass

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
LOCAL_MODEL = os.environ.get("LOCAL_MODEL", "qwen2.5:3b")
FRONTIER_URL = os.environ.get("FRONTIER_URL", "https://api.openai.com/v1/chat/completions")
FRONTIER_KEY = os.environ.get("FRONTIER_API_KEY", "")
FRONTIER_MODEL = os.environ.get("FRONTIER_MODEL", "gpt-4o")

CONCISE = "Answer directly and concisely - a number, a word, or one or two sentences, no working shown.\nQuestion: {q}"

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


def ollama(prompt, num_predict=256, temperature=0.0):
    body = json.dumps({
        "model": LOCAL_MODEL, "prompt": prompt, "stream": False, "keep_alive": "5m",
        "options": {"num_predict": num_predict, "temperature": temperature},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return r.get("response", "").strip(), time.time() - t0


def escalate(query):
    """Send to any OpenAI-compatible frontier endpoint (set FRONTIER_URL / _API_KEY / _MODEL)."""
    body = json.dumps({
        "model": FRONTIER_MODEL,
        "messages": [{"role": "user", "content": query}],
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
        return f"[escalation failed: {e} - set FRONTIER_URL / FRONTIER_API_KEY / FRONTIER_MODEL]", time.time() - t0


def _key(ans):
    """Normalize an answer to a comparable key for self-consistency voting."""
    s = ans.lower().strip()
    nums = re.findall(r"-?\$?\d[\d,]*\.?\d*", s)
    if nums:
        return re.sub(r"[^\d.]", "", nums[-1]).rstrip(".") or s[:30]
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s).split()[:6])


def local_consistency(query, k=3):
    """Sample the local model k times; 'confident' iff all answers agree."""
    samples, t = [], 0.0
    for _ in range(k):
        a, dt = ollama(CONCISE.format(q=query), num_predict=80, temperature=0.6)
        samples.append(a); t += dt
    keys = [_key(s) for s in samples]
    top, n = Counter(keys).most_common(1)[0]
    best = next(s for s, kk in zip(samples, keys) if kk == top)
    return (n >= k), best, n, t  # unanimous: escalate unless the model fully agrees with itself


def route(query):
    if _HARD.search(query) or len(query) > 220:
        ans, dt = escalate(query)
        return {"route": "ESCALATE", "why": "rule: hard category", "backend": FRONTIER_MODEL,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    if _OPEN.search(query):
        ans, dt = ollama(CONCISE.format(q=query), num_predict=200)
        return {"route": "LOCAL", "why": "open-ended (local ok)", "backend": LOCAL_MODEL,
                "answer": ans, "router_s": 0.0, "answer_s": round(dt, 2)}
    confident, best, agree, ct = local_consistency(query)
    if confident:
        return {"route": "LOCAL", "why": f"self-consistent {agree}/3", "backend": LOCAL_MODEL,
                "answer": best, "router_s": round(ct, 2), "answer_s": 0.0}
    ans, dt = escalate(query)
    return {"route": "ESCALATE", "why": "uncertain (self-inconsistent)", "backend": FRONTIER_MODEL,
            "answer": ans, "router_s": round(ct, 2), "answer_s": round(dt, 2)}


DEMO = [
    "What is the capital of Japan?",
    "What is 47 times 19?",
    "Define photosynthesis in one sentence.",
    "Rewrite 'hey can u send me that file' more formally.",
    "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball?",
    "If a chicken and a half lays an egg and a half in a day and a half, how many eggs does one chicken lay in one day?",
    "What is 17 to the power of 4?",
    "Prove that the square root of 2 is irrational.",
    "Write a Python function that returns the longest palindromic substring of a string.",
    "I have a 3-liter and a 5-liter jug and unlimited water. How do I measure exactly 4 liters?",
]


def demo():
    print(f"{'#':>2}  {'ROUTE':<9} {'why':<24} {'lat':>6}  query")
    print("-" * 92)
    local = esc = 0
    tlocal = tesc = 0.0
    for i, q in enumerate(DEMO, 1):
        r = route(q)
        tot = r["router_s"] + r["answer_s"]
        if r["route"] == "LOCAL":
            local += 1; tlocal += tot
        else:
            esc += 1; tesc += tot
        print(f"{i:>2}  {r['route']:<9} {r['why']:<24} {tot:>5.1f}s  {q[:40]}")
        print(f"     -> {r['answer'][:108].replace(chr(10), ' ')}")
    n = len(DEMO)
    print("-" * 92)
    print(f"LOCAL:     {local}/{n} ({100*local//n}%)  avg {tlocal/max(local,1):.1f}s   free / private")
    print(f"ESCALATED: {esc}/{n} ({100*esc//n}%)  avg {tesc/max(esc,1):.1f}s   frontier ({FRONTIER_MODEL})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo()
    else:
        q = " ".join(sys.argv[1:]) or sys.stdin.read().strip()
        r = route(q)
        print(f"[{r['route']}: {r['why']} -> {r['backend']}  ({r['router_s']+r['answer_s']:.1f}s)]")
        print(r["answer"])
