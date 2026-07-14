#!/usr/bin/env python3
"""
Tests for per-request token accounting — the decision log's measured-spend field.

All offline: the transports are exercised against a faked urllib (scripted backend
responses carrying each backend's own token fields), and the routing-level checks
fake ollama()/escalate() the same way test_route does. Pins:

  1. TALLY MECHANICS — no tally outside a request (no-op, never a crash); adds
     accumulate per tier; garbage counts read as 0 but the call still counts.
  2. TRANSPORT EXTRACTION — Ollama's prompt_eval_count/eval_count, llama-server's
     tokens_evaluated/tokens_predicted, and the frontier's usage object each land
     on the right tier of the current tally.
  3. THREAD PROPAGATION — the vote's worker threads add to the tally of the
     request that voted (contextvars copied per submit), and concurrent requests
     keep separate tallies.
  4. RESULT SHAPE — route() and route_messages() always attach `tokens`; SOLVED
     shows all zeros (that IS the datapoint); escalation counts frontier calls.

    python test_tokens.py
"""
import json
import threading
import urllib.request
import hybrid

_REAL_OLLAMA, _REAL_ESCALATE = hybrid.ollama, hybrid.escalate
_REAL_URLOPEN = urllib.request.urlopen
_REAL_BACKEND = hybrid.LOCAL_BACKEND

FAILS = []
COUNT = [0]


def check(name, cond, detail=""):
    COUNT[0] += 1
    print(f"{'ok ' if cond else 'XX '} {name:<52} {str(detail)[:44]}")
    if not cond:
        FAILS.append((name, detail))


class FakeHTTP:
    """Stands in for urllib.request.urlopen: returns the scripted payload."""

    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def __call__(self, req, timeout=None):
        self.calls += 1
        payload = self.payload
        class R:
            def read(self):
                return json.dumps(payload).encode()
        return R()


# ── 1. tally mechanics ──────────────────────────────────────────────────────
# Run the no-tally check on a fresh thread so no earlier reset can leak in.
_noop_err = []


def _noop_probe():
    try:
        hybrid._tokens_add("local", 5, 5)   # no reset has happened on this thread
    except Exception as e:                   # pragma: no cover - the failure itself
        _noop_err.append(e)


th = threading.Thread(target=_noop_probe)
th.start(); th.join()
check("no tally -> _tokens_add is a silent no-op", not _noop_err, repr(_noop_err))

t = hybrid._tokens_reset()
check("fresh tally starts at zero", t == {"local_in": 0, "local_out": 0,
                                          "local_calls": 0, "frontier_in": 0,
                                          "frontier_out": 0, "frontier_calls": 0}, t)
hybrid._tokens_add("local", 10, 5)
hybrid._tokens_add("local", None, 3)        # backend omitted a count -> 0, call counts
hybrid._tokens_add("frontier", 100, 50)
check("local adds accumulate", (t["local_in"], t["local_out"], t["local_calls"]) == (10, 8, 2), t)
check("frontier adds separate", (t["frontier_in"], t["frontier_out"], t["frontier_calls"]) == (100, 50, 1), t)
hybrid._tokens_add("frontier", "garbage", object())
check("garbage counts read as 0, call still counted",
      (t["frontier_in"], t["frontier_out"], t["frontier_calls"]) == (100, 50, 2), t)

# ── 2. transport extraction ─────────────────────────────────────────────────
t = hybrid._tokens_reset()
hybrid.LOCAL_BACKEND = "ollama"
urllib.request.urlopen = FakeHTTP({"response": "hi", "prompt_eval_count": 11, "eval_count": 4})
try:
    ans, _ = hybrid.ollama("q")
finally:
    urllib.request.urlopen = _REAL_URLOPEN
    hybrid.LOCAL_BACKEND = _REAL_BACKEND
check("ollama transport extracts eval counts", ans == "hi" and
      (t["local_in"], t["local_out"], t["local_calls"]) == (11, 4, 1), t)

t = hybrid._tokens_reset()
urllib.request.urlopen = FakeHTTP({"content": "x", "tokens_evaluated": 20, "tokens_predicted": 3})
try:  # family=None -> no /slots probe, so the fake serves only the /completion call
    r, _ = hybrid._llamacpp_call("q", 16, 0.0, None, None, None)
finally:
    urllib.request.urlopen = _REAL_URLOPEN
check("llamacpp transport extracts tokens_evaluated/predicted",
      r["content"] == "x" and (t["local_in"], t["local_out"], t["local_calls"]) == (20, 3, 1), t)

t = hybrid._tokens_reset()
urllib.request.urlopen = FakeHTTP({"choices": [{"message": {"content": "deep"}}],
                                   "usage": {"prompt_tokens": 100, "completion_tokens": 42}})
try:
    ans, _ = hybrid.escalate("hard q")
finally:
    urllib.request.urlopen = _REAL_URLOPEN
check("frontier extracts usage", ans == "deep" and
      (t["frontier_in"], t["frontier_out"], t["frontier_calls"]) == (100, 42, 1), t)

t = hybrid._tokens_reset()
urllib.request.urlopen = FakeHTTP({"choices": [{"message": {"content": "ok"}}]})
try:  # a frontier that reports no usage at all still counts the call
    ans, _ = hybrid.escalate("q")
finally:
    urllib.request.urlopen = _REAL_URLOPEN
check("usage-less frontier counts the call at 0 tokens", ans == "ok" and
      (t["frontier_in"], t["frontier_out"], t["frontier_calls"]) == (0, 0, 1), t)

# ── 3. thread propagation ───────────────────────────────────────────────────
def counting_ollama(prompt, num_predict=256, temperature=0.0, model=None,
                    grammar=None, family=None):
    hybrid._tokens_add("local", 7, 2)   # what the real transport does on success
    return "42", 0.01


t = hybrid._tokens_reset()
hybrid.ollama = counting_ollama
try:
    unanimous, best, agree, _ = hybrid._vote("p", 3, "m", 16)
finally:
    hybrid.ollama = _REAL_OLLAMA
check("vote workers add to the voter's tally", unanimous and
      (t["local_in"], t["local_out"], t["local_calls"]) == (21, 6, 3), t)

results = {}


def _one_request(name):
    hybrid.ollama = counting_ollama          # already set; assignment is idempotent
    r = hybrid.route("What is the capital of France?")
    results[name] = r


hybrid.ollama = counting_ollama
try:
    a = threading.Thread(target=_one_request, args=("a",))
    b = threading.Thread(target=_one_request, args=("b",))
    a.start(); b.start(); a.join(); b.join()
finally:
    hybrid.ollama = _REAL_OLLAMA
check("concurrent requests keep separate tallies",
      results["a"]["tokens"]["local_calls"] == 3 and
      results["b"]["tokens"]["local_calls"] == 3,
      (results["a"]["tokens"], results["b"]["tokens"]))

# ── 4. result shape ─────────────────────────────────────────────────────────
r = hybrid.route("What is 47 times 19?")
check("SOLVED carries an all-zero tally", r["route"] == "SOLVED" and
      r["tokens"]["local_calls"] == 0 and r["tokens"]["frontier_calls"] == 0, r["tokens"])


def counting_escalate(query, messages=None):
    hybrid._tokens_add("frontier", 300, 120)
    return "frontier answer", 0.01


hybrid.escalate = counting_escalate
try:
    r = hybrid.route("Prove that the square root of 2 is irrational.")
finally:
    hybrid.escalate = _REAL_ESCALATE
check("ESCALATE counts the frontier call", r["route"] == "ESCALATE" and
      (r["tokens"]["frontier_in"], r["tokens"]["frontier_out"],
       r["tokens"]["frontier_calls"]) == (300, 120, 1), r["tokens"])


def label_ollama(prompt, num_predict=256, temperature=0.0, model=None,
                 grammar=None, family=None):
    hybrid._tokens_add("local", 30, 1)
    return "build", 0.01


hybrid.ollama = label_ollama
hybrid.LOCAL_BACKEND = "ollama"              # vote path (logit read is llamacpp-only)
try:
    r = hybrid.route_messages("Classify the request.",
                              [{"role": "user", "content": "add CI caching"}],
                              labels=["build", "research"])
finally:
    hybrid.ollama = _REAL_OLLAMA
    hybrid.LOCAL_BACKEND = _REAL_BACKEND
check("labelled classification tallies its vote", r["route"] == "LOCAL" and
      r["answer"] == "build" and r["tokens"]["local_calls"] == 3, r["tokens"])

print(f"\n{COUNT[0] - len(FAILS)}/{COUNT[0]} passed")
if FAILS:
    raise SystemExit(1)
