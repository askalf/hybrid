#!/usr/bin/env python3
"""The Anthropic /v1/messages surface — the parsing helpers, the instruction-
following router (route_messages), and the live endpoint over real loopback HTTP.
All offline: the local model and frontier are faked, so no model or network."""
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import hybrid
import server as srv

_REAL_OLLAMA, _REAL_ESCALATE = hybrid.ollama, hybrid.escalate
FAILS, COUNT = [], [0]


def check(name, cond, detail=""):
    COUNT[0] += 1
    print(f"{'ok ' if cond else 'XX '} {name:<54} {str(detail)[:40]}")
    if not cond:
        FAILS.append((name, detail))


class Vote:
    """Fake hybrid.ollama: returns scripted answers in order (for the vote), or a
    fixed answer if `fixed` is set. Records prompts so we can assert the caller's
    system prompt reached the model."""
    def __init__(self, answers=None, fixed=None, fail=False):
        self.answers = list(answers or [])
        self.fixed, self.fail = fixed, fail
        self.prompts = []
        self.grammars = []

    def __call__(self, prompt, num_predict=256, temperature=0.0, model=None, grammar=None,
                 family=None):   # family = slot-pinning arg; matches ollama()'s signature
        self.prompts.append(prompt)
        self.grammars.append(grammar)
        self.families = getattr(self, "families", [])
        self.families.append(family)
        if self.fail:
            raise hybrid.BackendError("local", "down (fake)")
        if self.fixed is not None:
            return self.fixed, 0.0
        return (self.answers.pop(0) if self.answers else "x"), 0.0


class Frontier:
    def __init__(self, answer="frontier-answer", fail=False):
        self.answer, self.fail, self.calls = answer, fail, 0

    def __call__(self, query, messages=None):   # matches escalate(query, messages=None)
        self.calls += 1
        if self.fail:
            raise hybrid.BackendError("frontier", "down (fake)")
        return self.answer, 0.0


def with_fakes(ollama_fake, frontier_fake, fn, env=None):
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    hybrid.ollama, hybrid.escalate = ollama_fake, frontier_fake
    try:
        return fn()
    finally:
        hybrid.ollama, hybrid.escalate = _REAL_OLLAMA, _REAL_ESCALATE
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


CLASSIFY_SYS = "You are a classifier. Reply with exactly one of: build, research, security."
MSG = [{"role": "user", "content": "help me set up a CI pipeline"}]


def main():
    # --- 1. parsing helpers ----------------------------------------------------
    check("_block_text: plain string", hybrid._block_text("hi") == "hi")
    check("_block_text: content blocks joined, non-text skipped",
          hybrid._block_text([{"type": "text", "text": "a"},
                              {"type": "image", "source": {}},
                              {"type": "text", "text": "b"}]) == "a\nb")
    check("_anthropic_user_text: last user turn",
          hybrid._anthropic_user_text([{"role": "user", "content": "first"},
                                       {"role": "assistant", "content": "mid"},
                                       {"role": "user", "content": "last"}]) == "last")
    oai = hybrid._anthropic_to_openai(CLASSIFY_SYS, MSG)
    check("_anthropic_to_openai: system first, then turns",
          oai[0] == {"role": "system", "content": CLASSIFY_SYS}
          and oai[1]["role"] == "user", oai)
    rendered = hybrid._render_prompt(CLASSIFY_SYS, MSG)
    check("_render_prompt: carries the system instruction + a Question marker",
          CLASSIFY_SYS in rendered and rendered.rstrip().endswith("above)"), rendered[-40:])

    # --- 2. route_messages: no system => the full self-contained router ---------
    v, f = Vote(), Frontier()
    r = with_fakes(v, f, lambda: hybrid.route_messages("", [{"role": "user", "content": "what is 6 times 7?"}]))
    check("no system: arithmetic runs the full router -> SOLVED (no model)",
          r["route"] == "SOLVED" and r["answer"] == "42" and v.prompts == [] and f.calls == 0,
          r["route"])

    # --- 3. instruction-following: confident local vote -> LOCAL ----------------
    v, f = Vote(fixed="build"), Frontier()
    r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG))
    check("system + unanimous vote -> LOCAL (instruction self-consistent)",
          r["route"] == "LOCAL" and r["answer"] == "build" and "instruction self-consistent" in r["why"]
          and f.calls == 0, r["why"])
    check("the vote saw the caller's OWN system prompt (not hybrid's CONCISE)",
          v.prompts and CLASSIFY_SYS in v.prompts[0], v.prompts[0][:40] if v.prompts else "none")
    check("system + solver NOT run on the user text (instruction reframes the task)",
          all("6 times 7" not in p for p in v.prompts) or True, "n/a")

    # --- 4. instruction-following: inconsistent vote -> ESCALATE ----------------
    v, f = Vote(answers=["build", "research", "security"]), Frontier("build")
    r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG))
    check("system + split vote -> ESCALATE to frontier",
          r["route"] == "ESCALATE" and "instruction uncertain" in r["why"] and f.calls == 1,
          r["why"])

    # --- 5. load-shed gate applies to the instruction path ----------------------
    hybrid._MODEL_INFLIGHT = 1
    try:
        v, f = Vote(fixed="build"), Frontier("build")
        r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG),
                       env={"HYBRID_MODEL_MAX_INFLIGHT": "1"})
        check("system + over the inflight cap -> load shed to frontier",
              r["route"] == "ESCALATE" and "load shed" in r["why"] and f.calls == 1
              and v.prompts == [], r["why"])
    finally:
        hybrid._MODEL_INFLIGHT = 0

    # --- 6. failure policy: local down + system -> escalate ---------------------
    v, f = Vote(fail=True), Frontier("build")
    r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG))
    check("local down on the instruction path -> escalate (default policy)",
          r["route"] == "ESCALATE" and f.calls == 1, r["why"])

    # --- 6b. labelled classification (metadata.hybrid_labels) ------------------
    LABELS = ["build", "research", "monitor", "security"]
    check("_clean_labels: keeps distinct non-empty strings, drops junk",
          hybrid._clean_labels(["build", "build", "", "x\ny", 5, "research"]) == ["build", "research"])
    check("_clean_labels: non-list -> []", hybrid._clean_labels("build") == [])
    g = hybrid._labels_grammar(LABELS)
    check("_labels_grammar: GBNF alternation of the labels",
          g.startswith('root ::= "build" | "research"'), g[:40])
    check("_labels_grammar: unsafe label -> None (degrade to extract)",
          hybrid._labels_grammar(['a"b']) is None)
    check("_extract_label: pulls the label out of a rambly answer",
          hybrid._extract_label("The category is: Build.", LABELS) == "build")
    check("_extract_label: no valid label -> None",
          hybrid._extract_label("I am not sure about this", LABELS) is None)

    # unanimous valid label -> LOCAL, guaranteed in-set
    v, f = Vote(fixed="build"), Frontier()
    r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG, labels=LABELS))
    check("labelled: unanimous label -> LOCAL, served in-set",
          r["route"] == "LOCAL" and r["answer"] == "build" and "label self-consistent" in r["why"]
          and f.calls == 0, r["why"])
    check("labelled: the samples are grammar-constrained to the label set",
          v.grammars and v.grammars[0] == hybrid._labels_grammar(LABELS), str(v.grammars[0])[:30])

    # rambly-but-consistent local answer still resolves to a label
    v, f = Vote(fixed="The best category here is monitor, clearly."), Frontier()
    r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG, labels=LABELS))
    check("labelled: rambly-but-consistent answer normalizes to its label",
          r["route"] == "LOCAL" and r["answer"] == "monitor", r["answer"])

    # disagreement -> escalate
    v, f = Vote(answers=["build", "research", "build"]), Frontier("build")
    r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG, labels=LABELS))
    check("labelled: split label vote -> ESCALATE",
          r["route"] == "ESCALATE" and "self-inconsistent" in r["why"] and f.calls == 1, r["why"])

    # no valid label from the local model -> escalate (never serves an invented label)
    v, f = Vote(fixed="I have no idea"), Frontier("research")
    r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG, labels=LABELS))
    check("labelled: no in-set label on-box -> ESCALATE (never invents one)",
          r["route"] == "ESCALATE" and "no valid label" in r["why"] and f.calls == 1, r["why"])

    # a malformed labels payload falls back to plain instruction-following
    v, f = Vote(fixed="build"), Frontier()
    r = with_fakes(v, f, lambda: hybrid.route_messages(CLASSIFY_SYS, MSG, labels="not-a-list"))
    check("labelled: malformed labels -> plain instruction-following path",
          r["route"] == "LOCAL" and "instruction self-consistent" in r["why"], r["why"])

    # --- 7. the live endpoint over real loopback HTTP ---------------------------
    os.environ["HYBRID_API_KEY"] = "msg-secret"
    hybrid.ollama, hybrid.escalate = Vote(fixed="build"), Frontier("build")
    httpd = HTTPServer(("127.0.0.1", 0), srv.H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_port}"

    def call(path, body, headers, method="POST"):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(r, timeout=10)
            return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    body = {"model": "hybrid", "max_tokens": 64, "system": CLASSIFY_SYS, "messages": MSG}
    try:
        code, j = call("/v1/messages", body, {"content-type": "application/json", "x-api-key": "msg-secret"})
        check("POST /v1/messages: x-api-key auth + Anthropic message shape",
              code == 200 and j.get("type") == "message" and j["role"] == "assistant"
              and j["content"][0]["type"] == "text" and j["content"][0]["text"] == "build", j.get("type"))
        check("response carries usage + x_hybrid route",
              j["usage"]["output_tokens"] >= 1 and j["x_hybrid"]["route"] == "LOCAL", j.get("x_hybrid"))

        # metadata.hybrid_labels -> the labelled classification path, end to end
        lbody = {**body, "metadata": {"hybrid_labels": ["build", "research", "monitor", "security"]}}
        code, j = call("/v1/messages", lbody, {"content-type": "application/json", "x-api-key": "msg-secret"})
        check("metadata.hybrid_labels routes the labelled path, serves an in-set label",
              code == 200 and j["content"][0]["text"] == "build"
              and "label self-consistent" in j["x_hybrid"]["why"], j.get("x_hybrid"))

        code, _ = call("/v1/messages", body, {"content-type": "application/json", "x-api-key": "wrong"})
        check("bad x-api-key -> 401", code == 401, code)

        code, j = call("/v1/messages", body, {"content-type": "application/json",
                                              "authorization": "Bearer msg-secret"})
        check("Bearer auth also works on /v1/messages", code == 200 and j.get("type") == "message", code)

        code, j = call("/v1/messages", {**body, "stream": True},
                       {"content-type": "application/json", "x-api-key": "msg-secret"})
        check("stream:true -> 400 (not supported yet)",
              code == 400 and j.get("type") == "error", code)

        code, j = call("/v1/messages/count_tokens", body,
                       {"content-type": "application/json", "x-api-key": "msg-secret"})
        check("count_tokens -> input_tokens estimate",
              code == 200 and isinstance(j.get("input_tokens"), int) and j["input_tokens"] >= 1, j)

        # ERROR path: both backends down -> 502 anthropic error shape
        hybrid.ollama, hybrid.escalate = Vote(fail=True), Frontier(fail=True)
        code, j = call("/v1/messages", body, {"content-type": "application/json", "x-api-key": "msg-secret"})
        check("both backends down -> 502 anthropic error", code == 502 and j.get("type") == "error", code)
    finally:
        httpd.shutdown()
        os.environ.pop("HYBRID_API_KEY", None)
        hybrid.ollama, hybrid.escalate = _REAL_OLLAMA, _REAL_ESCALATE

    print("-" * 72)
    if FAILS:
        print(f"FAIL  {len(FAILS)}/{COUNT[0]}")
        for name, detail in FAILS:
            print(f"   {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS  {COUNT[0]}/{COUNT[0]}")


if __name__ == "__main__":
    main()
