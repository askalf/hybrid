#!/usr/bin/env python3
"""
Tests for the OpenAI-compatible server — offline (hybrid.route is faked, the HTTP
surface is real: a ThreadingHTTPServer on an ephemeral loopback port).

What must hold:
  1. PROTOCOL   — completions round-trip; `stream: true` emits well-formed SSE ending
                  in [DONE]; multi-turn conversations reach route() whole.
  2. LIMITS     — body cap (413), bad json / no user message (400), unknown paths (404).
  3. AUTH       — HYBRID_API_KEY gates everything except /health.
  4. ERRORS     — a route ERROR maps to 502 with an OpenAI-shaped error object,
                  never a 200 with error-shaped content.
  5. OBSERVABILITY — one JSONL decision line per request; query text only on opt-in.

    python test_server.py
"""
import json, os, tempfile, threading, urllib.error, urllib.request

import hybrid
import server

FAILS = []
COUNT = [0]


def check(name, cond, detail=""):
    COUNT[0] += 1
    print(f"{'ok ' if cond else 'XX '} {name:<52} {str(detail)[:44]}")
    if not cond:
        FAILS.append((name, detail))


CALLS = []


def fake_route(query, messages=None):
    CALLS.append((query, messages))
    if query == "boom":
        return {"route": "ERROR", "why": "local backend unavailable", "backend": "local",
                "answer": "[local backend unavailable: fake]",
                "router_s": 0.0, "answer_s": 0.0, "error": True}
    if query == "exact":
        return {"route": "SOLVED", "why": "deterministic arithmetic",
                "backend": "python (exact)", "answer": "42",
                "router_s": 0.0, "answer_s": 0.0}
    return {"route": "LOCAL", "why": "test-fake", "backend": "fake-model",
            "answer": "the answer is 42", "router_s": 0.1, "answer_s": 0.2}


hybrid.route = fake_route
LOG = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
LOG.close()
os.environ["HYBRID_LOG"] = LOG.name

SRV = server.ThreadingHTTPServer(("127.0.0.1", 0), server.H)
PORT = SRV.server_address[1]
threading.Thread(target=SRV.serve_forever, daemon=True).start()


def call(path, body=None, headers=None, method=None):
    """-> (status, parsed-or-raw body, headers dict)"""
    url = f"http://127.0.0.1:{PORT}{path}"
    data = None if body is None else (body if isinstance(body, bytes)
                                      else json.dumps(body).encode())
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"content-type": "application/json",
                                          **(headers or {})})
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        raw, code, hdrs = resp.read(), resp.status, dict(resp.headers)
    except urllib.error.HTTPError as e:
        raw, code, hdrs = e.read(), e.code, dict(e.headers)
    try:
        return code, json.loads(raw), hdrs
    except Exception:
        return code, raw, hdrs


def last_log():
    with open(LOG.name, encoding="utf-8") as f:
        return json.loads(f.readlines()[-1])


def main():
    # --- 1. protocol -----------------------------------------------------------
    code, body, _ = call("/health")
    check("/health is live and reports the version",
          code == 200 and body.get("version") == hybrid.__version__, body)

    code, body, _ = call("/v1/models")
    check("/v1/models lists model 'hybrid'",
          code == 200 and body["data"][0]["id"] == "hybrid", body)

    code, body, _ = call("/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "hello"}]})
    check("completion round-trips with x_hybrid + usage",
          code == 200 and body["choices"][0]["message"]["content"] == "the answer is 42"
          and body["x_hybrid"]["route"] == "LOCAL" and body["usage"]["total_tokens"] > 0
          and body["model"] == "hybrid:local", body.get("model"))

    code, body, _ = call("/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "exact"}]})
    check("SOLVED answers are model 'hybrid:local' (on-box)",
          code == 200 and body["model"] == "hybrid:local", body.get("model"))

    convo = [{"role": "system", "content": "be brief"},
             {"role": "user", "content": "first question"},
             {"role": "assistant", "content": "first answer"},
             {"role": "user", "content": "second question"}]
    CALLS.clear()
    code, body, _ = call("/v1/chat/completions", {"messages": convo})
    check("multi-turn: routes on last user msg, passes whole conversation",
          code == 200 and CALLS[-1][0] == "second question" and CALLS[-1][1] == convo,
          CALLS[-1][0])

    CALLS.clear()
    call("/v1/chat/completions", {"messages": [{"role": "user", "content": "solo"}]})
    check("single-turn: no conversation passed (stub-safe path)",
          CALLS[-1] == ("solo", None), CALLS[-1])

    code, raw, hdrs = call("/v1/chat/completions",
                           {"stream": True,
                            "messages": [{"role": "user", "content": "hello"}]})
    lines = [ln for ln in raw.decode().split("\n") if ln.startswith("data: ")]
    chunks = [json.loads(ln[6:]) for ln in lines if ln != "data: [DONE]"]
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    check("stream: SSE content-type, delta chunks, [DONE] terminal",
          code == 200 and hdrs.get("content-type", "").startswith("text/event-stream")
          and lines[-1] == "data: [DONE]" and content == "the answer is 42"
          and chunks[-1]["choices"][0]["finish_reason"] == "stop"
          and chunks[-1]["x_hybrid"]["route"] == "LOCAL",
          f"{len(lines)} data lines")

    # --- 2. limits -------------------------------------------------------------
    code, body, _ = call("/v1/chat/completions", b"{not json")
    check("bad json -> 400", code == 400, body)

    code, body, _ = call("/v1/chat/completions", {"messages": []})
    check("no user message -> 400", code == 400, body)

    os.environ["HYBRID_MAX_BODY"] = "64"
    code, body, _ = call("/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "x" * 200}]})
    os.environ.pop("HYBRID_MAX_BODY")
    check("oversized body -> 413 (cap re-read per request)", code == 413, body)

    code, body, _ = call("/v1/embeddings", {"input": "x"})
    check("unknown POST path -> 404", code == 404, body)
    code, body, _ = call("/v1/nope")
    check("unknown GET path -> 404", code == 404, body)

    # --- 3. auth ---------------------------------------------------------------
    os.environ["HYBRID_API_KEY"] = "sekrit"
    code, _, _ = call("/v1/chat/completions",
                      {"messages": [{"role": "user", "content": "hello"}]})
    check("auth on: bare request -> 401", code == 401)
    code, _, _ = call("/v1/chat/completions",
                      {"messages": [{"role": "user", "content": "hello"}]},
                      headers={"authorization": "Bearer sekrit"})
    check("auth on: correct bearer -> 200", code == 200)
    code, _, _ = call("/health")
    check("auth on: /health stays open (liveness probes)", code == 200)
    os.environ.pop("HYBRID_API_KEY")

    # --- 4. errors -------------------------------------------------------------
    code, body, _ = call("/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "boom"}]})
    check("route ERROR -> 502 with error object, never 200-shaped",
          code == 502 and body["error"]["type"] == "backend_unavailable"
          and body["x_hybrid"]["route"] == "ERROR", body.get("error", {}).get("type"))

    # --- 5. observability ------------------------------------------------------
    call("/v1/chat/completions", {"messages": [{"role": "user", "content": "log me"}]})
    rec = last_log()
    check("decision line: route/why/latency present, query text absent",
          rec["route"] == "LOCAL" and "q_sha" in rec and "query" not in rec
          and rec["status"] == 200, rec.get("q_sha"))

    os.environ["HYBRID_LOG_QUERIES"] = "1"
    call("/v1/chat/completions", {"messages": [{"role": "user", "content": "log me too"}]})
    os.environ.pop("HYBRID_LOG_QUERIES")
    check("decision line: query text on explicit opt-in",
          last_log().get("query") == "log me too", last_log().get("query"))

    SRV.shutdown()
    os.unlink(LOG.name)
    print("-" * 72)
    if FAILS:
        print(f"FAIL  {len(FAILS)}/{COUNT[0]}")
        for name, detail in FAILS:
            print(f"   {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS  {COUNT[0]}/{COUNT[0]}")


if __name__ == "__main__":
    main()
