#!/usr/bin/env python3
"""
OpenAI-compatible front end for the hybrid router. Point any OpenAI client at
http://localhost:8080/v1 (model: "hybrid") and every request routes local-first,
escalating only the hard ones to a frontier model — transparently.

  python server.py
  curl http://localhost:8080/v1/chat/completions \
    -d '{"model":"hybrid","messages":[{"role":"user","content":"capital of France?"}]}'

Endpoints:
  POST /v1/chat/completions   `stream: true` supported (SSE). Multi-turn: routing and
                              the local tiers use the LAST user message; an escalated
                              call carries the whole conversation to the frontier.
  GET  /v1/models
  GET  /health                (and /) — unauthenticated liveness + version

Every request emits one JSONL decision line — route / why / backend / latency /
status — the observable trail of what the router decided. Query text stays OUT of
the log unless opted in; a sha256 prefix correlates repeats. The startup banner goes
to stderr so stdout is pure JSONL (pipe it, or point systemd/docker at it).

Responses carry an "x_hybrid" field (route / why / backend / latency) so callers can
see which tier answered. `usage` is a chars/4 ESTIMATE (flagged in x_hybrid) — the
local tier is not a token-metered API. Dependency-free (stdlib only).

Config (env — the knobs marked per-request are re-read on every request, so a live
service can be re-tuned without a restart):
  PORT                default 8080
  HYBRID_HOST         default 127.0.0.1. Set 0.0.0.0 to expose beyond the machine —
                      and set HYBRID_API_KEY when you do.
  HYBRID_API_KEY      per-request. If set, everything except /health requires
                      "Authorization: Bearer <key>".
  HYBRID_MAX_BODY     per-request. Request-body cap in bytes, default 1048576.
  HYBRID_LOG          per-request. Decision-log JSONL file (append); default stdout.
  HYBRID_LOG_QUERIES  per-request. "1" includes query text in the log (default off).
  HYBRID_CACHE_TTL    per-request. Seconds to serve a repeated single-turn query from
                      memory instead of re-routing it (default 0 = off). Real traffic
                      repeats; a hit answers in ~0 ms with x_hybrid.cached = true.
                      Multi-turn requests, ERROR results, and DEGRADED answers are
                      never cached.
  HYBRID_CACHE_MAX    per-request. Cache entry cap, LRU-evicted (default 512).
"""
import hashlib, hmac, json, os, sys, threading, time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hybrid

PORT = int(os.environ.get("PORT", "8080"))
HOST = os.environ.get("HYBRID_HOST", "127.0.0.1")


def _est_tokens(s):
    return max(1, len(s) // 4)


def _log_decision(rec):
    line = json.dumps(rec, separators=(",", ":"))
    path = os.environ.get("HYBRID_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    else:
        print(line, flush=True)


# --- answer cache (opt-in via HYBRID_CACHE_TTL) --------------------------------
# Single-turn only: with a conversation in play the same last message can mean a
# different thing, so multi-turn requests always re-route.
_CACHE = OrderedDict()               # sha256(query) -> (expiry, routed result)
_CACHE_LOCK = threading.Lock()


def _cache_get(key):
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if not hit:
            return None
        expiry, r = hit
        if time.time() > expiry:
            del _CACHE[key]
            return None
        _CACHE.move_to_end(key)
        return dict(r)


def _cache_put(key, r, ttl):
    try:
        cap = int(os.environ.get("HYBRID_CACHE_MAX", "512"))
    except ValueError:
        cap = 512
    with _CACHE_LOCK:
        _CACHE[key] = (time.time() + ttl, dict(r))
        _CACHE.move_to_end(key)
        while len(_CACHE) > max(1, cap):
            _CACHE.popitem(last=False)


class H(BaseHTTPRequestHandler):
    timeout = 30                 # a stalled client releases its thread
    server_version = "hybrid"    # don't advertise python/stdlib versions
    sys_version = ""

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _authed(self):
        key = os.environ.get("HYBRID_API_KEY", "")
        if not key:
            return True
        got = self.headers.get("authorization", "")
        if hmac.compare_digest(got, "Bearer " + key):
            return True
        self._json(401, {"error": {"message": "missing or invalid bearer token",
                                   "type": "invalid_request_error"}})
        return False

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("", "/health"):          # liveness is never behind auth
            return self._json(200, {"status": "ok", "service": "hybrid",
                                    "version": hybrid.__version__,
                                    "model_inflight": hybrid.model_inflight()})
        if not self._authed():
            return None                      # _authed already sent the 401
        if path.endswith("/models"):
            return self._json(200, {"object": "list",
                                    "data": [{"id": "hybrid", "object": "model",
                                              "owned_by": "local"}]})
        return self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self):
        if not self._authed():
            return None                      # _authed already sent the 401
        if not self.path.split("?")[0].rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": {"message": "not found",
                                              "type": "invalid_request_error"}})
        try:
            n = int(self.headers.get("content-length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            return self._json(411, {"error": {"message": "content-length required",
                                              "type": "invalid_request_error"}})
        cap = int(os.environ.get("HYBRID_MAX_BODY", "1048576"))
        if n > cap:
            return self._json(413, {"error": {"message": f"request body over {cap} bytes",
                                              "type": "invalid_request_error"}})
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": {"message": "bad json",
                                              "type": "invalid_request_error"}})
        messages = req.get("messages") or []
        query = next((m.get("content", "") for m in reversed(messages)
                      if m.get("role") == "user"), "")
        if not query:
            return self._json(400, {"error": {"message": "no user message",
                                              "type": "invalid_request_error"}})

        t0 = time.time()
        try:
            ttl = float(os.environ.get("HYBRID_CACHE_TTL", "0"))
        except ValueError:
            ttl = 0.0
        single_turn = len(messages) <= 1
        key = hashlib.sha256(query.encode()).hexdigest()
        r = _cache_get(key) if (ttl > 0 and single_turn) else None
        cached = r is not None
        if not cached:
            r = hybrid.route(query, messages=messages if len(messages) > 1 else None)
            if (ttl > 0 and single_turn and r["route"] in ("SOLVED", "LOCAL", "ESCALATE")
                    and "DEGRADED" not in r["why"]):
                _cache_put(key, r, ttl)
        xh = {"route": r["route"], "why": r["why"], "backend": r["backend"],
              "latency_s": 0.0 if cached else round(r["router_s"] + r["answer_s"], 2),
              "usage_estimated": True}
        if cached:
            xh["cached"] = True
        status = 502 if r["route"] == "ERROR" else 200
        rec = {"ts": round(time.time(), 3), "route": r["route"], "why": r["why"],
               "backend": r["backend"], "latency_s": xh["latency_s"],
               "wall_s": round(time.time() - t0, 2), "status": status,
               "stream": bool(req.get("stream")),
               "q_sha": key[:12],
               "q_chars": len(query)}
        if cached:
            rec["cached"] = True
        if os.environ.get("HYBRID_LOG_QUERIES") == "1":
            rec["query"] = query
        _log_decision(rec)

        if r["route"] == "ERROR":
            return self._json(502, {"error": {"message": r["answer"],
                                              "type": "backend_unavailable"},
                                    "x_hybrid": xh})

        model = f"hybrid:{'frontier' if r['route'] == 'ESCALATE' else 'local'}"
        usage = {"prompt_tokens": _est_tokens(query),
                 "completion_tokens": _est_tokens(r["answer"]),
                 "total_tokens": _est_tokens(query) + _est_tokens(r["answer"])}
        created = int(time.time())

        if req.get("stream"):
            # SSE-correct streaming: role delta, ONE content delta (routing has to
            # finish before the answer exists — the verify tiers see it whole), then
            # a stop chunk carrying x_hybrid/usage, then [DONE].
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.end_headers()
            base = {"id": "chatcmpl-hybrid", "object": "chat.completion.chunk",
                    "created": created, "model": model}
            for chunk in (
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant"},
                                      "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {"content": r["answer"]},
                                      "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                 "x_hybrid": xh, "usage": usage},
            ):
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            return None                      # stream already written

        return self._json(200, {
            "id": "chatcmpl-hybrid", "object": "chat.completion", "created": created,
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": r["answer"]}}],
            "usage": usage, "x_hybrid": xh,
        })

    def log_message(self, *a):
        pass                     # the decision log IS the access log


def main():
    """Console entry point (installed as `hybrid-server`)."""
    print(f"hybrid v{hybrid.__version__} -> http://{HOST}:{PORT}/v1  (model 'hybrid', "
          f"stream ok)   health: /health   decision log: "
          f"{os.environ.get('HYBRID_LOG', 'stdout')}", file=sys.stderr)
    ThreadingHTTPServer((HOST, PORT), H).serve_forever()


if __name__ == "__main__":
    main()
