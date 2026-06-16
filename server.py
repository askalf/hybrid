#!/usr/bin/env python3
"""
OpenAI-compatible front end for the hybrid router. Point any OpenAI client at
http://localhost:8080/v1 (model: "hybrid") and every request routes local-first,
escalating only the hard ones to a frontier model — transparently.

  python server.py
  curl http://localhost:8080/v1/chat/completions \
    -d '{"model":"hybrid","messages":[{"role":"user","content":"capital of France?"}]}'

The response carries an "x_hybrid" field (route / why / backend / latency) so you
can see which tier answered. Dependency-free (stdlib only); reuses hybrid.route().

Config (env): PORT (default 8080).
"""
import os, json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hybrid

PORT = int(os.environ.get("PORT", "8080"))


class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list",
                             "data": [{"id": "hybrid", "object": "model", "owned_by": "local"}]})
        else:
            self._json(200, {"status": "ok", "service": "hybrid"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self._json(404, {"error": {"message": "not found"}})
        try:
            n = int(self.headers.get("content-length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"error": {"message": "bad json"}})
        query = next((m.get("content", "") for m in reversed(req.get("messages", []))
                      if m.get("role") == "user"), "")
        if not query:
            return self._json(400, {"error": {"message": "no user message"}})
        r = hybrid.route(query)
        self._json(200, {
            "id": "chatcmpl-hybrid", "object": "chat.completion", "created": int(time.time()),
            "model": f"hybrid:{'local' if r['route'] == 'LOCAL' else 'frontier'}",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": r["answer"]}}],
            "x_hybrid": {"route": r["route"], "why": r["why"], "backend": r["backend"],
                         "latency_s": round(r["router_s"] + r["answer_s"], 2)},
        })

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"hybrid OpenAI-compatible server -> http://localhost:{PORT}/v1   (model id: 'hybrid')")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
