#!/usr/bin/env python3
"""The llamacpp transport, tested against a real loopback HTTP server standing in for
llama-server's native /completion — what hybrid SENDS (ChatML wrap, cache_prompt,
grammar, stop) and how it degrades (retry once, then BackendError). Plus the grammar
sanity pins: llama-server SILENTLY generates unconstrained when a grammar fails to
parse (it only logs), so a malformed grammar here would not break anything visibly —
it would quietly disarm the constraint. These tests are the alarm."""
import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ["HYBRID_LOCAL_BACKEND"] = "llamacpp"   # before import: read at module load

import hybrid

FAILS = []
COUNT = [0]


def check(name, cond, detail=""):
    COUNT[0] += 1
    print(f"{'ok ' if cond else 'XX '} {name:<56} {str(detail)[:40]}")
    if not cond:
        FAILS.append((name, detail))


REQUESTS = []
BEHAVIOR = {"fail_times": 0}


class FakeLlamaServer(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        body["_port"] = self.server.server_port
        REQUESTS.append(body)
        if BEHAVIOR["fail_times"] > 0:
            BEHAVIOR["fail_times"] -= 1
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        out = json.dumps({"content": " 42 ", "timings": {"prompt_n": 1}}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def main():
    srv = HTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    hybrid.LLAMACPP_URL = f"http://127.0.0.1:{srv.server_port}/completion"

    # --- request shape -----------------------------------------------------
    ans, _ = hybrid.ollama(hybrid.SETUP_PROMPT.format(q="What is 2 plus 2?"),
                           num_predict=99, temperature=0.0, grammar=hybrid.GRAMMAR_SETUP)
    req = REQUESTS[-1]
    check("answer text comes from /completion 'content', stripped", ans == "42", ans)
    check("prompt is ChatML-wrapped", req["prompt"].startswith("<|im_start|>system\n"),
          req["prompt"][:34])
    check("instructions land in the system slot (cacheable prefix)",
          "Set the problem up as equations" in req["prompt"].split("<|im_end|>")[0],
          req["prompt"][:60])
    check("only the question lands in the user turn",
          "<|im_start|>user\nQuestion: What is 2 plus 2?<|im_end|>" in req["prompt"],
          req["prompt"][-120:])
    check("cache_prompt is on", req.get("cache_prompt") is True, req.get("cache_prompt"))
    check("n_predict passes through", req.get("n_predict") == 99, req.get("n_predict"))
    check("stop token closes the assistant turn", req.get("stop") == ["<|im_end|>"],
          req.get("stop"))
    check("grammar rides along when the tier passes one",
          req.get("grammar") == hybrid.GRAMMAR_SETUP, str(req.get("grammar"))[:30])

    hybrid.ollama("just a plain prompt with no question marker", grammar=None)
    check("no grammar field when the tier passes none", "grammar" not in REQUESTS[-1],
          list(REQUESTS[-1]))
    check("a prompt without the Question marker still wraps (all-user)",
          "just a plain prompt" in REQUESTS[-1]["prompt"], REQUESTS[-1]["prompt"][:60])

    os.environ["HYBRID_GRAMMAR"] = "0"
    hybrid.ollama(hybrid.FUSED_PROMPT.format(q="2 plus 2?"), grammar=hybrid.GRAMMAR_FUSED)
    check("HYBRID_GRAMMAR=0 is the escape hatch (grammar stripped)",
          "grammar" not in REQUESTS[-1], list(REQUESTS[-1]))
    os.environ.pop("HYBRID_GRAMMAR")

    # --- failure contract ----------------------------------------------------
    BEHAVIOR["fail_times"] = 1
    n0 = len(REQUESTS)
    ans, _ = hybrid.ollama("Question time.\nQuestion: 1 plus 1?")
    check("one 500 is retried and recovered", ans == "42" and len(REQUESTS) == n0 + 2,
          f"{len(REQUESTS) - n0} requests")

    BEHAVIOR["fail_times"] = 2
    try:
        hybrid.ollama("Question time.\nQuestion: 1 plus 1?")
        check("two failures raise BackendError", False, "no exception")
    except hybrid.BackendError as e:
        check("two failures raise BackendError", e.tier == "local", e.tier)

    # --- defaults ------------------------------------------------------------
    check("fusion is OFF by default, even on llamacpp (experimental opt-in)",
          not hybrid._fused(), os.environ.get("HYBRID_FUSE", "(unset)"))

    # --- split-server fast tier ----------------------------------------------
    srv_fast = HTTPServer(("127.0.0.1", 0), FakeLlamaServer)
    threading.Thread(target=srv_fast.serve_forever, daemon=True).start()
    main_port = srv.server_port
    hybrid.LLAMACPP_URL_FAST = f"http://127.0.0.1:{srv_fast.server_port}/completion"
    saved_fast = hybrid.LOCAL_MODEL_FAST
    hybrid.LOCAL_MODEL_FAST = "tiny-fake-3b"

    hybrid.ollama("vote prompt\nQuestion: capital of France?", model="tiny-fake-3b")
    check("a LOCAL_MODEL_FAST call routes to the fast server",
          REQUESTS[-1]["_port"] == srv_fast.server_port, REQUESTS[-1]["_port"])
    hybrid.ollama(hybrid.SETUP_PROMPT.format(q="2 plus 2?"), grammar=hybrid.GRAMMAR_SETUP)
    check("a transcription call (no model arg) stays on the primary server",
          REQUESTS[-1]["_port"] == main_port, REQUESTS[-1]["_port"])
    hybrid.LLAMACPP_URL_FAST = ""
    hybrid.ollama("vote prompt\nQuestion: capital of France?", model="tiny-fake-3b")
    check("without LLAMACPP_URL_FAST every call stays on the primary",
          REQUESTS[-1]["_port"] == main_port, REQUESTS[-1]["_port"])
    hybrid.LOCAL_MODEL_FAST = saved_fast
    srv_fast.shutdown()

    # --- grammar sanity pins -------------------------------------------------
    for name, g in (("setup", hybrid.GRAMMAR_SETUP), ("fused", hybrid.GRAMMAR_FUSED)):
        for cls in re.findall(r"\[(?:[^\]\\]|\\.)*\]", g):
            check(f"{name}: no \\- escape inside a char class (GBNF has none)",
                  r"\-" not in cls, cls)
            # a dash between two alphanumerics is a RANGE; any literal dash must be last
            literal_dashes = re.sub(r"[A-Za-z0-9]-[A-Za-z0-9]", "", cls[1:-1])
            check(f"{name}: a literal dash sits last in the class or is absent",
                  "-" not in literal_dashes or literal_dashes.endswith("-"), cls)
        check(f"{name}: grammar has a root rule", g.lstrip().startswith("root ::="), g[:20])
        check(f"{name}: every quoted newline is the two-char GBNF escape",
              "\\n" in g and "\n\"" not in g.replace('"\\n"', ""), "ok")

    srv.shutdown()
    print("-" * 72)
    if FAILS:
        print(f"FAIL  {len(FAILS)}/{COUNT[0]}")
        for name, detail in FAILS:
            print(f"   {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS  {COUNT[0]}/{COUNT[0]}")


if __name__ == "__main__":
    main()
