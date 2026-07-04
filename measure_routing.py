#!/usr/bin/env python3
"""
Router economics — what local-first routing actually buys, in dollars.

On-box *query share* is not the same as *dollar share*. The queries that escalate
are the token-heavy ones (a proof, a code-gen), so the money saved is less than the
on-box rate suggests. This quantifies that honestly: route the bench_router labeled
set, and price EVERY query's frontier cost —
  - escalated queries -> the real cost we paid (captured during routing)
  - on-box queries    -> a counterfactual frontier call: what it WOULD have cost had
                         we not answered on-box. That counterfactual is the $ saved.

Tokens come from the frontier's OpenAI-compatible `usage` block; priced at the rate
you set for your frontier model.

    FRONTIER_API_KEY=... [FRONTIER_URL=...] [FRONTIER_MODEL=...] \
    [PRICE_IN_PER_M=3.0] [PRICE_OUT_PER_M=15.0] python measure_routing.py

Defaults to claude-sonnet-4-6 list pricing ($3 / $15 per 1M). Pricing is independent
of FRONTIER_MODEL — set PRICE_IN_PER_M / PRICE_OUT_PER_M to match whatever you escalate to.
"""
import sys, os, time, json, urllib.request
import hybrid
from bench_router import CASES

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows cp1252 chokes on non-ASCII
except Exception:
    pass  # best-effort console tweak; stdout may be replaced/unreconfigurable (tests, pipes)

PRICE_IN = float(os.environ.get("PRICE_IN_PER_M", "3.0")) / 1_000_000
PRICE_OUT = float(os.environ.get("PRICE_OUT_PER_M", "15.0")) / 1_000_000


def metered_escalate(query):
    """Same call as hybrid.escalate, but also returns the frontier's token usage.
    Returns (answer, latency_s, {input_tokens, output_tokens, ok})."""
    body = json.dumps({
        "model": hybrid.FRONTIER_MODEL,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 512,
    }).encode()
    headers = {"content-type": "application/json"}
    if hybrid.FRONTIER_KEY:
        headers["authorization"] = "Bearer " + hybrid.FRONTIER_KEY
    req = urllib.request.Request(hybrid.FRONTIER_URL, data=body, headers=headers)
    t0 = time.time()
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=120).read())
        text = r["choices"][0]["message"]["content"].strip()
        u = r.get("usage", {}) or {}
        return text, time.time() - t0, {
            "input_tokens": int(u.get("prompt_tokens", 0)),
            "output_tokens": int(u.get("completion_tokens", 0)),
            "ok": True,
        }
    except Exception:
        return "", time.time() - t0, {"input_tokens": 0, "output_tokens": 0, "ok": False}


def cost(u):
    return u["input_tokens"] * PRICE_IN + u["output_tokens"] * PRICE_OUT


# Record the real frontier usage for queries the router escalates.
_usage_during_route = {}


def _escalate_recording(query):
    text, dt, usage = metered_escalate(query)
    _usage_during_route[query] = usage
    return text, dt


hybrid.escalate = _escalate_recording


def main():
    if not hybrid.FRONTIER_KEY:
        print("Set FRONTIER_API_KEY (and FRONTIER_URL / FRONTIER_MODEL) to price escalations.")
        return
    print(f"{'#':>2}  {'CAT':<8} {'ROUTE':<8} {'lat':>6}  {'in':>5} {'out':>5} {'front$':>8}  query")
    print("-" * 100)
    onbox_lat = saved = spent = esc_lat = 0.0
    onbox_count = esc_count = esc_in = esc_out = missing = 0
    for i, (q, truth, cat) in enumerate(CASES, 1):
        r = hybrid.route(q)
        route = r["route"]
        lat = r["router_s"] + r["answer_s"]
        if route in ("SOLVED", "LOCAL"):
            _, _, usage = metered_escalate(q)  # counterfactual frontier cost
            onbox_count += 1
            onbox_lat += lat
            saved += cost(usage)
        else:
            usage = _usage_during_route.get(q, {"input_tokens": 0, "output_tokens": 0, "ok": False})
            esc_count += 1
            esc_lat += lat
            spent += cost(usage)
            esc_in += usage["input_tokens"]
            esc_out += usage["output_tokens"]
        if not usage.get("ok", True):
            missing += 1
        print(f"{i:>2}  {cat:<8} {route:<8} {lat:>5.1f}s "
              f"{usage['input_tokens']:>5} {usage['output_tokens']:>5} "
              f"${cost(usage):>7.5f}  {q[:42]}")

    n = len(CASES)
    all_frontier = saved + spent
    print("-" * 100)
    print(f"ON-BOX:       {onbox_count}/{n} ({100*onbox_count//n}%)  "
          f"avg {onbox_lat/max(onbox_count,1):.1f}s   frontier calls avoided")
    print(f"ESCALATED:    {esc_count}/{n} ({100*esc_count//n}%)  "
          f"avg {esc_lat/max(esc_count,1):.1f}s   {esc_in} in + {esc_out} out tok  = ${spent:.5f} spent")
    print()
    print(f"$ SAVED:        ${saved:.5f}   (counterfactual frontier cost of the {onbox_count} on-box answers)")
    print(f"$ SPENT:        ${spent:.5f}   (real cost of the {esc_count} escalations)")
    print(f"$ ALL-FRONTIER: ${all_frontier:.5f}   (every query to the frontier)")
    if all_frontier > 0:
        print(f"REDUCTION:    {100*saved/all_frontier:.0f}% of frontier spend avoided by answering on-box")
        print(f"              (on-box query share is {100*onbox_count//n}% — dollars saved is lower because "
              f"the few escalations are the token-heavy ones)")
    print()
    print(f"Per-1000 queries (this mix): ~${1000*all_frontier/n:.2f} all-frontier "
          f"vs ~${1000*spent/n:.2f} hybrid  ->  ~${1000*saved/n:.2f} saved")
    if missing:
        print(f"\nNOTE: {missing} frontier measurement(s) failed -> counted as $0; true savings slightly higher.")


if __name__ == "__main__":
    main()
