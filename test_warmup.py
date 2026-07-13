#!/usr/bin/env python3
"""Tests for hybrid.warmup() — the startup prefill primer (HYBRID_WARMUP).

Warmup must: fire one local-model forward pass per FIXED tier carrying that tier's
preamble, stay entirely local (never route/escalate/call the frontier), keep decode
trivial (num_predict=1), and NEVER raise — a cold/unreachable backend degrades to an
error marker so startup proceeds.
"""
import hybrid

FAILS = []
COUNT = [0]


def check(name, cond, detail=""):
    COUNT[0] += 1
    print(f"{'ok ' if cond else 'XX '} {name:<52} {str(detail)[:44]}")
    if not cond:
        FAILS.append((name, detail))


class RecordingModel:
    """Stands in for hybrid.ollama — records each call; optionally simulates a cold
    backend by raising BackendError (what a not-yet-loaded llama-server does)."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, prompt, num_predict=256, temperature=0.0, model=None,
                 grammar=None, family=None):
        self.calls.append({"prompt": prompt, "num_predict": num_predict, "model": model})
        if self.fail:
            raise hybrid.BackendError("local", "simulated cold backend")
        return ("4", 0.01)


def main():
    real_ollama = hybrid.ollama

    # ── 1. happy path: one forward pass per fixed tier, carrying its preamble ──
    rec = RecordingModel()
    hybrid.ollama = rec
    try:
        res = hybrid.warmup()
    finally:
        hybrid.ollama = real_ollama

    check("returns a dict summary", isinstance(res, dict))
    check("one local call per fixed tier",
          len(rec.calls) == len(hybrid._WARMUP_TIERS), f"{len(rec.calls)} calls")
    check("decode kept trivial (num_predict=1)",
          all(c["num_predict"] == 1 for c in rec.calls))
    check("carries the CONCISE preamble",
          any("concisely" in c["prompt"] for c in rec.calls))
    check("carries the FUSED derive preamble (EQN/CHECK)",
          any(("EQN:" in c["prompt"] and "CHECK:" in c["prompt"]) for c in rec.calls))
    check("summary reports a time per tier (no error markers)",
          all(isinstance(v, (int, float)) for v in res.values()), res)

    # ── 2. warmup is local-only: never touches the frontier/escalation ──
    real_escalate = getattr(hybrid, "escalate", None)
    esc_calls = [0]

    def spy_escalate(*a, **k):
        esc_calls[0] += 1
        return ("x", 0.0)

    rec2 = RecordingModel()
    hybrid.ollama = rec2
    if real_escalate is not None:
        hybrid.escalate = spy_escalate
    try:
        hybrid.warmup()
    finally:
        hybrid.ollama = real_ollama
        if real_escalate is not None:
            hybrid.escalate = real_escalate
    check("never escalates to the frontier during warmup", esc_calls[0] == 0)

    # ── 3. a cold/dead backend must NOT crash startup ──
    rec3 = RecordingModel(fail=True)
    hybrid.ollama = rec3
    raised = False
    res3 = {}
    try:
        res3 = hybrid.warmup()
    except Exception:
        raised = True
    finally:
        hybrid.ollama = real_ollama
    check("never raises when the backend is down", not raised)
    check("reports err markers when the backend is down",
          (not raised) and all(str(v).startswith("err") for v in res3.values()),
          res3 if not raised else "raised")

    print("-" * 72)
    if FAILS:
        print(f"FAIL  {len(FAILS)}/{COUNT[0]}")
        for name, detail in FAILS:
            print(f"   {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS  {COUNT[0]}/{COUNT[0]}")


if __name__ == "__main__":
    main()
