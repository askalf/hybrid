#!/usr/bin/env python3
"""
Tests for the ROUTER itself — tier order, fallthrough, and the failure policy.

The oracle modules have their own suites; this one pins the plumbing that connects
them, which otherwise only gets exercised with a live model. Both backends are faked
(scripted answers, call logs), so the whole routing surface tests offline:

  1. TIER ORDER   — solve short-circuits everything; hard/open rules fire before the
                    oracle tiers; derive runs before plug-back; vote is last.
  2. GATING       — the quantitative gate sends digit-and-number-word queries through
                    the oracle tiers and spares factual queries the extra local calls.
  3. VERDICT->ROUTE — mismatch/false-check escalate; derived/checked serve locally
                    with the oracle's exact value.
  4. FAILURE POLICY — a dead backend degrades per HYBRID_ON_*_FAIL, and an error is
                    always an ERROR result, never an answer-shaped string.

    python test_route.py
"""
import os
import hybrid

_REAL_OLLAMA, _REAL_ESCALATE = hybrid.ollama, hybrid.escalate

FAILS = []
COUNT = [0]


def check(name, cond, detail=""):
    COUNT[0] += 1
    print(f"{'ok ' if cond else 'XX '} {name:<52} {detail[:44]}")
    if not cond:
        FAILS.append((name, detail))


class FakeModel:
    """Stands in for hybrid.ollama. The prompt template identifies the tier asking:
    SETUP_PROMPT carries 'EQN:', VERIFY_PROMPT carries 'CHECK:', CONCISE neither."""

    def __init__(self, setup="no equations here", verify="no checks here",
                 concise=None, fail=False):
        self.setup, self.verify = setup, verify
        self.concise = list(concise or [])
        self.fail = fail
        self.kinds = []

    def __call__(self, prompt, num_predict=256, temperature=0.0):
        kind = ("setup" if "EQN:" in prompt else
                "verify" if "CHECK:" in prompt else "concise")
        self.kinds.append(kind)
        if self.fail:
            raise hybrid.BackendError("local", "connection refused (fake)")
        if kind == "setup":
            return self.setup, 0.0
        if kind == "verify":
            return self.verify, 0.0
        return (self.concise.pop(0) if self.concise else "ok"), 0.0


class FakeFrontier:
    def __init__(self, answer="frontier-answer", fail=False):
        self.answer, self.fail, self.calls = answer, fail, 0

    def __call__(self, query):
        self.calls += 1
        if self.fail:
            raise hybrid.BackendError("frontier", "unreachable (fake)")
        return self.answer, 0.0


def run(query, fm, ff, env=None):
    saved = {k: os.environ.get(k) for k in (env or {})}
    os.environ.update(env or {})
    hybrid.ollama, hybrid.escalate = fm, ff
    try:
        return hybrid.route(query)
    finally:
        hybrid.ollama, hybrid.escalate = _REAL_OLLAMA, _REAL_ESCALATE
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


BAT = ("A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. "
       "How much is the ball?")
BAT_EQNS = "EQN: bat + ball = 1.10\nEQN: bat = ball + 1.00\nANSWER: ball = "
CHICKEN = ("If a chicken and a half lays an egg and a half in a day and a half, "
           "how many eggs does one chicken lay in one day?")


def main():
    # --- 1. tier order ---------------------------------------------------------
    fm, ff = FakeModel(), FakeFrontier()
    r = run("What is 47 times 19?", fm, ff)
    check("solve short-circuits (no model touched)",
          r["route"] == "SOLVED" and r["answer"] == "893"
          and fm.kinds == [] and ff.calls == 0, str(r["answer"]))

    fm, ff = FakeModel(), FakeFrontier()
    r = run("Prove that the square root of 2 is irrational.", fm, ff)
    check("hard rule escalates before any local call",
          r["route"] == "ESCALATE" and ff.calls == 1 and fm.kinds == [], r["why"])

    fm, ff = FakeModel(concise=["Certainly - please send the file."]), FakeFrontier()
    r = run("Rewrite 'hey can u send me that file' more formally.", fm, ff)
    check("open-ended stays local via concise only",
          r["route"] == "LOCAL" and fm.kinds == ["concise"] and ff.calls == 0, r["why"])

    # --- 2. quantitative gate --------------------------------------------------
    fm, ff = FakeModel(concise=["Tokyo", "Tokyo", "Tokyo"]), FakeFrontier()
    r = run("What is the capital of Japan?", fm, ff)
    check("factual query skips the oracle tiers",
          r["route"] == "LOCAL" and "setup" not in fm.kinds and "verify" not in fm.kinds,
          ",".join(fm.kinds))

    fm = FakeModel(concise=["2/3", "2/3", "2/3"])   # setup+verify defaults -> nothing usable
    r = run(CHICKEN, fm, FakeFrontier())
    check("number-words (no digits) open the oracle tiers",
          fm.kinds[0] == "setup", ",".join(fm.kinds))

    # --- 3. verdict -> route ---------------------------------------------------
    fm, ff = FakeModel(setup=BAT_EQNS + "0.10"), FakeFrontier("It is $0.05.")
    r = run(BAT, fm, ff)
    check("derive mismatch -> hard escalate",
          r["route"] == "ESCALATE" and r["why"].startswith("setup derives")
          and ff.calls == 1, r["why"])

    fm, ff = FakeModel(setup=BAT_EQNS + "0.05"), FakeFrontier()
    r = run(BAT, fm, ff)
    check("derive match serves the exact derived value",
          r["route"] == "LOCAL" and r["answer"] == "0.05" and ff.calls == 0, r["why"])

    fm, ff = (FakeModel(verify="The total is 961.7. CALC: 23.7 * 41 = 961.7"),
              FakeFrontier("971.7"))
    r = run("Each crate weighs 23.7 kg. What do 41 crates weigh?", fm, ff)
    check("false stated arithmetic -> hard escalate",
          r["route"] == "ESCALATE" and "local math wrong" in r["why"] and ff.calls == 1,
          r["why"])

    fm, ff = (FakeModel(verify="ANSWER: $87.50\nCHECK: 7 * 12.50 = 87.50"),
              FakeFrontier())
    r = run("A store sells notebooks at $12.50 each. How much do 7 notebooks cost?",
            fm, ff)
    check("checks-hold serves locally",
          r["route"] == "LOCAL" and "87.5" in r["answer"] and ff.calls == 0, r["why"])

    fm, ff = FakeModel(concise=["42", "42", "42"]), FakeFrontier()
    r = run("What is the answer to everything?", fm, ff)
    check("unanimous vote stays local",
          r["route"] == "LOCAL" and "self-consistent" in r["why"] and ff.calls == 0,
          r["why"])

    fm, ff = FakeModel(concise=["42", "41", "40"]), FakeFrontier("The answer is 42.")
    r = run("What is the answer to everything?", fm, ff)
    check("split vote escalates",
          r["route"] == "ESCALATE" and "uncertain" in r["why"] and ff.calls == 1,
          r["why"])

    # --- 4. failure policy -----------------------------------------------------
    fm, ff = FakeModel(fail=True), FakeFrontier("Tokyo.")
    r = run("What is the capital of Japan?", fm, ff)
    check("local down -> escalate (default policy)",
          r["route"] == "ESCALATE" and "local backend down" in r["why"] and ff.calls == 1,
          r["why"])

    fm, ff = FakeModel(fail=True), FakeFrontier()
    r = run("What is the capital of Japan?", fm, ff,
            env={"HYBRID_ON_LOCAL_FAIL": "error"})
    check("local down + policy=error -> ERROR result",
          r["route"] == "ERROR" and r.get("error") is True and ff.calls == 0, r["why"])

    fm, ff = FakeModel(), FakeFrontier(fail=True)
    r = run("Prove that the square root of 2 is irrational.", fm, ff)
    check("frontier down -> honest ERROR (default policy)",
          r["route"] == "ERROR" and "frontier" in r["why"], r["why"])

    fm, ff = FakeModel(concise=["Sketch: assume p/q in lowest terms..."]), FakeFrontier(fail=True)
    r = run("Prove that the square root of 2 is irrational.", fm, ff,
            env={"HYBRID_ON_FRONTIER_FAIL": "local"})
    check("frontier down + policy=local -> labelled DEGRADED",
          r["route"] == "LOCAL" and "DEGRADED" in r["why"], r["why"])

    fm, ff = FakeModel(fail=True), FakeFrontier(fail=True)
    r = run("What is the capital of Japan?", fm, ff)
    check("both backends down -> ERROR, never an answer-shaped string",
          r["route"] == "ERROR" and r.get("error") is True, r["why"])

    print("-" * 72)
    if FAILS:
        print(f"FAIL  {len(FAILS)}/{COUNT[0]}")
        for name, detail in FAILS:
            print(f"   {name}: {detail}")
        raise SystemExit(1)
    print(f"PASS  {COUNT[0]}/{COUNT[0]}")


if __name__ == "__main__":
    main()
