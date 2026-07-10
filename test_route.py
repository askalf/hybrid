#!/usr/bin/env python3
"""
Tests for the ROUTER itself — tier order, fallthrough, and the failure policy.

The oracle modules have their own suites; this one pins the plumbing that connects
them, which otherwise only gets exercised with a live model. Both backends are faked
(scripted answers, call logs), so the whole routing surface tests offline:

  1. TIER ORDER   — solve and the template transcriber short-circuit everything
                    (templates even out-rank the hard rule — an exact parse beats a
                    stray keyword); hard/open rules fire before the model-oracle
                    tiers; derive runs before plug-back; vote is last.
  2. GATING       — the quantitative gate sends digit-and-number-word queries through
                    the oracle tiers and spares factual queries the extra local calls.
  3. VERDICT->ROUTE — mismatch/false-check escalate; derived/checked serve locally
                    with the oracle's exact value.
  4. FAILURE POLICY — a dead backend degrades per HYBRID_ON_*_FAIL, and an error is
                    always an ERROR result, never an answer-shaped string.

    python test_route.py
"""
import os
from contextlib import contextmanager
import hybrid

_REAL_OLLAMA, _REAL_ESCALATE = hybrid.ollama, hybrid.escalate


@contextmanager
def envset(**kv):
    """Temporarily set env vars, restoring prior values (or unsetting) on exit."""
    saved = {k: os.environ.get(k) for k in kv}
    os.environ.update(kv)
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

FAILS = []
COUNT = [0]


def check(name, cond, detail=""):
    COUNT[0] += 1
    print(f"{'ok ' if cond else 'XX '} {name:<52} {detail[:44]}")
    if not cond:
        FAILS.append((name, detail))


class FakeModel:
    """Stands in for hybrid.ollama. The prompt template identifies the tier asking:
    FUSED_PROMPT carries both 'EQN:' and 'CHECK:', SETUP_PROMPT only 'EQN:',
    VERIFY_PROMPT only 'CHECK:', CONCISE neither."""

    def __init__(self, setup="no equations here", verify="no checks here",
                 fused="nothing derivable or checkable", concise=None, fail=False):
        self.setup, self.verify, self.fused = setup, verify, fused
        self.concise = list(concise or [])
        self.fail = fail
        self.kinds = []
        self.models = []
        self.grammars = []

    def __call__(self, prompt, num_predict=256, temperature=0.0, model=None, grammar=None):
        kind = ("fused" if ("EQN:" in prompt and "CHECK:" in prompt) else
                "setup" if "EQN:" in prompt else
                "verify" if "CHECK:" in prompt else "concise")
        self.kinds.append(kind)
        self.models.append(model)
        self.grammars.append(grammar)
        if self.fail:
            raise hybrid.BackendError("local", "connection refused (fake)")
        if kind == "fused":
            return self.fused, 0.0
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
# same problem, phrased OFF the template's rigid shape ("together total" != "cost") —
# it falls through the transcriber and exercises the model-derive tier
BAT_ODD = ("A bat and a ball together total $1.10. The bat costs $1.00 more than "
           "the ball. How much is the ball?")
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
    r = run(BAT, fm, ff)
    check("template answers the shaped word problem, no model",
          r["route"] == "SOLVED" and r["answer"] == "0.05"
          and r["why"] == "template: sum-diff" and fm.kinds == [] and ff.calls == 0,
          str(r["answer"]))

    fm, ff = FakeModel(), FakeFrontier()
    r = run("Each fuel tank holds 13.9 liters. How many liters do 73 tanks hold?", fm, ff)
    check("template out-ranks the hard rule ('liters' keyword)",
          r["route"] == "SOLVED" and r["answer"] == "1014.7" and ff.calls == 0,
          r["why"])

    fm, ff = FakeModel(), FakeFrontier()
    r = run("Emma has 5 brothers. Each brother has 3 sisters. "
            "How many sisters does Emma have?", fm, ff)
    check("template declines set-logic -> falls through to oracle tiers",
          r["route"] == "LOCAL" and fm.kinds and fm.kinds[0] == "setup",
          ",".join(fm.kinds))

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
    run(CHICKEN, fm, FakeFrontier())
    check("number-words (no digits) open the oracle tiers",
          fm.kinds[0] == "setup", ",".join(fm.kinds))

    # --- 3. verdict -> route ---------------------------------------------------
    fm, ff = FakeModel(setup=BAT_EQNS + "0.10"), FakeFrontier("It is $0.05.")
    r = run(BAT_ODD, fm, ff)
    check("derive mismatch -> hard escalate",
          r["route"] == "ESCALATE" and r["why"].startswith("setup derives")
          and ff.calls == 1, r["why"])

    fm, ff = FakeModel(setup=BAT_EQNS + "0.05"), FakeFrontier()
    r = run(BAT_ODD, fm, ff)
    check("derive match serves the exact derived value",
          r["route"] == "LOCAL" and r["answer"] == "0.05" and ff.calls == 0, r["why"])

    fm, ff = (FakeModel(verify="The total is 966.5. CALC: 23.7 * 41 - 10 = 966.5"),
              FakeFrontier("961.7"))
    r = run("Each crate weighs 23.7 kg. What do 41 crates weigh after removing 10 kg "
            "of packaging?", fm, ff)
    check("false stated arithmetic -> hard escalate",
          r["route"] == "ESCALATE" and "local math wrong" in r["why"] and ff.calls == 1,
          r["why"])

    fm, ff = (FakeModel(verify="ANSWER: $87.50\nCHECK: 7 * 12.50 = 87.50"),
              FakeFrontier())
    r = run("A store sells notebooks at $12.50 each and pens at $2.00 each. "
            "How much do 7 notebooks cost?", fm, ff)
    check("checks-hold serves locally",
          r["route"] == "LOCAL" and "87.5" in r["answer"] and ff.calls == 0, r["why"])

    fm, ff = FakeModel(concise=["42", "42", "42"]), FakeFrontier()
    r = run("What is the answer to everything?", fm, ff)
    check("unanimous vote stays local",
          r["route"] == "LOCAL" and "self-consistent" in r["why"] and ff.calls == 0,
          r["why"])

    # --- 3.5 split model: fast model on the vote tier, never on transcription ---
    saved_fast = hybrid.LOCAL_MODEL_FAST
    hybrid.LOCAL_MODEL_FAST = "tiny-fake"
    try:
        fm, ff = FakeModel(concise=["Tokyo", "Tokyo", "Tokyo"]), FakeFrontier()
        r = run("What is the capital of Japan?", fm, ff)
        check("vote tier uses LOCAL_MODEL_FAST",
              r["route"] == "LOCAL" and fm.models == ["tiny-fake"] * 3
              and r["backend"] == "tiny-fake", ",".join(map(str, fm.models)))

        fm, ff = FakeModel(concise=["2/3", "2/3", "2/3"]), FakeFrontier()
        run(CHICKEN, fm, ff)
        check("transcription tiers stay on LOCAL_MODEL (never the fast model)",
              fm.kinds[:2] == ["setup", "verify"] and fm.models[0] is None
              and fm.models[1] is None, ",".join(map(str, fm.models[:2])))
    finally:
        hybrid.LOCAL_MODEL_FAST = saved_fast

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

    # --- 6. fused transcription tier (HYBRID_FUSE=1; default on llamacpp) ------
    FUSE = {"HYBRID_FUSE": "1"}
    fused_ok = BAT_EQNS + "0.05\nCHECK: 0.05 + 1.05 = 1.10"
    fm, ff = FakeModel(fused=fused_ok), FakeFrontier()
    r = run(BAT_ODD, fm, ff, env=FUSE)
    check("fused: derive serves in ONE call",
          r["route"] == "LOCAL" and "fused" in r["why"] and fm.kinds == ["fused"]
          and r["answer"] == "0.05" and ff.calls == 0, ",".join(fm.kinds))
    check("fused: the transcription call carries the fused grammar",
          fm.grammars == [hybrid.GRAMMAR_FUSED], str(fm.grammars)[:44])

    fm, ff = FakeModel(fused=BAT_EQNS + "0.10\nCHECK: 0.10 + 1.10 = 1.10"), FakeFrontier("$0.05.")
    r = run(BAT_ODD, fm, ff, env=FUSE)
    check("fused: derive mismatch -> hard escalate",
          r["route"] == "ESCALATE" and r["why"].startswith("setup derives") and ff.calls == 1,
          r["why"])

    # a sloppy false CHECK next to a CONFIRMED derive must not escalate (the
    # self-referential brick, live): derive verdict out-ranks plug-back.
    brick = ("EQN: b = 1 + 0.5 * b\nANSWER: b = 2\nCHECK: 2 = 1.5")
    fm, ff = FakeModel(fused=brick), FakeFrontier()
    r = run("A brick weighs 1 kg plus half a brick. What does the brick weigh in kg? "
            "Give a number.", fm, ff, env=FUSE)
    check("fused: derive verdict out-ranks a sloppy false CHECK",
          r["route"] == "LOCAL" and r["answer"] == "2" and ff.calls == 0, r["why"])

    # nothing derivable, but a FALSE check -> hard escalate (plug-back still armed)
    fm, ff = FakeModel(fused="ANSWER: total = 85\nCHECK: 7 * 12.50 = 85"), FakeFrontier("$87.50")
    r = run("A pen costs $12.50. What do 7 pens cost, given a 0-dollar discount?", fm, ff, env=FUSE)
    check("fused: false CHECK (no derive) -> hard escalate",
          r["route"] == "ESCALATE" and "local math wrong" in r["why"] and ff.calls == 1,
          r["why"])

    # nothing derivable, checks hold -> served off the same single call
    fm, ff = FakeModel(fused="ANSWER: total = 87.5\nCHECK: 7 * 12.50 = 87.5"), FakeFrontier()
    r = run("A pen costs $12.50. What do 7 pens cost, given a 0-dollar discount?", fm, ff, env=FUSE)
    check("fused: checks hold -> LOCAL in one call",
          r["route"] == "LOCAL" and "fused" in r["why"] and fm.kinds == ["fused"]
          and ff.calls == 0, r["why"])

    # nothing derivable NOR checkable -> falls through to the vote (1 fused + 3 votes)
    fm, ff = FakeModel(fused="I am not sure.", concise=["42", "42", "42"]), FakeFrontier()
    r = run("A widget batch has 3 dozen plus 6 widgets. How many is that?", fm, ff, env=FUSE)
    check("fused: nothing usable falls through to the vote",
          r["route"] == "LOCAL" and fm.kinds == ["fused", "concise", "concise", "concise"],
          ",".join(fm.kinds))

    # default (ollama transport): fusion OFF -> the two-call flow, setup first
    fm, ff = FakeModel(), FakeFrontier()
    r = run(BAT_ODD, fm, ff, env={"HYBRID_FUSE": "0"})
    check("HYBRID_FUSE=0 keeps the two-call flow (setup then verify)",
          fm.kinds[:2] == ["setup", "verify"], ",".join(fm.kinds))
    check("two-call setup carries the setup grammar",
          fm.grammars and fm.grammars[0] == hybrid.GRAMMAR_SETUP, str(fm.grammars[0])[:40])

    # --- 7. load shedding (concurrency cap + latency budget) -------------------
    import threading

    # the gate primitive, unit-tested directly (no slot left dangling on shed)
    assert hybrid.model_inflight() == 0
    with envset(HYBRID_MODEL_MAX_INFLIGHT="0"):
        r0 = hybrid._enter_model_or_shed(0.0)
    check("gate off -> takes a slot (returns None)",
          r0 is None and hybrid.model_inflight() == 1, str(r0))
    hybrid._leave_model()
    check("_leave_model releases the slot", hybrid.model_inflight() == 0, "")

    with envset(HYBRID_LATENCY_BUDGET_MS="1000", HYBRID_MODEL_TIER_MS="8000"):
        r1 = hybrid._enter_model_or_shed(0.0)
    check("budget: one tier (8s) over a 1s budget -> shed, no slot",
          isinstance(r1, str) and "latency budget" in r1 and hybrid.model_inflight() == 0, r1)

    with envset(HYBRID_LATENCY_BUDGET_MS="20000", HYBRID_MODEL_TIER_MS="8000"):
        r2 = hybrid._enter_model_or_shed(0.0)
    check("budget: one tier under a 20s budget -> proceed", r2 is None, str(r2))
    hybrid._leave_model()

    # cap: with the gauge already at 1, a cap of 1 sheds; the deterministic tiers
    # (SOLVED/template) must NEVER shed even when the box is 'full'.
    hybrid._MODEL_INFLIGHT = 1  # simulate one model request in flight
    try:
        with envset(HYBRID_MODEL_MAX_INFLIGHT="1"):
            fm, ff = FakeModel(), FakeFrontier()
            r = run("What is 47 times 19?", fm, ff, env={"HYBRID_MODEL_MAX_INFLIGHT": "1"})
            check("SOLVED never sheds (deterministic, free) even at the cap",
                  r["route"] == "SOLVED" and ff.calls == 0, r["route"])
            fm, ff = FakeModel(concise=["x"]), FakeFrontier("frontier")
            r = run("What is the capital of Japan?", fm, ff, env={"HYBRID_MODEL_MAX_INFLIGHT": "1"})
            check("a model-path query sheds at the cap -> frontier",
                  r["route"] == "ESCALATE" and "load shed" in r["why"] and ff.calls == 1
                  and "concise" not in fm.kinds, r["why"])
    finally:
        hybrid._MODEL_INFLIGHT = 0

    # threaded integration: one request genuinely HOLDS a model slot while a second,
    # concurrent model-path request arrives and sheds — the real server behavior.
    holding = threading.Event()
    release = threading.Event()

    class HoldingModel:
        """Blocks inside the vote's model call until released, holding the slot."""
        def __call__(self, prompt, num_predict=256, temperature=0.0, model=None, grammar=None):
            holding.set()
            release.wait(10)
            return "held", 0.0

    hybrid.ollama, hybrid.escalate = HoldingModel(), FakeFrontier("frontier")
    os.environ["HYBRID_MODEL_MAX_INFLIGHT"] = "1"
    try:
        t = threading.Thread(target=lambda: hybrid.route("What is the capital of France?"))
        t.start()
        got = holding.wait(10)  # wait until the first request is inside its model slot
        inflight_during = hybrid.model_inflight()
        r = hybrid.route("What is the capital of Spain?")  # arrives while the slot is held
        release.set(); t.join(10)
        check("a slot is held across the in-flight request's model call",
              got and inflight_during == 1, f"held={got} inflight={inflight_during}")
        check("concurrent model-path request sheds while the slot is held",
              r["route"] == "ESCALATE" and "load shed" in r["why"], r["why"])
        check("the slot is released after the in-flight request finishes",
              hybrid.model_inflight() == 0, str(hybrid.model_inflight()))
    finally:
        release.set()
        os.environ.pop("HYBRID_MODEL_MAX_INFLIGHT", None)
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
