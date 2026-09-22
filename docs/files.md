# hybrid files

Back to the [README](../README.md).

## Files

- `hybrid.py` — router + dispatch + `--demo`
- `solver.py` — deterministic arithmetic + exact unit/percentage/multiple conversion (the SOLVED tier)
- `templates.py` — deterministic word-problem transcriber: five rigid shapes parsed and
  solved in closed form over `Fraction`s, no model; ruthlessly conservative
- `equations.py` — setup re-derivation: solve the model's transcribed equation system exactly
  (linear systems, Gaussian elimination over `Fraction`s); conservative
- `verify.py` — verify-the-local-answer: re-derive the model's plugged-in checks exactly
- `test_*.py` — 365 tests (oracles + transcriber + router plumbing + failure policy +
  server surface + cache + llamacpp transport + the Anthropic door + token accounting +
  warmup); all offline, no model needed
- `bench_offline.py` — what the solver buys versus a no-solver router (no model needed)
- `bench_router.py` — full-router benchmark: on-box rate, on-box safety, catches (frontier stubbed)
- `measure_routing.py` — router economics: prices every query's frontier cost to show real $ saved
- `server.py` — OpenAI-compatible (and Anthropic-compatible) front end: SSE streaming,
  JSONL decision log, body caps, optional bearer auth
- `pyproject.toml` / `Dockerfile` / `deploy/` — pip/pipx packaging (console commands
  `hybrid` + `hybrid-server`), container image, compose + systemd examples
