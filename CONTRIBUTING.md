# Contributing to hybrid

Thanks for your interest in improving **hybrid** — a local-first LLM router that
answers the easy majority on a small local model and escalates only the hard
queries to any OpenAI-compatible frontier endpoint, backed by a deterministic
verifier. It's ~160 lines of stdlib-only Python across six small readable
modules. Part of [Own Your Stack](https://github.com/askalf).

## Ground rules

- Be respectful. This project follows our [Code of Conduct](CODE_OF_CONDUCT.md).
- Found a security issue? **Do not open a public issue** — follow
  [SECURITY.md](SECURITY.md) to report it privately.

## Development setup

hybrid is a Python project with **no runtime dependencies** — standard library
only. You need Python **3.10, 3.11, 3.12, or 3.13** (the versions CI tests
against). The distribution name is `hybrid-router`; the modules import as
`hybrid`, `solver`, `templates`, `verify`, `equations`, and `server`.

```bash
git clone https://github.com/askalf/hybrid.git
cd hybrid
python -m venv .venv && . .venv/bin/activate   # optional
python -m pip install -e .                      # editable install (setuptools)
```

The test suite is plain-stdlib and runs offline — no model, no network, no
dependencies. Run each module directly (the counts match the CI job names):

```bash
python -m py_compile hybrid.py solver.py verify.py equations.py server.py \
  bench_offline.py bench_router.py measure_routing.py
python test_solver.py      # solver oracle
python test_verify.py      # plug-back verifier
python test_equations.py   # setup re-derivation
python test_route.py       # router plumbing + failure policy
python test_server.py      # server surface
```

## Making a change

1. Branch off `main`.
2. Keep the change focused — one concern per PR. The project's whole premise is
   staying small and readable, so resist adding dependencies or line count
   without a strong reason.
3. Add or update tests for any behavior change — especially to the router's
   escalation decision or the deterministic verifier, which are the product.
4. Run the test modules above locally before pushing.
5. Open a pull request against `main`.

## What CI requires

Every PR must pass these checks to merge:

- `test (3.10)`, `test (3.11)`, `test (3.12)`, `test (3.13)` — compile check +
  the full offline suite on each Python version
- `package` — builds the sdist + wheel, installs the wheel, and runs a
  console-script smoke (`hybrid` / `hybrid-server`) away from the checkout
- `analyze (python)` — **CodeQL** static analysis

OpenSSF Scorecard also runs on the repo.

## Conventions

- GitHub Actions are **pinned to a commit SHA**, never a mutable tag. New or
  updated workflow steps must keep this.
- Commit messages: short imperative subject, with a wrapped body explaining the
  *why* when it isn't obvious.

## Releases

Releases are tag-driven: bump `__version__` in `hybrid.py`, then create a GitHub
release tagged `vX.Y.Z` (the tag must match `hybrid.__version__`). `publish.yml`
builds and publishes `hybrid-router` to PyPI via OIDC trusted publishing (no
tokens). A normal PR needs no release steps.
