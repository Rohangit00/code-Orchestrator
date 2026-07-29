# Fugu — Coding-Oriented LLM Orchestrator

Small **planner** model that learns to coordinate larger coding **workers**
(Qwen / Gemma / Ornith on vLLM) for SWE-bench-style repair, formulated as an MDP.

## Readiness (honest)

| Mode | Ready? |
|------|--------|
| Unit tests / local logic | Yes |
| Controlled **smoke transfer** to an H200 VM | Yes, after clone + install |
| LiveCodeBench Python collect/train | **Yes** — standalone workspace + one pytest image |
| **Real** third-party SWE-bench collection | **Partial** — docker sandbox exists; not official harness; multi-repo images hard |
| Full train/eval on H200s | Only after smoke stages pass (see [docs/VM_RUNBOOK.md](docs/VM_RUNBOOK.md)) |

Source of truth for open work: `task.md`, `implementation_issues.md`.

## Quick start (dev machine or VM)

```bash
# Python 3.11+ required
python3.11 -m venv .venv
source .venv/bin/activate

# Install package so `import fugu` and CLIs work without PYTHONPATH
pip install -U pip
pip install -e ".[dev]"

pytest tests/ -v
```

Optional: `pyproject.toml` sets `pythonpath = ["src"]` for pytest even without install,
but **VM and CLI use must use `pip install -e ".[dev]"`**.

## Configuration

- Defaults: `configs/default.yaml`
- Overrides: env vars with `FUGU_` prefix (nested with `__`), e.g.  
  `FUGU_WORKER__QWEN_URL=http://localhost:8001/v1`

Worker endpoints (canonical):

| Worker | URL |
|--------|-----|
| Qwen | `http://localhost:8001/v1` |
| Gemma | `http://localhost:8002/v1` |
| Ornith | `http://localhost:8003/v1` |

## Safety — do not skip

```yaml
# configs/default.yaml (env section)
isolation_mode: host          # use "docker" for untrusted remotes
allow_host_execution: false   # refuse untrusted remote test runs on host
docker_image: "python:3.11-slim"
```

- **Untrusted remotes:** set `isolation_mode: docker` (or `FUGU_ENV__ISOLATION_MODE=docker`).
  Tests/compile run in a container; this is **not** the official SWE-bench harness.
- Never set `allow_host_execution=true` for third-party GitHub clones on a shared VM.
- See [docs/VM_START.md](docs/VM_START.md) for collect smoke after docker is enabled.

## CLI (after install)

```bash
fugu-collect --help
fugu-train --help
fugu-eval --help
```

Datasets for collect/eval:

- **Primary:** LiveCodeBench Python — `livecodebench-train` / `-val` / `-test` / `livecodebench`
  (standalone workspace; **one** Docker image with pytest is enough)
- **Optional:** SWE-bench (`swebench-lite`, `swebench-full`, `swebench-verified`) for
  repo-backed experiments (harder envs)

## Docs

| Doc | Purpose |
|-----|---------|
| [docs/VM_START.md](docs/VM_START.md) | **Start here on the VM** — LCB path, pytest image, collect `-n 1`, train |
| [docs/VM_RUNBOOK.md](docs/VM_RUNBOOK.md) | Full H200 checklist (LCB primary; SWE optional) |
| `explanation.md` | Design / MDP motivation |
| `implementation_plan.md` | Full architecture plan |
| `implementation_issues.md` | Issue tracker + fix status |
| `implementation_solutions.md` | Solutions + checklist |
| `project_context.md` | Specs and current semantics |
| `task.md` | Checklist |

## License / status

Research prototype. Not production-hardened.
