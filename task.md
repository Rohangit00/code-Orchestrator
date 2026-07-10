# Task List — Fugu Orchestrator

Status snapshot after issues **1–9**, **11 light**, **12 gate**, **14 basic**,
**15–17** (see `implementation_issues.md` / `implementation_solutions.md`).

## Deployment readiness

| Target | Ready? |
|--------|--------|
| Unit tests / logic | Yes |
| Controlled smoke transfer to H200 VM | Yes (after this commit + [docs/VM_RUNBOOK.md](docs/VM_RUNBOOK.md)) |
| Real SWE-bench collection on VM | **No** — no docker executor; host execution unsafe for untrusted remotes |
| Full train/eval at scale | **No** — needs smoke stages + isolation first |

Default config: `allow_host_execution: false`.

## Phase 1: Foundation (Core + Mock Pipeline)

### Project Setup
- [x] `pyproject.toml` with all dependencies
- [x] Package structure (`src/fugu/`)
- [x] Configuration system (`config.py` + `default.yaml`)
- [x] Python 3.11+ `.venv` for local validation
- [ ] `README.md`

### Core Types
- [x] `core/actions.py` — PlannerAction enum
- [x] `core/state.py` — PlannerState, Transition, Metadata, HistoryEntry
- [x] `core/reward.py` — RewardCalculator

### Worker Interface
- [x] `workers/base.py` — BaseWorker ABC, WorkerResponse
- [x] `workers/vllm.py` — VLLMWorker (OpenAI-compatible HTTP client)
- [x] `workers/mock.py` — MockWorker for testing
- [x] `workers/pool.py` — WorkerPool dispatch

### Repository Management
- [x] `repo/manager.py` — RepoManager (clone/patch/reset/cleanup)
- [x] `repo/context.py` — RepoContext (file tree, relevant files)

### Test Execution
- [x] `execution/runner.py` — TestRunner (authoritative `test_command`, robust parse, isolation gate)

### Coding Environment
- [x] `env/coding_env.py` — CodingEnvironment (auto-terminate on solve; env-owned transitions)

### Dataset Adapters
- [x] `datasets/base.py` — CodingTask, BaseDataset
- [x] `datasets/swebench.py` — SWE-bench / SWE-bench Lite adapter (**collection/eval path**)
- [x] `datasets/humaneval.py` — HumanEval adapter (inspection only; not runnable in env)
- [x] `datasets/mbpp.py` — MBPP adapter (inspection only; not runnable in env)

### Trajectory Generation
- [x] `trajectory/strategies.py` — Fixed strategies (`RETRY` = full re-call; no double CALL)
- [x] `trajectory/collector.py` — TrajectoryCollector (`env.transitions[-1]`, STOP via env, try/finally cleanup)

### Replay Buffer
- [x] `buffer/replay_buffer.py` — Fixed-size compressed buffer

### Planner Model
- [x] `planner/model.py` — PlannerModel with QLoRA
- [x] `planner/tokenizer.py` — State serialization
- [x] `planner/prompts.py` — Shared train/infer prompt builder

### Training
- [x] `training/trainer.py` — PlannerTrainer (SFT; light quality filter on buffer)
- [x] `training/data.py` — TransitionDataset
- [x] `training/filter.py` — Episode return / solve filtering

### CLI
- [x] `cli/collect.py` — fugu-collect (SWE-bench only; isolation + cleanup wired)
- [x] `cli/train.py` — fugu-train
- [x] `cli/eval.py` — fugu-eval (SWE-bench only)

### Tests
- [x] `tests/conftest.py` — shared fixtures
- [x] `tests/test_actions.py`
- [x] `tests/test_state.py`
- [x] `tests/test_reward.py`
- [x] `tests/test_workers.py`
- [x] `tests/test_env.py`
- [x] `tests/test_buffer.py`
- [x] `tests/test_runner_parse.py`
- [x] `tests/test_strategies.py`
- [x] `tests/test_cli_datasets.py`
- [x] `tests/test_prompts.py`
- [x] `tests/test_filter.py`
- [ ] `tests/test_datasets.py` — optional SWE-bench load smoke (needs HF network)

### Integration / remaining Phase 1
- [x] `README.md` with install + readiness + CLI usage
- [x] `docs/VM_RUNBOOK.md` — staged H200 smoke (not real SWE-bench)
- [ ] End-to-end: collect trajectories with mock workers on a **local** toy repo
- [ ] End-to-end: train planner for 10 steps on mock buffer data
- [ ] End-to-end: eval planner on mock tasks
- [ ] Wire mock-worker flag into `fugu-collect` for one-command VM smoke
- [ ] Reproducible CUDA/PyTorch/vLLM install notes or lockfile from a known-good VM
- [ ] Docker isolation executor before real SWE-bench

## Deferred (see implementation_solutions.md)

- [ ] **#10** Full-sequence action probabilities (RL / soft BC)
- [ ] **#11 heavy** Counterfactual / quality-based worker routing labels
- [ ] **#12 docker** Container executor for real third-party SWE-bench
- [ ] **#13** Offload blocking git/tests from async event loop
- [ ] **#14 full** Enforce `max_disk_mb` before/after clone
- [ ] HumanEval / MBPP standalone workspace harness

## How to validate

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"   # required for import fugu / CLIs on VM
pytest tests/ -v          # pythonpath=src also set in pyproject for bare pytest
```

H200 staged procedure: **docs/VM_RUNBOOK.md** (do not skip isolation warnings).

Worker endpoints (source of truth: `configs/default.yaml`):

| Worker | Port |
|--------|------|
| Qwen | `http://localhost:8001/v1` |
| Gemma | `http://localhost:8002/v1` |
| Ornith | `http://localhost:8003/v1` |
