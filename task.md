# Task List — Fugu Orchestrator

## Phase 1: Foundation (Core + Mock Pipeline)

### Project Setup
- [x] `pyproject.toml` with all dependencies
- [x] Package structure (`src/fugu/`)
- [x] Configuration system (`config.py` + `default.yaml`)
- [x] `README.md`

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
- [x] `execution/runner.py` — TestRunner, CompileChecker

### Coding Environment
- [x] `env/coding_env.py` — CodingEnvironment (Gymnasium-style)

### Dataset Adapters
- [x] `datasets/base.py` — CodingTask, BaseDataset
- [x] `datasets/swebench.py` — SWE-bench / SWE-bench Lite adapter
- [x] `datasets/humaneval.py` — HumanEval adapter
- [x] `datasets/mbpp.py` — MBPP adapter

### Trajectory Generation
- [x] `trajectory/strategies.py` — Fixed orchestration strategies
- [x] `trajectory/collector.py` — TrajectoryCollector

### Replay Buffer
- [x] `buffer/replay_buffer.py` — Fixed-size compressed buffer

### Planner Model
- [x] `planner/model.py` — PlannerModel with QLoRA
- [x] `planner/tokenizer.py` — State serialization
- [x] `planner/prompts.py` — Shared train/infer prompt builder

### Training
- [x] `training/trainer.py` — PlannerTrainer (SFT with cross-entropy)
- [x] `training/data.py` — TransitionDataset
- [x] `training/filter.py` — Episode return/solved filtering

### CLI
- [x] `cli/collect.py` — fugu-collect command
- [x] `cli/train.py` — fugu-train command
- [x] `cli/eval.py` — fugu-eval command

### Tests (38 passing)
- [x] `tests/conftest.py` — shared fixtures
- [x] `tests/test_actions.py`
- [x] `tests/test_state.py`
- [x] `tests/test_reward.py`
- [x] `tests/test_workers.py`
- [x] `tests/test_env.py`
- [x] `tests/test_buffer.py`
- [x] `tests/test_cli_datasets.py`
- [x] `tests/test_runner_parse.py`
- [x] `tests/test_strategies.py`
- [x] `tests/test_prompts.py`
- [x] `tests/test_filter.py`

### Documentation
- [x] `docs/VM_RUNBOOK.md`
- [x] `implementation_plan.md`
- [x] `implementation_issues.md`
- [x] `implementation_solutions.md`
- [x] `explanation.md`
- [x] `project_context.md`

---

## Remaining Work

### Integration (not yet run)
- [ ] End-to-end: collect trajectories with mock workers on SWE-bench Lite
- [ ] End-to-end: train planner for 10 steps on mock data
- [ ] End-to-end: eval planner on mock tasks

### Open Issues (deferred, not blockers)
- [ ] #10 — Full-sequence action probabilities (needed for RL, not SFT)
- [ ] #13 — Offload blocking subprocess work for concurrent collection
- [ ] #14 full — Enforce `max_disk_mb` estimates at scale
- [x] #12 docker — Fugu container executor (`isolation_mode=docker`; not official harness)
- [ ] Official SWE-bench harness / per-instance images (eval fidelity)
