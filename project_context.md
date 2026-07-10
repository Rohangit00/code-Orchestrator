# Fugu Orchestrator - Project Context & Initial Instructions

This document preserves the core architecture, constraints, and implementation
notes for the Fugu orchestrator. Module scaffold is **in place**; correctness
fixes for issues **1–9**, **11 light**, **12 gate**, **14 basic**, **15–17**
are applied. See `implementation_issues.md`, `implementation_solutions.md`,
and `task.md` for live status.

## Project Architecture & Core Formalism

Fugu is a coding-oriented LLM orchestrator that learns to coordinate multiple coding models using a Markov Decision Process (MDP) formalism.

- **Infrastructure**: Designed to run Qwen 2.5, Gemma 27B/31B, and Ornith on vLLM/h200s (across multiple ports).
- **Dataset Priority**: SWE-bench Lite is the primary (and currently only) target for trajectories and evaluation.
- **Constraints**: Strict disk usage limit (~8GB workspace) handled by `RepoManager` (single-repo-at-a-time). Episode cleanup via `cleanup_on_done` is enforced; full `max_disk_mb` pre-checks still deferred.
- **Worker endpoints (source of truth: `configs/default.yaml`)**:
  - Qwen → `http://localhost:8001/v1`
  - Gemma → `http://localhost:8002/v1`
  - Ornith → `http://localhost:8003/v1`

### The MDP State & Action Space

**State (`PlannerState`)**:
- Serialized via `to_prompt()` and wrapped by shared `build_planner_prompt()` for train and infer.
- Contains: `task_description`, `repo_context`, `history` (last steps), `test_results`, `compile_status`, `current_patch`, `step_number`, `max_steps`, `remaining_budget`.

**Actions (`PlannerAction` IntEnum)** — 7 discrete actions:

1. `CALL_QWEN` (0)
2. `CALL_GEMMA` (1)
3. `CALL_ORNITH` (2)
4. `RUN_TESTS` (3)
5. `VERIFY` (4) — compile check + run tests
6. `RETRY` (5) — **complete** re-call of the last worker with error context (env applies patch + tests)
7. `STOP` (6)

**Episode termination**

- After any action that produces a **fresh** test evaluation (worker / `RETRY` / `RUN_TESTS` / `VERIFY`): if target tests pass and compile succeeds → episode ends (terminal success reward on that step).
- `STOP` and step-budget exhaustion also end the episode.

**Test commands**

- `task.test_command` (or native harness command) is **authoritative**.
- Pytest auto-detection is **fallback only** (no `-x` for reward measurement).

**Isolation**

- `env.isolation_mode`: `host` | `docker`
- `env.allow_host_execution`: must be `false` for untrusted remote repos until a docker executor exists (`isolation_mode=docker` raises `NotImplementedError` until implemented).

---

## Implemented module notes

### Trajectory generation & replay buffer

**`src/fugu/trajectory/strategies.py`** — Fixed orchestration policies

- `BaseStrategy(ABC)`: `name`, `select_action(state, step) -> PlannerAction`
- `SingleWorkerStrategy`: `CALL_X → RETRY* → STOP` (env evaluates after workers; early STOP if solved)
- `RoundRobinStrategy`: `CALL_QWEN → CALL_GEMMA → CALL_ORNITH → STOP`
- `RetryOnFailStrategy`: `CALL_PRIMARY → RETRY* → CALL_FALLBACK…` (**no** `RETRY` + `CALL_PRIMARY`)
- `VerifyFirstStrategy`: `VERIFY → CALL_X → VERIFY → STOP`
- `ALL_STRATEGIES`: Instantiated defaults

**`src/fugu/trajectory/collector.py`**

- `collect_episode(task, strategy)`: steps env (including `STOP`), records `env.transitions[-1]`, `try`/`finally` → `env.close()`
- `collect_dataset` / `collect_multi_strategy`: dataset sweeps with logging

**`src/fugu/buffer/replay_buffer.py`**

- `add` / `add_episode` / `sample` / `save` / `load` — msgpack + zstandard

### Planner model & tokenizer

**`src/fugu/planner/prompts.py`**

- `SYSTEM_PROMPT`, `build_planner_prompt(state, tokenizer=…)` — single path for train and infer

**`src/fugu/planner/model.py`**

- QLoRA planner; `predict` / `predict_with_probs` (first-token probs still approximate — issue #10 open)
- `get_action_prompt` → `build_planner_prompt`

**`src/fugu/planner/tokenizer.py`**

- `StateTokenizer.format_for_training` uses shared prompt builder + action name completion

### Training pipeline

**`src/fugu/training/data.py`** — `TransitionDataset` with prompt-token label masking (`-100`)

**`src/fugu/training/filter.py`** — episode grouping; min-return / require-solved filters

**`src/fugu/training/trainer.py`** — SFT via HF Trainer; `train_from_buffer` applies light quality filter

### CLI

**`fugu-collect` / `fugu-eval`** — SWE-bench only (`swebench-lite|full|verified`); wires isolation + `cleanup_on_done`

**`fugu-train`** — train from buffer

### Tests

```bash
source .venv/bin/activate && export PYTHONPATH=src && pytest tests/ -v
```

---

## Remaining (not blocking local mock development)

1. Docker/container executor for real third-party SWE-bench (#12 full)
2. Disk budget estimates at scale (#14 full)
3. Heavy routing labels / return-weighted BC (#11 heavy)
4. Multi-token action probabilities (#10) and async offload (#13)
5. README + mock E2E collect/train/eval smokes (`task.md`)
6. HumanEval/MBPP standalone harness (deferred under #5)
