# Implementation Issues

This document records issues found in the Fugu implementation and their
resolution status. Companion: `implementation_solutions.md` (solutions +
work order), `task.md` (checklist).

**Last updated:** after fixes for issues **1–9**, **11 (light)**, **12 (gate)**,
**14 (basic cleanup)**, **15–17**. Unit suite: **38 tests passing**.

---

## Status summary

| # | Title | Status |
|---|--------|--------|
| 1 | Collector `NameError` | **FIXED** |
| 2 | CLI `variant` vs `split` | **FIXED** |
| 3 | Duplicate / incomplete transitions | **FIXED** |
| 4 | `STOP` bypasses env | **FIXED** |
| 5 | HumanEval/MBPP not executable | **FIXED** (SWE-bench only) |
| 6 | Unreliable test / reward signal | **FIXED** |
| 7 | Solved tasks do not auto-terminate | **FIXED** |
| 8 | Confusing `RETRY` supervision | **FIXED** |
| 9 | Train/infer prompt mismatch | **FIXED** |
| 10 | Approximate multi-token action probs | **Open** (defer to RL) |
| 11 | BC of heuristics ≠ quality routing | **Partial** (light filter) |
| 12 | Untrusted tests on host | **Partial** (gate; no docker yet) |
| 13 | Blocking work in async env | **Open** (sequential OK for now) |
| 14 | Disk budget not enforced | **Partial** (cleanup only) |
| 15 | Worker port doc mismatch | **FIXED** |
| 16 | No automated tests | **FIXED** |
| 17 | Local runtime (Python 3.11) | **FIXED** |

---

## Blockers (1–5) — all fixed

### 1. Trajectory collection raises `NameError` — FIXED

`TrajectoryCollector.collect_episode()` referenced `PlannerAction` and
`Metadata` without importing them.

- **Fix:** Collector always calls `env.step(action)` and records
  `env.transitions[-1]` (no direct use of those symbols).

### 2. SWE-bench CLI construction uses an invalid argument — FIXED

CLIs passed `variant=` to `SWEBenchDataset`, which accepts `split=`.

- **Fix:** `split` in both `cli/collect.py` and `cli/eval.py` dataset maps.

### 3. Collector stores incomplete duplicate transitions — FIXED

Collector rebuilt transitions and missed env metadata.

- **Fix:** Use `env.transitions[-1]`; env also sets `info["metadata"]` /
  `info["transition"]`.

### 4. `STOP` bypasses environment finalization — FIXED

Collector fabricated a zero-reward terminal transition without calling the env.

- **Fix:** `STOP` goes through `await env.step(action)`.

### 5. HumanEval and MBPP have no executable task environment — FIXED (SWE-bench only)

Standalone datasets lack `repo_url` / git patches.

- **Fix:** Collection and eval are SWE-bench-only. `CodingEnvironment.reset()`
  rejects missing `repo_url`. HumanEval/MBPP remain inspection-only.

---

## High-priority correctness

### 6. Test results are not a reliable reward signal — FIXED

- **Was:** `pytest -x` auto-detect; first `====` summary; harness commands
  could be rewritten.
- **Fix:** `task.test_command` is authoritative; pytest auto-detect is
  fallback only (no `-x`); parse **last** summary; append `specific_tests`
  only for pytest commands.
- **Code:** `src/fugu/execution/runner.py`

### 7. Solved tasks do not automatically terminate — FIXED

- **Was:** `all_tests_passed` computed but episode continued until `STOP` /
  budget.
- **Fix:** After **any** action with a **fresh** test evaluation (worker,
  `RETRY`, `RUN_TESTS`, `VERIFY`), if tests pass and compile succeeds →
  `done=True` and terminal reward on that step.
- **Code:** `src/fugu/env/coding_env.py`

### 8. Retry logic and fixed strategies produce confusing supervision — FIXED

- **Was:** Env `RETRY` re-called the worker; strategies also emitted
  `CALL_PRIMARY` after `RETRY`.
- **Fix:** `RETRY` remains a complete re-call. Strategies:
  `CALL → RETRY* → fallback CALL_*` (no double primary).
- **Code:** `src/fugu/trajectory/strategies.py`

### 9. Planner training and inference use different prompt formats — FIXED

- **Was:** Train used bare `to_prompt()`; infer added system + chat template.
- **Fix:** Shared `build_planner_prompt` in `planner/prompts.py` used by
  training and `PlannerModel`.
- **Code:** `src/fugu/planner/prompts.py`, `tokenizer.py`, `model.py`,
  `training/data.py`

### 10. Action probability calculation is approximate for multi-token actions — OPEN

`predict_with_probs()` still scores the first token of each action name.

- **When:** RL / soft BC / exploration.
- **Fix options:** Full-sequence scoring, single-token action IDs, or
  constrained decoding.

---

## Architecture and experiment design

### 11. Heuristic BC cannot yet learn quality-based routing — PARTIAL (light)

- **Light done:** `training/filter.py`; `train_from_buffer` drops
  non-positive-return episodes (falls back if filter empties the buffer).
- **Still open:** Same-task worker comparisons, return-weighted BC,
  preference losses, online RL.

### 12. Untrusted repository tests execute on the host — PARTIAL (gated)

- **Gate done:** `EnvConfig.isolation_mode` (`host`|`docker`) and
  `allow_host_execution`. Untrusted remote URLs raise `IsolationError` when
  host execution is disabled. CLI wires these into `TestRunner`.
- **Still open:** Real docker/container executor (`NotImplementedError` if
  `isolation_mode=docker`). **Required before real third-party SWE-bench.**

### 13. Blocking subprocess work inside async env — OPEN

Clone/compile/tests still block the event loop. Sequential collection is fine
for the current milestone.

### 14. Disk budget configured but not enforced — PARTIAL (basic cleanup)

- **Basic done:** Collector `try`/`finally` → `env.close()`; env honors
  `cleanup_on_done`.
- **Still open:** Pre/post-clone `max_disk_mb` checks and estimates.

### 15. Documentation and configuration disagree on worker ports — FIXED

Canonical map (`configs/default.yaml` = source of truth):

| Worker | Endpoint |
|--------|----------|
| Qwen | `http://localhost:8001/v1` |
| Gemma | `http://localhost:8002/v1` |
| Ornith | `http://localhost:8003/v1` |

---

## Validation

### 16. No automated tests — FIXED

`tests/` covers actions, state, reward, runner parse, strategies, buffer, CLI
datasets, prompts, filter, workers, env (auto-terminate, STOP, cleanup).

```bash
source .venv/bin/activate
export PYTHONPATH=src
pytest tests/ -v
```

### 17. Local runtime cannot validate the project — FIXED

Use Python **3.11+** (project requires `>=3.11`). Example:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install pytest pytest-asyncio msgpack zstandard \
  pydantic pydantic-settings pyyaml click rich httpx
export PYTHONPATH=src
pytest tests/ -v
```

Full package install: `pip install -e ".[dev]"` (pulls torch/transformers).

---

## Deployment readiness

**Not ready for real H200 SWE-bench collection.** Ready only for a controlled
smoke-test transfer after this tree is **committed and cloned**.

Blockers for real GPU-VM collection:

1. Docker isolation not implemented (`NotImplementedError`); host execution of
   untrusted remotes must stay off (`allow_host_execution: false` default).
2. No vLLM launch automation — operators must start workers on 8001–8003.
3. No CUDA/PyTorch lockfile — validate stack on the VM before large downloads.
4. Mock collect → train → eval not yet demonstrated end-to-end.

See **README.md** and **docs/VM_RUNBOOK.md**.

## Remaining work (ordered)

1. Commit + transfer (git or deliberate archive) — not an empty initial commit.
2. VM staged smoke per `docs/VM_RUNBOOK.md` (install, pytest, CUDA, vLLM health).
3. **Mock / local E2E** — collect → buffer → 10-step train → eval.
4. **#12 docker** — container executor before any real remote SWE-bench.
5. **#14 full** — enforce disk budget when sweeping many repos.
6. **#11 heavy** — quality routing after first SFT baseline.
7. **#10, #13** — when starting RL or concurrent collection.

---

## Explicitly deferred product scope

- HumanEval / MBPP executable harness (until standalone workspace exists).
- Online RL (PPO / REINFORCE) — Phase 4 in `explanation.md`.
