# Implementation Solutions — Issues 6–17

Solutions, prioritization, and **implementation status** for Fugu issues after
blockers **1–5** (collector, CLI `split`, transitions, `STOP`, SWE-bench-only).

**Context:** SWE-bench is the only supported collection/eval path for now.
HumanEval / MBPP are deferred until a standalone workspace exists.
See `implementation_issues.md`, `project_context.md`, `task.md`, and
`explanation.md`.

### Implementation status (current)

| Work item | Status |
|-----------|--------|
| #17 + #16 toolchain + tests | **Done** (`.venv`, 38 tests) |
| #6 test command / parsing | **Done** |
| #7 + #8 terminate + RETRY | **Done** |
| #15 worker ports | **Done** (8001/8002/8003) |
| #14 basic cleanup | **Done** (try/finally; estimates deferred) |
| #9 + #11 light prompts/filter | **Done** |
| #12 isolation gate + docker sandbox | **Done** (not official SWE-bench harness) |
| #10, #11 heavy, #13, #14 full | **Deferred** |

---

## Priority overview (historical work order — completed through P2 gate)

| Priority | Issues | Goal |
|----------|--------|------|
| **P0 — toolchain & safety net** | 17, 16 | Python 3.11+ env; minimal tests so later fixes have regression coverage |
| **P0 — before real collection** | 6, 7, 8, 15, 14 (basic) | Trustworthy rewards/labels; config clarity; always cleanup |
| **P1 — before planner SFT** | 9, 11 (light) | Train/serve parity; success/return-based filtering |
| **P2 — before real third-party SWE-bench** | 12 | Isolated execution (required before untrusted repos) |
| **P3 — later / research** | 10, 11 (heavy), 13, 14 (full disk estimates) | RL probs, counterfactual routing, concurrency, sophisticated disk budgeting |

### Work order (executed)

1. **#17 + #16** — Python 3.11 environment and minimal test suite.
2. **#6** — Test execution and parsing.
3. **#7 + #8** — Episode completion; `RETRY` complete re-call; strategies.
4. **#15** — Reconcile worker ports in config and docs.
5. **#14 (basic cleanup)** — `cleanup_on_done` via `try`/`finally`.
6. **#9 + #11 (light)** — Shared prompts and success/return-based filtering.
7. **#12** — Isolation **gate** (docker implementation deferred).
8. **Still deferred** — **#10**, **#11 (heavy)**, **#13**, full **#14** disk estimates, docker executor.

---

## High-priority correctness

### 6. Test results are not a reliable reward signal — IMPLEMENTED

**Problem.** Auto-detected commands use `pytest -x`, so runs stop after the
first failure. Pass-rate and test-delta rewards become wrong. The parser also
takes the first `==== … ====` block, which may be a heading rather than the
final summary.

**Why it matters.** The MDP reward and trajectory quality depend on test
deltas. Bad test signals poison the replay buffer and any downstream SFT.
Fix this **before collecting** any training trajectories.

**Solution.**

| Area | Change |
|------|--------|
| Command authority | Treat `task.test_command` or the benchmark’s **native evaluation harness** as authoritative. SWE-bench spans projects with different setup and test entrypoints; do **not** assume every task can run as `python -m pytest` plus appended test IDs. |
| Fallback only | Use pytest auto-detection only when no task/harness command is available (dev fixtures, mock tasks). |
| Flags (when pytest is used) | Do **not** use `-x` for reward measurement. Prefer full suite/summary or machine-readable reports (`--tb=no -q`, junit XML, JSON plugin, etc.). |
| Parsing | Parse the **last** summary line/block, or machine-readable report only. |
| Failure types | Distinguish collection/setup errors from assertion failures; surface both in `TestResults` (e.g. `errors` vs `failed`). |
| SWE-bench scoping | When the harness or task already defines which tests to run, pass that command through unchanged. Optional node-id lists are an optimization only when the project’s runner is known to be pytest-compatible. |

**Primary files**

- `src/fugu/execution/runner.py`
- `src/fugu/env/coding_env.py` (forward `task.test_command` / harness command)
- Dataset adapters if native harness command is not yet populated on `CodingTask`

**Done when**

- Tasks with an explicit `test_command` always run that command (no silent
  rewrite to generic pytest).
- Fallback pytest path (no command provided) reports full pass/fail/error
  counts without `-x` early stop.
- Synthetic pytest output (and ideally a tiny real suite) parses to expected
  counts.
- Reward deltas move in the expected direction when more tests pass.
- Regression tests cover parsing and “authoritative command is not overwritten.”

---

### 7. Solved tasks do not automatically terminate — IMPLEMENTED

**Problem.** `all_tests_passed` is computed but does not set `done`. Episodes
continue until `STOP` or the step budget. Terminal success bonus can be missed
if the policy never issues `STOP`.

**Why it matters.** Episode returns, strategy length, and eval metrics
(`solved` vs “tests happened to pass once”) diverge.

**Solution — auto-terminate on fresh solve after any action.**

The environment already runs tests immediately after a successful worker
patch (and after `RETRY`). Limiting auto-termination to `RUN_TESTS` / `VERIFY`
would force an extra step even when the suite already passed on the worker
step.

```text
After ANY action that produces a fresh test evaluation:
  if target tests pass AND compile succeeds (when compile was run) → done = True
  apply terminal success reward on that step

This includes:
  - CALL_* / RETRY when the env applies a patch and runs tests
  - RUN_TESTS
  - VERIFY

STOP still ends the episode (explicit halt / give-up)
Budget exhaustion still ends the episode
```

Do **not** require a follow-up `RUN_TESTS` or `VERIFY` solely to mark success
when tests already ran as part of the worker path.

**Primary files**

- `src/fugu/env/coding_env.py` (termination block ~`all_tests_passed`)
- `src/fugu/core/reward.py` (terminal bonus on the solving step)
- Strategies that assume STOP-only termination (may still emit `STOP` for
  unsolved episodes)

**Done when**

- Mock episode that fixes all tests on a worker call ends on that step
  (no extra `RUN_TESTS` / `STOP` required for success).
- Terminal success bonus appears on the solving step.
- Behavior is documented in code comments / this file.
- Regression test covers worker-step auto-termination and explicit `STOP`.

---

### 8. Retry logic and fixed strategies produce confusing supervision — IMPLEMENTED

**Problem.** Env `RETRY` already re-calls the last worker with error context.
`RetryOnFailStrategy` also emits `RETRY` then `CALL_PRIMARY`, so the worker
runs twice and labels do not teach a clear “when to retry” policy.

**Why it matters.** Behavioral cloning will learn noise: double worker actions
and ambiguous meaning of `RETRY` vs `CALL_*`.

**Solution — choose Option A (matches `project_context.md`).**

| Option | Meaning of `RETRY` | Strategy pattern |
|--------|--------------------|------------------|
| **A (adopt)** | Full re-generate on last worker | `… → RETRY → …` (no extra `CALL_*` after `RETRY`) |
| B (reject for now) | Hint only; env does not call worker | Strategy: `RETRY` then `CALL_*` |

**Concrete changes**

1. Keep env `RETRY` as the complete re-generation action (patch + tests when
   applicable). Because of **#7**, a successful `RETRY` can end the episode
   immediately.
2. Update `RetryOnFailStrategy` (and any docs) so retry loops do **not** pair
   `RETRY` with another `CALL_PRIMARY`. Pattern:
   `CALL_PRIMARY → (RETRY)* → fallback CALL_OTHER → …`, with tests already
   driven by the env after worker actions; add explicit `RUN_TESTS` only when
   the strategy needs a test-only step.
3. If no previous worker exists, `RETRY` is a no-op with a clear log and small
   penalty (already partially true).

**Primary files**

- `src/fugu/trajectory/strategies.py`
- `src/fugu/env/coding_env.py` (only if clarifying no-op / logging)
- `project_context.md` / strategy docstrings if they still show dual pattern

**Done when**

- A fail-then-retry episode has one worker invocation per `RETRY` step.
- Buffer labels show a clean action sequence suitable for BC.
- Strategy unit tests lock the no-double-call sequence.

---

### 9. Planner training and inference use different prompt formats — IMPLEMENTED

**Problem.** Training builds labels from `PlannerState.to_prompt()` only.
Inference adds a system instruction and may apply the chat template.
Train/serve mismatch hurts action prediction.

**Why it matters.** Even good trajectories will not transfer if the model never
saw inference-time formatting during SFT.

**Solution.**

1. Single function used everywhere, e.g.  
   `build_planner_prompt(state: PlannerState) -> str`  
   (live in `planner/tokenizer.py` or a small `planner/prompts.py`).
2. Same steps for train and infer:
   - state serialization (tagged fields)
   - optional system instruction
   - model chat template / special tokens
   - action completion boundary (what is label vs context)
3. `TransitionDataset` and `PlannerModel.predict` / `predict_with_probs` both
   call this builder.
4. Unit test: training string prefix equals inference prompt prefix for the
   same state.

**Primary files**

- `src/fugu/planner/tokenizer.py`
- `src/fugu/planner/model.py`
- `src/fugu/training/data.py`

**Done when**

- One code path constructs prompts for train and eval.
- A smoke train + greedy eval no longer depends on accidental format luck.
- Regression test enforces train/infer prompt parity.

---

### 10. Action probability calculation is approximate for multi-token actions

**Problem.** `predict_with_probs()` scores only the first token of each action
name. Multi-token names are not true sequence probabilities.

**Why it matters.** Relevant for exploration, soft labels, and later RL—not for
greedy SFT argmax over full generated action strings.

**Solution (defer until RL / soft BC).**

Pick one when needed:

| Approach | Notes |
|----------|--------|
| Autoregressive score of full action token sequence | Accurate; a bit more compute |
| Single-token action IDs / short aliases | Map `CALL_QWEN` → one reserved token |
| Constrained decoding over the 7 action strings | Clean discrete policy |

**Primary file:** `src/fugu/planner/model.py`

**Done when (later):** `predict_with_probs` sums/logs over full action
sequences and is tested against a fixed tokenizer.

---

## Architecture and experiment design

### 11. Heuristic behavioral cloning cannot yet learn quality-based routing — LIGHT IMPLEMENTED

**Problem.** Every fixed-strategy action is treated as a positive label. No
filtering by success/return, no same-state worker comparisons. The planner
imitates schedules, not “best worker for this state.”

**Why it matters.** Expected for Phase 1–2. Not a bug in the trainer; a
limitation of the data recipe.

**Solution — staged.**

**Light (P1, with #9, before SFT):**

1. Collect multi-strategy trajectories on SWE-bench Lite (small N) only after
   **#6–#8** and isolation policy for the chosen workload.
2. Filter or upweight episodes with positive return / tests improved / solved.
3. Evaluate planner against fixed strategies (accuracy + solve rate + cost).

**Heavy (research / Phase 3+):**

- Same-task comparisons across workers for routing labels.
- Return-weighted BC or preference-style losses.
- Online RL (PPO / REINFORCE) per `explanation.md` Phase 4.

**Primary files**

- Collection: `trajectory/*`, `cli/collect.py`
- Training filter: `training/data.py` or a buffer filter utility
- Eval: `cli/eval.py`

**Done when (light):** training set is not dominated by failed no-op schedules;
eval reports planner vs baseline strategies.

---

### 12. Untrusted repository tests execute directly on the host — GATE IMPLEMENTED

**Problem.** `shell=True` in cloned third-party repos. Timeout alone does not
stop network, filesystem abuse, secret access, or resource exhaustion.

**Why it matters.** Real SWE-bench on a shared machine is a security and
stability risk. Mock / toy repos are lower risk.

**Solution — require isolation before real third-party SWE-bench.**

| Stage | Controls |
|-------|----------|
| **Dev / mock** | Dedicated workspace dir; `cleanup_on_done`; timeouts; no secrets in env. OK without containers. |
| **Real third-party SWE-bench** | **Required:** container or VM with CPU/mem/disk limits, network off or allowlisted, mount only workspace, non-root. Do not run real repos on the bare host. |
| **Ideal** | Per-task container image aligned with SWE-bench harness practices |

**Primary files**

- `src/fugu/execution/runner.py`
- Possibly `src/fugu/repo/manager.py` (workspace layout)

**Done when**

- Real collection path refuses or is clearly gated without an isolated
  executor.
- Host home / credentials are not mounted into the test process.

---

### 13. Blocking subprocess work runs inside an async environment

**Problem.** `CodingEnvironment.step()` is async, but clone/compile/tests are
sync and block the event loop.

**Why it matters.** Only matters when parallelizing many concurrent episodes.
Current CLIs are effectively sequential (`asyncio.run` per batch).

**Solution.**

- **Now:** keep collection sequential (no change required for correctness).
- **Later:** run blocking ops in `asyncio.to_thread` or a process pool before
  multi-worker collection.

**Primary file:** `src/fugu/env/coding_env.py`

**Done when (later):** concurrent collection does not stall the loop under load.

---

### 14. Disk budget is configured but not enforced — BASIC CLEANUP IMPLEMENTED

**Problem.** `RepoManager.max_disk_mb` is stored; clone does not check budget
before/after. Conflicts with the ~8GB / single-repo design in
`project_context.md`. Cleanup may also be skipped on exceptions.

**Why it matters.** Missed cleanup fills the disk even at small scale.
Sophisticated pre-clone estimates matter mainly at Lite-scale sweeps.

**Solution — two tiers.**

**Basic (do now, step 5 in work order):**

1. Honor `cleanup_on_done` with `try` / `finally` in the collector (and env
   `close()`), so the active repo is removed on success, failure, or exception.
2. Log cleanup and current disk usage per episode when easy.

**Full (defer until scale):**

1. Before clone: if current workspace usage + estimate ≥ budget → cleanup /
   refuse.
2. After clone / tests: re-check; delete artifacts if over budget.
3. Enforce `max_disk_mb` loudly rather than filling the disk.

**Primary files**

- `src/fugu/trajectory/collector.py` / `env/coding_env.py` (guaranteed close)
- `src/fugu/repo/manager.py` (later: budget checks)

**Done when (basic)**

- Episode end always removes the active repo when `cleanup_on_done` is true,
  including exception paths.
- Regression test or mock integration proves cleanup runs after failure.

**Done when (full, later)**

- Exceeding `max_disk_mb` fails loudly or forces cleanup before clone.

---

### 15. Documentation and configuration disagree on worker ports — IMPLEMENTED

**Problem.** Plan docs say ports 8000–8002; `configs/default.yaml` uses
8001–8003. Wrong endpoints fail in non-obvious ways.

**Solution.**

1. **Source of truth:** `configs/default.yaml` (and env overrides `FUGU_*`).
2. Align `implementation_plan.md` (and any future README) to the same map.
3. Optional: print resolved worker URLs at start of `fugu-collect` / `fugu-eval`.

**Suggested canonical map (match current config unless infra says otherwise):**

| Worker | URL |
|--------|-----|
| Qwen | `http://localhost:8001/v1` |
| Gemma | `http://localhost:8002/v1` |
| Ornith | `http://localhost:8003/v1` |

**Primary files**

- `configs/default.yaml`
- `implementation_plan.md`
- CLI banners (optional)

**Done when** docs and config list the same three endpoints.

---

## Validation gaps

### 16. There are no automated tests in the repository — IMPLEMENTED

**Problem.** `pyproject.toml` configures pytest; no `tests/` directory. Recent
collector/CLI fixes have no regression net. Subsequent issue fixes should not
land without coverage.

**Solution — minimal first suite (with #17, first in work order).**

| Test module | Covers |
|-------------|--------|
| `tests/test_actions.py` | Enum values / names |
| `tests/test_state.py` | `to_prompt`, serialize round-trip |
| `tests/test_reward.py` | Test delta, terminal bonus, retry penalty |
| `tests/test_runner_parse.py` | Pytest summary / JSON parsing; command not overwritten (issue 6) |
| `tests/test_workers.py` | Mock worker + pool dispatch |
| `tests/test_env.py` | reset → steps → STOP/success; transition metadata; auto-terminate on worker solve (issue 7) |
| `tests/test_buffer.py` | add / sample / save / load |
| `tests/test_cli_datasets.py` | SWE-bench map uses `split=` only |
| Integration | One mock-worker episode → buffer → reload; cleanup on failure (issue 14 basic) |

Expand this suite as **#6–#9** and **#14** are implemented (each fix adds or
extends tests).

**Primary layout:** `tests/` as listed in `task.md` / `implementation_plan.md`.

**Done when**

```bash
pytest tests/ -v
```

passes on a 3.11+ env with `pip install -e ".[dev]"`.

---

### 17. The current local runtime cannot validate the project — IMPLEMENTED

**Problem.** Package requires Python ≥ 3.11; some machines still have 3.9.
Dev extras (pytest, ruff, mypy) may be missing.

**Solution (ops, first with #16).**

```bash
# Minimal (enough for current unit suite)
python3.11 -m venv .venv
source .venv/bin/activate
pip install pytest pytest-asyncio msgpack zstandard \
  pydantic pydantic-settings pyyaml click rich httpx
export PYTHONPATH=src
pytest tests/ -v

# Full project install (torch / transformers / peft)
pip install -e ".[dev]"
```

Pin the interpreter in README when added. CI later can enforce 3.11+.

**Done when** a developer can install and run the first test suite without
version errors. **Met:** 38 tests pass on Python 3.11 `.venv`.

---

## Mapping to project phases

| Phase | Issues |
|-------|--------|
| Toolchain | 17, 16 |
| Foundation / mock E2E | 6, 7, 8, 15, 14 (basic cleanup) |
| First real SFT | 9, 11 light |
| Real third-party SWE-bench | 12 (required), then 14 full at scale |
| RL / advanced routing | 10, 11 heavy, 13 |

---

## Explicitly out of scope for this doc

- Issues **1–5** (already addressed: collector, CLI `split`, transitions,
  `STOP`, SWE-bench-only).
- Building HumanEval / MBPP standalone harness (tracked as deferred under #5).
- Full PPO / online RL implementation (Phase 4).
- Sophisticated pre-clone disk estimation (deferred part of #14).

---

## Checklist (copy into PRs)

Work roughly in this order; add tests with each behavioral change.

- [x] **#17** Python 3.11+ venv (`.venv`) + pytest runnable
- [x] **#16** Minimal `tests/` suite green (38 tests)
- [x] **#6** Authoritative `test_command` / native harness; pytest only as fallback; no `-x`; robust parsing
- [x] **#7** Auto-terminate after any action with fresh passing tests (including worker/`RETRY` steps)
- [x] **#8** `RETRY` = full re-call only; strategies do not double-call
- [x] **#15** Single worker port map in config + docs (8001/8002/8003)
- [x] **#14** Basic: `cleanup_on_done` via `try`/`finally` (full disk estimates later)
- [x] **#9** Shared train/infer prompt builder (`planner/prompts.py`)
- [x] **#11** Light: filter non-positive-return episodes in `train_from_buffer`
- [x] **#12** Isolation gate (`allow_host_execution` / `isolation_mode`); docker executor still TODO
- [ ] **#10** (later) Full-sequence action probabilities
- [ ] **#13** (later) Offload blocking work when parallelizing
- [ ] **#14** (later) Enforce `max_disk_mb` estimates at scale
- [ ] **#12** (later) Implement docker/container executor for real SWE-bench
