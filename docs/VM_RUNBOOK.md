# H200 VM runbook — Fugu

**First time?** Prefer the shorter [VM_START.md](VM_START.md).

**Primary path:** LiveCodeBench (Python) collect → buffer → train planner.  
**One Docker image with pytest** is enough for LCB.

SWE-bench is optional (repo-backed; multi-env pain). Official SWE harness scoring is out of scope for day-to-day train.

Tip commit for LCB: **`6465c4f`** or later on `main`.

---

## Not ready / ready

| Goal | Ready? |
|------|--------|
| Unit tests (57) | Yes |
| LCB collect smoke (`-n 1`) | Yes after image + workers |
| LCB scale collect (`-n 20`) | Yes; **analyze buffer** before train (Stage 6) |
| LCB train from buffer | Yes after healthy buffer summary + CUDA torch |
| Full SWE-bench Lite @ official harness | Separate tool (`swebench`); lots of disk if scaled |
| Host execution of untrusted git tests | No — keep `allow_host_execution: false` |

---

## Shell sessions, exports, screen

Exports die when the terminal exits. Re-export or:

```bash
# Prefer: copy example (gitignored private file)
cp env_vm.sh.example env_vm.sh
# edit env_vm.sh — set HF_TOKEN and worker URLs; never commit env_vm.sh

# env_vm.sh.example includes placeholders:
#   export HF_TOKEN=hf_YOUR_TOKEN_PLACEHOLDER
#   export FUGU_WORKER__QWEN_URL=http://HOST:PORT/v1
#   export FUGU_ENV__ISOLATION_MODE=docker
#   export FUGU_ENV__DOCKER_IMAGE=fugu-py311-pytest
```

```bash
screen -S fugu          # keep secrets only in this session
source .venv/bin/activate
source env_vm.sh        # HF_TOKEN + workers + docker; dies when screen ends
```

Hub auth: set **`HF_TOKEN`** or **`FUGU_HF_TOKEN`** (read token from
https://huggingface.co/settings/tokens). Fugu passes it into Hub downloads;
the token value is never logged.

URLs must end in **`/v1`** only.

**screen** for long collect:

```bash
screen -S fugu
# … run collect …
# detach: Ctrl-A D   reattach: screen -r fugu
```

---

## Stage 0 — Code

```bash
git clone <remote> clever-turing   # or cd existing
cd clever-turing
git pull
git log -1 --oneline   # 6465c4f+ for LiveCodeBench
```

---

## Stage 1 — Python + package

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
# H200: install CUDA torch before editable install if needed
# pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python -c "import fugu; print(fugu.__version__)"
```

Torch must be in **this** venv. Optional: `pip freeze > vm-requirements-freeze.txt`.

---

## Stage 2 — Unit tests

```bash
pytest tests/ -v
# expect 57 passed
```

---

## Stage 3 — Planner GPU smoke (optional until train)

Skip if only collecting. Needed before serious `fugu-train`.

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
from fugu.config import PlannerConfig
from fugu.planner.model import PlannerModel
m = PlannerModel(PlannerConfig(base_model="Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True))
m.load()
print("planner ok")
PY
```

---

## Stage 4 — vLLM workers

Fugu does **not** start workers. If already up, only set URLs and curl `/v1/models`.

Defaults in config: 8001 / 8002 / 8003. Override with `FUGU_WORKER__*_URL`.

Confirm Fugu sees config:

```bash
python - <<'PY'
from fugu.config import FuguConfig
c = FuguConfig.from_yaml("configs/default.yaml")
print(c.worker.qwen_url, c.env.isolation_mode, c.env.docker_image)
PY
```

Env vars override YAML (after config fix in `6465c4f` lineage / `c830b91`).

---

## Stage 5 — Pipeline order (train vs score)

```text
1. Collect  (LCB tasks + workers + docker tests) → buffer
2. Train    (planner SFT on buffer)
3. Eval     (optional fugu-eval on livecodebench-test)
```

| Step | Planner weights? | Workers? | Docker image? |
|------|------------------|----------|---------------|
| Collect LCB | No | Yes | One with pytest |
| Train | Yes | No | No |
| Official SWE resolve % | No | No* | Harness images (*not Fugu) |

\* Harness is a separate `pip install swebench` flow — not required for training.

---

## Stage 6 — LiveCodeBench collect (primary)

### Rebuild image (e.g. after `docker rmi` / prune)

```bash
docker build -t fugu-py311-pytest - <<'EOF'
FROM python:3.11-slim
RUN pip install --no-cache-dir pytest
EOF
docker run --rm fugu-py311-pytest python -m pytest --version
```

### Delete old images

```bash
docker images
docker rmi <name_or_id>
docker image prune -f
docker system df
```

### Config

```bash
export FUGU_ENV__ISOLATION_MODE=docker
export FUGU_ENV__ALLOW_HOST_EXECUTION=false
export FUGU_ENV__DOCKER_IMAGE=fugu-py311-pytest
# source env_vm.sh for workers too
```

### Smoke (`-n 1`)

```bash
fugu-collect \
  -c configs/default.yaml \
  -d livecodebench-train \
  -s single-qwen \
  -n 1 \
  -o "data/buffer_lcb_smoke_$(date +%Y%m%d_%H%M%S)"
```

### Scale (`-n 20`) then analyze before train

```bash
fugu-collect -d livecodebench-train -s single-qwen -n 20 -o data/buffer_lcb_train
```

**Do not jump straight to train** until you have checked the buffer (next subsection).
A tiny 20-task buffer is for **pipeline smoke** (collect finished + labels sane), not paper numbers.

If long runs hang (e.g. after ~12 tasks), chunk and raise timeouts:

```bash
fugu-collect -d livecodebench-train -s single-qwen -n 10 -o data/buffer_lcb_qwen_a
fugu-collect -d livecodebench-train -s single-qwen -n 10 -o data/buffer_lcb_qwen_b
# raise worker / docker test timeouts via config or FUGU_* if needed
```

### What `fugu-collect` does (LCB)

1. Load HF LiveCodeBench problems (train split if `-d livecodebench-train`)  
2. Create **standalone** workspace (no git clone)  
3. Baseline tests in Docker  
4. Worker returns **Python code** → write `solution.py`  
5. Re-run tests → reward → buffer  
6. RETRY / STOP per strategy  

| Flag | Meaning |
|------|---------|
| `-d livecodebench-train` | Train split (default for collect) |
| `-d livecodebench-val` / `-test` / `livecodebench` | Other splits / all |
| `-s single-qwen` | Strong/expensive worker only |
| `-s single-ornith` | Cheap worker only (preferred default try) |
| `-s retry-on-fail` | Ornith → RETRY* → escalate Qwen |
| `-s round-robin` | Ornith → Qwen → STOP (Gemma disabled) |
| `-n 1` | Smoke |
| `-o dir` | Buffer directory (use unique names) |

**Active workers:** Qwen + Ornith only. `CALL_GEMMA` remains in the enum for compatibility but is **disabled** (not in prompt, not registered by default, env refuses, not in strategies).

### Buffer format

- File is often named `buffer.jsonl` but content is **binary** (`FUGU_RB` + zstd).  
- Do not open as text; use `ReplayBuffer.load` (see VM_START).  
- `data/` is gitignored (grey in IDE).  
- Reusing the same `-o` **overwrites**; empty failed runs can wipe good data.

### Permissions

If you see `Permission denied` on files under `/workspace`:

```bash
unset FUGU_ENV__DOCKER_USER
# or match host: export FUGU_ENV__DOCKER_USER="$(id -u):$(id -g)"
# chown workspace dirs if needed
```

LCB workspaces live under `repo.workspace_dir` (default `/tmp/fugu_workspaces`) + `standalone/`.

### Success / failure

| Good | Bad for training |
|------|------------------|
| Transitions &gt; 0 | `No module named pytest` |
| Real pytest output | Permission denied on mount |
| Some tests pass/fail counts | Always empty zero tests |

### Analyze `-n 20` (or any) buffer

`buffer.jsonl` is **binary** (`FUGU_RB` + msgpack + zstd). Do not `cat` it.
`data/` is gitignored (often grey in IDEs) — use the terminal.

```bash
ls -lah data/buffer_lcb_train/
file data/buffer_lcb_train/buffer.jsonl
```

**Full summary script** (repo root, venv active). Change `path` if you used a different `-o`:

```bash
python - <<'PY'
from pathlib import Path
from collections import Counter
from fugu.buffer.replay_buffer import ReplayBuffer
from fugu.training.filter import group_episodes, episode_return, episode_solved

path = Path("data/buffer_lcb_train/buffer.jsonl")  # change if needed
assert path.exists(), path

buf = ReplayBuffer(capacity=100_000, storage_dir=str(path.parent))
buf.load(str(path))
transitions = list(buf)

print("=== BUFFER ===")
print(f"transitions: {len(transitions)}")
print(f"file size:   {path.stat().st_size / 1e6:.2f} MB")

actions = Counter(t.action.name for t in transitions)
print("\n=== ACTIONS ===")
for k, v in actions.most_common():
    print(f"  {k:12s} {v}")

eps = group_episodes(transitions)
print(f"\n=== EPISODES ===")
print(f"episodes: {len(eps)}  (expect ~20 if -n 20 finished)")

solved = 0
returns = []
steps = []
rows = []
for i, ep in enumerate(eps):
    ret = episode_return(ep)
    ok = episode_solved(ep)
    solved += int(ok)
    returns.append(ret)
    steps.append(len(ep))
    last = ep[-1]
    ta = last.metadata.tests_after
    p = f = e = 0
    out = ""
    if ta is not None:
        p, f, e = ta.passed, ta.failed, ta.errors
        out = (ta.output or "")[:120].replace("\n", " ")
    tid = getattr(last.state, "task_id", "") or getattr(last.metadata, "task_id", "") or "?"
    rows.append((i, tid, len(ep), ret, ok, p, f, e, out))

print(f"solved (all public tests pass at end): {solved}/{len(eps)}")
if returns:
    print(f"return: mean={sum(returns)/len(returns):.3f}  "
          f"min={min(returns):.3f}  max={max(returns):.3f}")
    print(f"steps/ep: mean={sum(steps)/len(steps):.1f}  max={max(steps)}")

print("\n#  task_id                    steps   return  solved  p/f/e  tail output")
for i, tid, n, ret, ok, p, f, e, out in rows:
    print(f"{i:2d}  {str(tid)[:24]:24s}  {n:3d}  {ret:7.2f}  {ok!s:5s}  "
          f"{p}/{f}/{e}  {out[:80]}")

bad = 0
for t in transitions:
    ta = t.metadata.tests_after
    if ta and ("No module named pytest" in (ta.output or "")
               or "Permission denied" in (ta.output or "")):
        bad += 1
print(f"\ntransitions with pytest-missing or permission errors: {bad}")

from fugu.training.filter import filter_transitions
kept = filter_transitions(transitions, min_return=0.0)
print(f"kept after min_return=0: {len(kept)} / {len(transitions)}")
PY
```

`fugu-collect` also prints a **Collection Summary** (transition count, elapsed, path) when the job finishes. If that table never appeared, the job was killed/timed out — the buffer may still load with fewer episodes.

#### How to read the numbers

| Signal | Healthy `-n 20` | Unhealthy |
|--------|-----------------|-----------|
| **Episodes** | ≈ 20 (or slightly fewer if one task crashed) | 0, or hung midway (e.g. stop at ~12) |
| **Transitions** | A few × episodes (CALL → tests → RETRY* → STOP) | 0 |
| **Actions** | Mix of `CALL_QWEN`, maybe `RETRY`, `STOP` | Only `STOP`, or never `CALL_*` |
| **Solved rate** | Can be low (contest hard + single worker); even **0/20** can still train if tests ran | N/A for “smoke OK” |
| **Return** | Some positive (test Δ / terminal +2) | All ~0 with empty tests |
| **Test output** | Real pytest pass/fail | `No module named pytest`, always empty, permission errors |
| **SyntaxError on solution** | Rare after `code_format=python` | Many → worker still emitting diffs |

**Smoke success ≠ high solve rate.** For `-n 20` you mainly want:

1. Run **finished** (≈20 episodes, not stuck after ~12).  
2. **Transitions &gt; 0**.  
3. Tests **actually ran** (p/f/e not always 0/0/0 with empty output).  
4. **No systemic harness bugs** (pytest missing, docker perms, pure-diff SyntaxError).

#### Practical bar: good enough to train on this buffer

- ≥ ~15 episodes completed  
- Transitions &gt; 0  
- 0 systemic pytest/permission errors  
- Some episodes with **non-zero** test counts  
- Ideally ≥1 positive return (not required)

**Do not train** on buffers full of pytest-missing / permission garbage — labels are wrong.

#### Decision tree

```text
-n 20 finished?
  ├─ no  → fix hang/timeouts; re-collect in chunks with unique -o
  └─ yes → tests real + transitions > 0?
              ├─ no  → fix docker/image/format; re-collect unique -o
              └─ yes → fugu-train smoke on this buffer
                         └─ then larger multi-strategy collect → real train → eval
```

#### What’s next after a healthy `-n 20`

**1. Train smoke** (CUDA torch required; proves SFT path — not paper quality):

```bash
fugu-train -c configs/default.yaml -b data/buffer_lcb_train
# adapter under outputs/planner/ (or training.output_dir)
```

**2. Scale collect** for real BC data (unique `-o` per chunk; optional strategies):

```bash
fugu-collect -d livecodebench-train -s single-qwen -n 10 -o data/buffer_lcb_qwen_a
fugu-collect -d livecodebench-train -s retry-on-fail -n 10 -o data/buffer_lcb_retry_a
fugu-collect -d livecodebench-train -s round-robin -n 10 -o data/buffer_lcb_rr_a
```

**3. Eval** after a non-toy train (same workers / budget as baselines):

```bash
fugu-eval -a outputs/planner/final_adapter -d livecodebench-val -n 10
# later: livecodebench-test
```

#### If analysis looks bad — fix before train

| Symptom | Likely fix |
|---------|------------|
| `No module named pytest` | Rebuild/use `fugu-py311-pytest`; `FUGU_ENV__DOCKER_IMAGE=...` |
| Permission denied under `/workspace` | `unset FUGU_ENV__DOCKER_USER` or `uid:gid` + chown |
| SyntaxError / empty solution | Confirm LCB path + `code_format=python` (post-LCB commits) |
| Always 0 tests | Dual harness / empty public_tests — check one dir under `/tmp/fugu_workspaces` |
| Hung mid-run | Chunk `-n 10`, raise timeouts, check vLLM logs |
| 0 transitions | Collect failed early — reread collect log / screen scrollback |

---

## Stage 7 — Optional SWE-bench (not primary)

```bash
fugu-collect -d swebench-lite -s single-qwen -n 1 -o data/buffer_swe_smoke
```

Needs working multi-repo test envs (or accept junk labels).  
Do **not** use `allow_host_execution=true` for untrusted remotes.

Official resolve rates: separate `pip install swebench` + `run_evaluation` (disk-heavy if scaled).

---

## Stage 8 — Train / eval planner

Only after Stage 6 **Analyze `-n 20`** looks healthy (or larger buffers).

```bash
fugu-train -c configs/default.yaml -b data/buffer_lcb_train
# fugu-eval -a outputs/planner/final_adapter -d livecodebench-val -n 10
# later: -d livecodebench-test
```

---

## Checklist

- [ ] `git pull` (6465c4f+ / LCB)
- [ ] `pip install -e ".[dev]"`; `import fugu`
- [ ] `pytest tests/` → **57** green
- [ ] `fugu-py311-pytest` built (after any image delete)
- [ ] Worker URLs; Fugu config shows docker + image
- [ ] `fugu-collect -d livecodebench-train -n 1` OK
- [ ] `-n 20` (or chunked) collect finished; **buffer summary script** healthy
- [ ] `fugu-train` smoke on that buffer
- [ ] Larger multi-strategy collect → real train → `fugu-eval`
- [ ] (Optional) planner CUDA load; pip freeze

---

## Related files

- [VM_START.md](VM_START.md) — short start  
- `configs/default.yaml` — workers, isolation, docker_image  
- `src/fugu/datasets/livecodebench.py` — LCB adapter + splits  
- `src/fugu/workspace/standalone.py` — no-git task dirs  
- `src/fugu/env/coding_env.py` — repo + standalone modes  
- `src/fugu/cli/collect.py` — entrypoint  
- `src/fugu/execution/runner.py` — docker test runner  
