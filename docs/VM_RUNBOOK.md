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
| LCB **production** multi-strategy collect | Yes — Stage **6b** (unique `-o`, merge, train) |
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

### What `fugu-collect` does (LCB)

1. Load HF LiveCodeBench problems (train split if `-d livecodebench-train`)  
2. Create **standalone** workspace (no git clone)  
3. Baseline tests in Docker  
4. Worker returns **Python code** → write `solution.py`  
5. Re-run tests → reward → buffer  
6. RETRY / STOP per strategy  

| Flag | Meaning |
|------|---------|
| `-d livecodebench-train` | **Train split only** for BC data (default) |
| `-d livecodebench-val` / `-test` | Held-out eval — **do not** train on these |
| `-s single-ornith` | Cheap worker only (frugal prior) |
| `-s single-qwen` | Strong / expensive worker only |
| `-s retry-on-fail` | Ornith → RETRY* → escalate Qwen |
| `-s round-robin` | Ornith → Qwen → STOP |
| `-s verify-first` | VERIFY then worker |
| `-n N` | Max tasks (omit = entire split) |
| `-o DIR` | Buffer directory (**must be unique every run** — see below) |

**Active workers:** Qwen + Ornith only. `CALL_GEMMA` is disabled (enum kept; not in prompt/strategies/pool).

---

### Rule: unique `-o` every run

`-o` is the folder where that job writes `buffer.jsonl`.

**Reusing the same path overwrites** the previous buffer when the new job saves. A failed 3-task run can wipe a good 50-task buffer.

**Always use a timestamped path:**

```bash
-o "data/buffer_lcb_ornith_$(date +%Y%m%d_%H%M%S)"
```

Examples of good names:

```bash
-o "data/buffer_lcb_ornith_$(date +%Y%m%d_%H%M%S)"
-o "data/buffer_lcb_escalate_$(date +%Y%m%d_%H%M%S)"
-o "data/buffer_lcb_qwen_$(date +%Y%m%d_%H%M%S)"
```

After many runs you will have several directories under `data/`. Keep the good ones; merge later for train. `data/` is gitignored (may look grey in the IDE — use the terminal).

---

### Buffer format

- File is often named `buffer.jsonl` but content is **binary** (`FUGU_RB` + zstd).  
- Do **not** `cat` it; use `ReplayBuffer.load` (analyze section below).  

### Permissions

If you see `Permission denied` on files under `/workspace`:

```bash
unset FUGU_ENV__DOCKER_USER
# or: export FUGU_ENV__DOCKER_USER="$(id -u):$(id -g)"
```

LCB workspaces: `repo.workspace_dir` (default `/tmp/fugu_workspaces`) + `standalone/`.

### Success / failure (any collect)

| Good | Bad for training |
|------|------------------|
| Transitions &gt; 0 | `No module named pytest` |
| Real pytest output | Permission denied on mount |
| Some tests pass/fail counts | Always empty zero tests |

---

## Stage 6b — Production collect (real training data)

Smoke (`-n 1`) only proves the pipe. For **actual BC data**, collect **multi-strategy** trajectories on **`livecodebench-train`**, analyze each buffer, merge, then train.

### 1. Session setup

```bash
cd clever-turing
git pull
source .venv/bin/activate
source env_vm.sh   # HF_TOKEN, Qwen + Ornith URLs, docker isolation

# Confirm config (Gemma not required)
python - <<'PY'
from fugu.config import FuguConfig
from fugu.core.actions import ENABLED_WORKER_ACTIONS
c = FuguConfig.from_yaml("configs/default.yaml")
print("isolation:", c.env.isolation_mode, "image:", c.env.docker_image)
print("qwen:", c.worker.qwen_url)
print("ornith:", c.worker.ornith_url)
print("enabled:", sorted(a.name for a in ENABLED_WORKER_ACTIONS))
PY

curl -sS "${FUGU_WORKER__QWEN_URL}/models" | head -c 200; echo
curl -sS "${FUGU_WORKER__ORNITH_URL}/models" | head -c 200; echo

screen -S fugu   # long runs: Ctrl-A D to detach; screen -r fugu to reattach
```

### 2. What to collect (strategy mix)

| Strategy | Role |
|----------|------|
| `single-ornith` | Cheap default — frugal prior |
| `retry-on-fail` | Escalate Ornith → Qwen when cheap fails (**most important**) |
| `single-qwen` | Strong baseline / hard problems |
| `round-robin` | Both workers in one episode (optional) |

Do **not** only run `single-qwen`. That does not teach cheap-vs-strong routing.

### 3. Recommended first real run (`-n 50` per strategy)

Use **unique timestamped `-o` every command**. Raise `N` to 100–200 when stable.

```bash
# inside screen, venv + env_vm.sh already sourced
N=50

fugu-collect -c configs/default.yaml \
  -d livecodebench-train -s single-ornith -n $N \
  -o "data/buffer_lcb_ornith_$(date +%Y%m%d_%H%M%S)"

fugu-collect -c configs/default.yaml \
  -d livecodebench-train -s retry-on-fail -n $N \
  -o "data/buffer_lcb_escalate_$(date +%Y%m%d_%H%M%S)"

fugu-collect -c configs/default.yaml \
  -d livecodebench-train -s single-qwen -n $N \
  -o "data/buffer_lcb_qwen_$(date +%Y%m%d_%H%M%S)"

# optional
fugu-collect -c configs/default.yaml \
  -d livecodebench-train -s round-robin -n $N \
  -o "data/buffer_lcb_rr_$(date +%Y%m%d_%H%M%S)"
```

List what you got:

```bash
ls -lah data/buffer_lcb_*
```

### 4. Scale further (same rules)

| Goal | How |
|------|-----|
| More tasks per strategy | Larger `-n` (e.g. 100, 200) or omit `-n` for **entire** train split (long) |
| Avoid mid-run wipe | **Never** reuse `-o`; always `$(date +%Y%m%d_%H%M%S)` |
| Hangs / timeouts | Smaller `-n`; raise worker / docker test timeouts; check vLLM logs |
| Task index note | There is **no `--offset`**: `-n N` is always the **first N** train tasks. Different strategies on the same first N is fine (diverse labels on shared problems). To cover more problems, use a **larger** `-n` in one job. |

Example larger escalate collect:

```bash
fugu-collect -c configs/default.yaml \
  -d livecodebench-train -s retry-on-fail -n 200 \
  -o "data/buffer_lcb_escalate_$(date +%Y%m%d_%H%M%S)"
```

### 5. Analyze each buffer (before train)

Set `path` to **that run’s** `buffer.jsonl` (from `ls data/buffer_lcb_*`).

```bash
python - <<'PY'
from pathlib import Path
from collections import Counter
from fugu.buffer.replay_buffer import ReplayBuffer
from fugu.training.filter import group_episodes, episode_return, episode_solved

# EDIT to the folder you just collected
path = Path("data/buffer_lcb_ornith_YYYYMMDD_HHMMSS/buffer.jsonl")
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
print("CALL_GEMMA count (should be 0):",
      sum(1 for t in transitions if t.action.name == "CALL_GEMMA"))

eps = group_episodes(transitions)
print(f"\n=== EPISODES ===")
print(f"episodes: {len(eps)}")

solved = sum(1 for ep in eps if episode_solved(ep))
returns = [episode_return(ep) for ep in eps]
steps = [len(ep) for ep in eps]
print(f"solved: {solved}/{len(eps)}")
if returns:
    print(f"return: mean={sum(returns)/len(returns):.3f}  "
          f"min={min(returns):.3f}  max={max(returns):.3f}")
    print(f"steps/ep: mean={sum(steps)/len(steps):.1f}  max={max(steps)}")

print("\n#  task_id                    steps   return  solved  p/f/e  tail")
for i, ep in enumerate(eps):
    last = ep[-1]
    ta = last.metadata.tests_after
    p = f = e = 0
    out = ""
    if ta is not None:
        p, f, e = ta.passed, ta.failed, ta.errors
        out = (ta.output or "")[:80].replace("\n", " ")
    tid = getattr(last.state, "task_id", "") or "?"
    print(f"{i:2d}  {str(tid)[:24]:24s}  {len(ep):3d}  {episode_return(ep):7.2f}  "
          f"{episode_solved(ep)!s:5s}  {p}/{f}/{e}  {out}")

bad = 0
for t in transitions:
    ta = t.metadata.tests_after
    if ta and ("No module named pytest" in (ta.output or "")
               or "Permission denied" in (ta.output or "")):
        bad += 1
print(f"\npytest-missing or permission errors: {bad}")
PY
```

#### How to read the numbers

| Signal | Healthy | Unhealthy |
|--------|---------|-----------|
| **Episodes** | ≈ your `-n` | 0, or stuck mid-run |
| **Transitions** | &gt; 0 | 0 |
| **Actions** | Match strategy (escalate: Ornith + some Qwen) | Only STOP; unexpected `CALL_GEMMA` |
| **Solved rate** | Can be low on hard LCB | N/A for “collect OK” |
| **Test output** | Real pytest p/f/e | pytest missing / always empty / permission |

**Collect OK ≠ high solve rate.** Require: run finished, tests real, no systemic harness bugs.

#### Fix before train if bad

| Symptom | Fix |
|---------|-----|
| `No module named pytest` | Rebuild `fugu-py311-pytest`; set `FUGU_ENV__DOCKER_IMAGE` |
| Permission denied | `unset FUGU_ENV__DOCKER_USER` or set `uid:gid` |
| Hung mid-run | Smaller `-n`, raise timeouts, check vLLM; re-run with **new** `-o` |
| 0 transitions | Read collect log / screen scrollback |

### 6. Merge good buffers for training

`fugu-train -b DIR` loads **one** directory. Merge several healthy runs into a new unique folder:

```bash
python - <<'PY'
from pathlib import Path
from fugu.buffer.replay_buffer import ReplayBuffer

# EDIT: list only the buffer dirs you verified as healthy
sources = [
    "data/buffer_lcb_ornith_YYYYMMDD_HHMMSS",
    "data/buffer_lcb_escalate_YYYYMMDD_HHMMSS",
    "data/buffer_lcb_qwen_YYYYMMDD_HHMMSS",
]
# Unique merge output (do not overwrite older merges)
from datetime import datetime
out = Path("data/buffer_lcb_train_merged_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
out.mkdir(parents=True, exist_ok=True)

merged = ReplayBuffer(capacity=200_000, storage_dir=str(out))
for d in sources:
    p = Path(d) / "buffer.jsonl"
    if not p.exists():
        print("skip missing", p)
        continue
    b = ReplayBuffer(capacity=200_000, storage_dir=d)
    b.load(str(p))
    for t in b:
        merged.add(t)
    print(d, "->", len(b), "transitions")
merged.save(str(out / "buffer.jsonl"))
print("MERGED", len(merged), "->", out / "buffer.jsonl")
print("Train with: fugu-train -c configs/default.yaml -b", out)
PY
```

### 7. Train on the merged buffer

```bash
fugu-train -c configs/default.yaml -b data/buffer_lcb_train_merged_YYYYMMDD_HHMMSS
# adapter under outputs/planner/ (or training.output_dir)
```

### 8. Eval baselines (before train) and planner (after train)

**Data leak:** Running baselines on **`livecodebench-test`** does **not** train the planner or fine-tune workers. Workers are **inference-only**. As long as you **do not** put test/val trajectories into the **train buffer** for SFT, there is **no planner leakage**. (Separate issue: worker pretraining contamination is what LCB time-splits address; not caused by your eval.)

**Do not** merge baseline eval runs into `buffer_lcb_train_*`. Eval writes **JSON results**, not training buffers (unless you deliberately collect on test).

#### Baselines (no adapter — run **before** train)

Same workers, docker, `-n`, and split for every method. Prefer **val** first; **test** for final numbers.

```bash
# Unique -o each run
N=50
SPLIT=livecodebench-val   # or livecodebench-test for final baselines

fugu-eval -c configs/default.yaml -s single-ornith -d $SPLIT -n $N \
  -o "results/baseline_ornith_${SPLIT}_$(date +%Y%m%d_%H%M%S).json"

fugu-eval -c configs/default.yaml -s single-qwen -d $SPLIT -n $N \
  -o "results/baseline_qwen_${SPLIT}_$(date +%Y%m%d_%H%M%S).json"

fugu-eval -c configs/default.yaml -s retry-on-fail -d $SPLIT -n $N \
  -o "results/baseline_escalate_${SPLIT}_$(date +%Y%m%d_%H%M%S).json"

# optional
fugu-eval -c configs/default.yaml -s round-robin -d $SPLIT -n $N \
  -o "results/baseline_rr_${SPLIT}_$(date +%Y%m%d_%H%M%S).json"
```

Each JSON has `aggregate.task_pass_rate`, `avg_call_qwen`, `avg_call_ornith`, `avg_steps`, etc.

#### Learned planner (after train)

```bash
fugu-eval -c configs/default.yaml \
  -a outputs/planner/final_adapter \
  -d livecodebench-val -n 50 \
  -o "results/planner_val_$(date +%Y%m%d_%H%M%S).json"
# later: -d livecodebench-test
```

Compare planner vs baselines on **success** and **avg CALL_QWEN** (cost proxy).

### Decision tree (production)

```text
Smoke -n 1 OK?
  └─ yes → Production collect (Stage 6b)
              ├─ multi-strategy, unique -o each run
              ├─ analyze each buffer
              ├─ merge healthy dirs
              ├─ fugu-train on merge
              └─ fugu-eval on val/test
```

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
- [ ] Production collect (Stage 6b): multi-strategy, **unique `-o` with `$(date …)` every run**
- [ ] Analyze each buffer; merge healthy ones
- [ ] Baseline `fugu-eval -s …` on val (and optionally test) **before** train
- [ ] Merge train buffers only (never test/val collect into train)
- [ ] `fugu-train` on merged **train** buffer
- [ ] `fugu-eval -a …` planner on val/test vs baseline JSONs
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
