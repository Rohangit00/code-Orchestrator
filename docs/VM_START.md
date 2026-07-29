# Start on the VM (Fugu)

**Primary path:** LiveCodeBench (Python) → buffer → train planner.  
**One Docker image with `pytest` is enough** (no 300 SWE-bench images).

SWE-bench remains optional for repo-backed experiments (harder envs).

Code tip should include LiveCodeBench (`6465c4f` or later on `main`).

For the long checklist see [VM_RUNBOOK.md](VM_RUNBOOK.md).

---

## 0. After deleting Docker images

Rebuild the test image before collect:

```bash
docker build -t fugu-py311-pytest - <<'EOF'
FROM python:3.11-slim
RUN pip install --no-cache-dir pytest
EOF

docker run --rm fugu-py311-pytest python -m pytest --version
docker images
```

Optional cleanup of junk:

```bash
docker image prune -f
# or: docker rmi <old-image>
```

---

## 1. Repo + venv

```bash
cd code-Orchestrator   # or your clone path
git pull
git log -1 --oneline   # expect 6465c4f or later (LCB commit)

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
```

---

## 2. PyTorch in *this* venv

System / other-venv torch does **not** count.

| Goal | Need torch here? |
|------|------------------|
| `pytest tests/` (57 unit tests) | Usually no at runtime |
| `pip install -e ".[dev]"` | Yes (declared dependency) |
| `fugu-train` / planner load | Yes (CUDA on H200) |
| Collect only (workers external) | Package still pulls torch; workers are separate |

```bash
# Prefer CUDA torch first on H200:
pip install torch --index-url https://download.pytorch.org/whl/cu124   # match driver
pip install -e ".[dev]"

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import fugu; print(fugu.__version__)"
```

---

## 3. Unit smoke (always)

```bash
pytest tests/ -v
# expect 57 passed
which fugu-collect fugu-train fugu-eval
```

---

## 4. Workers (Fugu does not start them)

If already running, only set URLs (`…/v1`, not `…/v1/chat/completions`).

```bash
export FUGU_WORKER__QWEN_URL=http://HOST:PORT/v1
export FUGU_WORKER__GEMMA_URL=http://HOST:PORT/v1
export FUGU_WORKER__ORNITH_URL=http://HOST:PORT/v1

curl -s "$FUGU_WORKER__QWEN_URL/models" | head
```

Optional: copy `env_vm.sh.example` → **`env_vm.sh`** (gitignored), fill placeholders,
and `source env_vm.sh` **only inside your screen** so tokens stay in that session:

```bash
cp env_vm.sh.example env_vm.sh
# edit HF_TOKEN=hf_... and worker URLs in env_vm.sh (not committed)
screen -S fugu
source .venv/bin/activate
source env_vm.sh
# export HF_TOKEN=...   # or set inside env_vm.sh
```

`HF_TOKEN` / `FUGU_HF_TOKEN` → authenticated Hub downloads (higher rate limits).  
Exports die when that shell/screen ends.

---

## 5. Docker + isolation for collect

```bash
export FUGU_ENV__ISOLATION_MODE=docker
export FUGU_ENV__ALLOW_HOST_EXECUTION=false
export FUGU_ENV__DOCKER_IMAGE=fugu-py311-pytest
# leave docker_user empty unless you hit mount permission issues
# export FUGU_ENV__DOCKER_USER="$(id -u):$(id -g)"  # only if needed + chown workspace

python - <<'PY'
from fugu.config import FuguConfig
c = FuguConfig.from_yaml("configs/default.yaml")
print("isolation_mode:", c.env.isolation_mode)   # must be docker
print("docker_image:", c.env.docker_image)       # fugu-py311-pytest
print("qwen:", c.worker.qwen_url)
PY
```

`echo` alone is not enough — always check via `FuguConfig.from_yaml`.

---

## 6. LiveCodeBench smoke (`-n 1`)

```bash
fugu-collect \
  -c configs/default.yaml \
  -d livecodebench-train \
  -s single-qwen \
  -n 1 \
  -o "data/buffer_lcb_smoke_$(date +%Y%m%d_%H%M%S)"
```

| Flag | Meaning |
|------|---------|
| `-d livecodebench-train` | Train split of LCB (Python contest problems) |
| `-s single-qwen` | CALL_QWEN → RETRY* → STOP |
| `-n 1` | One task smoke |
| `-o …` | New dir each run (avoid overwriting good buffers) |

First run may download the HF LiveCodeBench dataset (small vs SWE images).

**Success:** transitions &gt; 0; test output is **not** `No module named pytest`.

### Dataset names

| Name | Use |
|------|-----|
| `livecodebench-train` | collect / train (default collect) |
| `livecodebench-val` | validation |
| `livecodebench-test` | held-out eval |
| `livecodebench` | all |
| `swebench-lite` etc. | optional repo path |

### Buffer file

`buffer.jsonl` is **binary** (FUGU_RB), not text JSON. Inspect with:

```bash
python - <<'PY'
from pathlib import Path
from fugu.buffer.replay_buffer import ReplayBuffer
# set path to your buffer.jsonl
path = Path("data/buffer_lcb_smoke_XXXX/buffer.jsonl")
buf = ReplayBuffer(capacity=10000, storage_dir=str(path.parent))
buf.load(str(path))
print("transitions:", len(buf))
PY
```

`data/` is gitignored (often grey in IDEs) — use terminal `ls`.

---

## 7. Scale collect → train

```bash
fugu-collect -d livecodebench-train -s single-qwen -n 20 -o data/buffer_lcb_train
fugu-train -c configs/default.yaml -b data/buffer_lcb_train
```

Planner load needs CUDA torch (step 2). Skip planner GPU smoke until train.

---

## Optional: planner GPU smoke (before train)

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available()
from fugu.config import PlannerConfig
from fugu.planner.model import PlannerModel
m = PlannerModel(PlannerConfig(base_model="Qwen/Qwen2.5-3B-Instruct", load_in_4bit=True))
m.load()
print("planner ok", torch.cuda.get_device_name(0))
PY
```

---

## Optional: SWE-bench

Still available (`-d swebench-lite`) but needs richer/per-repo images for real tests.  
Do **not** set `allow_host_execution=true` for untrusted GitHub clones.

---

## Minimum bar

1. `git pull` past LCB commit; `pip install -e ".[dev]"`  
2. `pytest tests/` → **57** green  
3. `fugu-py311-pytest` image built  
4. Workers healthy; Fugu config shows docker + image  
5. `fugu-collect -d livecodebench-train -n 1` → buffer with transitions  
6. (Next) larger `-n` + `fugu-train`  

---

## TL;DR

```bash
git pull && source .venv/bin/activate && pip install -e ".[dev]"
pytest tests/ -q
docker build -t fugu-py311-pytest - <<'EOF'
FROM python:3.11-slim
RUN pip install --no-cache-dir pytest
EOF
export FUGU_ENV__ISOLATION_MODE=docker
export FUGU_ENV__DOCKER_IMAGE=fugu-py311-pytest
export FUGU_WORKER__QWEN_URL=http://HOST:PORT/v1
fugu-collect -d livecodebench-train -s single-qwen -n 1 \
  -o data/buffer_lcb_smoke_$(date +%Y%m%d_%H%M%S)
```
