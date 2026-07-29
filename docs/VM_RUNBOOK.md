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
| LCB train from buffer | Yes after non-empty buffer + CUDA torch |
| Full SWE-bench Lite @ official harness | Separate tool (`swebench`); lots of disk if scaled |
| Host execution of untrusted git tests | No — keep `allow_host_execution: false` |

---

## Shell sessions, exports, screen

Exports die when the terminal exits. Re-export or:

```bash
# env_vm.sh (do not commit secrets)
export FUGU_WORKER__QWEN_URL=http://HOST:PORT/v1
export FUGU_WORKER__GEMMA_URL=http://HOST:PORT/v1
export FUGU_WORKER__ORNITH_URL=http://HOST:PORT/v1
export FUGU_ENV__ISOLATION_MODE=docker
export FUGU_ENV__ALLOW_HOST_EXECUTION=false
export FUGU_ENV__DOCKER_IMAGE=fugu-py311-pytest
```

```bash
source .venv/bin/activate
source env_vm.sh
```

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

### Scale then train

```bash
fugu-collect -d livecodebench-train -s single-qwen -n 20 -o data/buffer_lcb_train
fugu-train -c configs/default.yaml -b data/buffer_lcb_train
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
| `-s single-qwen` | CLI name (not `single_worker_call_qwen`) |
| `-n 1` | Smoke |
| `-o dir` | Buffer directory (use unique names) |

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

```bash
fugu-train -c configs/default.yaml -b data/buffer_lcb_train
# fugu-eval -a outputs/planner/final_adapter -d livecodebench-test -n 10
```

---

## Checklist

- [ ] `git pull` (6465c4f+ / LCB)
- [ ] `pip install -e ".[dev]"`; `import fugu`
- [ ] `pytest tests/` → **57** green
- [ ] `fugu-py311-pytest` built (after any image delete)
- [ ] Worker URLs; Fugu config shows docker + image
- [ ] `fugu-collect -d livecodebench-train -n 1` OK
- [ ] Larger collect + `fugu-train`
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
