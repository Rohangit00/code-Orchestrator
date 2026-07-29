# Start on the VM (Fugu)

Controlled smoke only. This is **not** authorization to run real third-party
SWE-bench collection on the host.

**Primary collect path:** LiveCodeBench (Python) — one Docker image with
`pytest` is enough. SWE-bench remains optional for repo-backed experiments.

Default config keeps `allow_host_execution: false` and `isolation_mode: host`.
For untrusted remote SWE-bench clones, set `isolation_mode: docker`.

For the full staged checklist, see [VM_RUNBOOK.md](VM_RUNBOOK.md).

---

## 1. Enter the repo and create a venv

```bash
cd code-Orchestrator   # or your clone path
git log -1 --oneline   # expect 64f5c9b / d573a6e or later — not the empty initial commit

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
```

---

## 2. PyTorch in this venv (read this)

### Do I have to install torch now?

| Goal | Need torch **in this venv**? |
|------|------------------------------|
| `pytest tests/` (current 38 unit tests) | Usually **no at runtime** — they do not load the planner |
| `pip install -e ".[dev]"` | **Yes as a dependency** — install will pull `torch` unless already satisfied here |
| Planner CUDA smoke / `fugu-train` | **Yes** |
| `fugu-collect` with external vLLM workers | Workers are separate processes; full package install still expects torch in the Fugu env |
| Real SWE-bench tests on host | **Blocked** — use `isolation_mode=docker` instead |

**System or another venv’s torch does not count.** A normal venv is isolated:
torch on the machine is invisible until it is installed into *this* `.venv`
(unless you created the venv with `--system-site-packages`, which is uncommon).

| Where torch lives | Visible after `source .venv/bin/activate`? |
|-------------------|--------------------------------------------|
| System Python | No |
| Another conda/venv | No |
| This `.venv` | Yes |

### Why install “now”?

- Not because the 38 unit tests need a GPU.
- Because this venv is isolated, and the project’s install path lists `torch`
  as a dependency (`pyproject.toml`).
- Because the next useful VM stages (planner 4-bit load, train) need torch +
  CUDA in the same env.
- Installing **CUDA** torch **before** `pip install -e ".[dev]"` avoids a
  CPU wheel from PyPI and a painful reinstall later.

### Check what’s already in the venv

```bash
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

| Result | What to do |
|--------|------------|
| Import works and CUDA is `True` | Fine — skip reinstall |
| `ModuleNotFoundError` | Install torch into this venv (below) |
| Import works but CUDA is `False` | Likely CPU torch — reinstall CUDA torch for H200 planner smoke |

### Recommended: CUDA torch, then Fugu

Match the wheel index to the VM driver/CUDA stack (example for CUDA 12.4):

```bash
source .venv/bin/activate
pip install -U pip wheel
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, 'cuda', torch.cuda.is_available())"
```

Adjust `cu124` to whatever matches the VM image.

`pip install -e ".[dev]"` installs the package + runtime deps + pytest/dev tools
so `import fugu` and the CLIs work without `PYTHONPATH`.

---

## 3. Sanity check

```bash
python -c "import fugu; print(fugu.__version__)"
which fugu-collect fugu-train fugu-eval
pytest tests/ -v
```

Expect **38 passed**.

---

## 4. Optional: planner GPU smoke (no workers)

```bash
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0))

from fugu.config import PlannerConfig
from fugu.planner.model import PlannerModel

cfg = PlannerConfig(
    base_model="Qwen/Qwen2.5-3B-Instruct",
    load_in_4bit=True,
)
m = PlannerModel(cfg)
m.load()
print("planner loaded (4-bit + LoRA)")
PY
```

This downloads weights if they are not cached. Ensure disk / `HF_HOME` is large
enough.

---

## 5. Optional: start vLLM workers yourself

Fugu does **not** start workers. Launch OpenAI-compatible servers on the
canonical ports (model IDs are placeholders):

```bash
# Terminal A — Qwen
python -m vllm.entrypoints.openai.api_server \
  --model <QWEN_MODEL_ID> \
  --port 8001

# Terminal B — Gemma
python -m vllm.entrypoints.openai.api_server \
  --model <GEMMA_MODEL_ID> \
  --port 8002

# Terminal C — Ornith
python -m vllm.entrypoints.openai.api_server \
  --model <ORNITH_MODEL_ID> \
  --port 8003
```

Health checks:

```bash
curl -s http://localhost:8001/v1/models | head
curl -s http://localhost:8002/v1/models | head
curl -s http://localhost:8003/v1/models | head
```

Config overrides if needed:

```bash
export FUGU_WORKER__QWEN_URL=http://localhost:8001/v1
export FUGU_WORKER__GEMMA_URL=http://localhost:8002/v1
export FUGU_WORKER__ORNITH_URL=http://localhost:8003/v1
```

Canonical defaults live in `configs/default.yaml` (8001 / 8002 / 8003).

---

## 6. Record the stack (once things work)

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip freeze > vm-requirements-freeze.txt
```

There is no project lockfile yet — keep this freeze for reproducibility.

---

## Minimum bar for “it works on the VM”

1. `pip install -e ".[dev]"` and `import fugu` works  
2. `pytest tests/ -v` green  
3. (Optional) CUDA + planner 4-bit load  
4. (Optional) vLLM on 8001–8003 and `/v1/models` OK  

---

## Collect smoke (LiveCodeBench — recommended)

One image with pytest is enough (no multi-repo SWE-bench stacks).

```bash
# Image with pytest
docker build -t fugu-py311-pytest - <<'EOF'
FROM python:3.11-slim
RUN pip install --no-cache-dir pytest
EOF

export FUGU_ENV__ISOLATION_MODE=docker
export FUGU_ENV__ALLOW_HOST_EXECUTION=false
export FUGU_ENV__DOCKER_IMAGE=fugu-py311-pytest
# + FUGU_WORKER__* URLs

fugu-collect \
  -c configs/default.yaml \
  -d livecodebench-train \
  -s single-qwen \
  -n 5 \
  -o data/buffer_lcb_train
```

Splits: `livecodebench-train` / `livecodebench-val` / `livecodebench-test` / `livecodebench` (all).

Optional SWE-bench: `-d swebench-lite` (harder; needs richer images).

---

## TL;DR

```bash
cd <repo>
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel
# Prefer CUDA torch first if planner/train smoke is next:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python -c "import fugu; print(fugu.__version__)"
pytest tests/ -v
```

Then continue with stages in [VM_RUNBOOK.md](VM_RUNBOOK.md) (planner smoke,
vLLM, later mock E2E). Do **not** enable host execution for untrusted remote
SWE-bench until a container executor exists.
