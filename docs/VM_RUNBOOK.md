# H200 VM runbook — Fugu smoke path

**First time on the VM?** Use the shorter [VM_START.md](VM_START.md) (install,
torch in this venv, pytest, optional CUDA/vLLM). This runbook is the full
staged checklist.

This is a **controlled smoke-test** procedure. It is **not** authorization to run
real third-party SWE-bench collection on the host.

## Not ready for real GPU-scale collection until

1. Code is on the VM via **git clone/pull of a commit that contains this tree**
   (not the empty initial commit alone).
2. Docker (or other) isolation is **implemented** — today
   `isolation_mode=docker` raises `NotImplementedError`.
3. CUDA-compatible **PyTorch / bitsandbytes / vLLM** versions are validated on
   the H200 image.
4. Mock **collect → buffer → train → eval** has been demonstrated on the VM.

Default config **must keep** `allow_host_execution: false` for remote repos.

---

## Stage 0 — Transfer code

```bash
# On the VM
git clone <your-remote-with-fugu-commits> clever-turing
cd clever-turing
git log -1 --oneline   # confirm you are past the empty initial commit
```

If git remote is not set up yet, copy a **deliberate archive** of a committed
tree (not a half-synced working copy).

---

## Stage 1 — Python env + package install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel

# Prefer CUDA wheel index matching the VM driver (example — adjust for image):
# pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -e ".[dev]"

# Confirm import without PYTHONPATH
python -c "import fugu; print(fugu.__version__)"
which fugu-collect fugu-train fugu-eval
```

There is **no lockfile** yet. Record the successful stack on the VM:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
pip freeze > vm-requirements-freeze.txt
```

Validate before large downloads: **torch CUDA**, **bitsandbytes**, **transformers**,
**peft**, and **vLLM** versions known to work on H200.

---

## Stage 2 — Unit tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expect all tests green (no `PYTHONPATH=src` required after editable install).

---

## Stage 3 — Planner GPU smoke (no workers required)

```bash
source .venv/bin/activate
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print("device:", torch.cuda.get_device_name(0))

from fugu.config import PlannerConfig
from fugu.planner.model import PlannerModel

cfg = PlannerConfig(
    base_model="Qwen/Qwen2.5-3B-Instruct",  # downloads weights if not cached
    load_in_4bit=True,
)
m = PlannerModel(cfg)
m.load()
print("planner loaded (4-bit + LoRA)")
PY
```

Ensure HF cache / disk budget is large enough (`HF_HOME` / `TRANSFORMERS_CACHE`).

---

## Stage 4 — vLLM workers (manual; not managed by Fugu)

Fugu’s `WorkerPool.from_config` **assumes** OpenAI-compatible servers already
listen on:

| Worker | Default URL |
|--------|-------------|
| Qwen | `http://localhost:8001/v1` |
| Gemma | `http://localhost:8002/v1` |
| Ornith | `http://localhost:8003/v1` |

Example launch **shape** (replace model IDs with the ones provisioned on the VM):

```bash
# Terminal A — Qwen coding model (example IDs only)
python -m vllm.entrypoints.openai.api_server \
  --model <QWEN_MODEL_ID> \
  --port 8001 \
  --tensor-parallel-size 1

# Terminal B
python -m vllm.entrypoints.openai.api_server \
  --model <GEMMA_MODEL_ID> \
  --port 8002 \
  --tensor-parallel-size 1

# Terminal C
python -m vllm.entrypoints.openai.api_server \
  --model <ORNITH_MODEL_ID> \
  --port 8003 \
  --tensor-parallel-size 1
```

Health checks:

```bash
curl -s http://localhost:8001/v1/models | head
curl -s http://localhost:8002/v1/models | head
curl -s http://localhost:8003/v1/models | head
```

Point config at the host if needed:

```bash
export FUGU_WORKER__QWEN_URL=http://localhost:8001/v1
export FUGU_WORKER__GEMMA_URL=http://localhost:8002/v1
export FUGU_WORKER__ORNITH_URL=http://localhost:8003/v1
```

---

## Stage 5 — Mock end-to-end (local / toy only)

Do **not** point collection at real remote SWE-bench repos until Docker
isolation exists.

Preferred smoke order:

1. **Unit suite** (Stage 2) — done.
2. **Buffer + train smoke** using synthetic/mock transitions (no remote clone), e.g.
   a small script or future `fugu-collect` path with mock workers + **local**
   `file://` / path repos only.
3. **`fugu-train`** on a tiny buffer for a few steps.
4. **`fugu-eval`** only when a planner adapter exists and the env is safe.

CLI entry points after install:

```bash
fugu-collect --help
fugu-train --help
fugu-eval --help
```

Collection currently requires a real `CodingEnvironment` reset (git repo +
`repo_url`). Mock-worker pool is available in code (`WorkerPool.mock()`); a
dedicated mock CLI flag may still need wiring for one-command smoke.

Until that flag exists, treat **pytest + planner CUDA load + vLLM /v1/models**
as the minimum VM smoke bar.

---

## Stage 6 — Real SWE-bench (blocked)

**Do not run** real third-party SWE-bench collection with:

```yaml
isolation_mode: host
allow_host_execution: true   # unsafe for untrusted repos
```

Required first:

1. Implement container executor in `TestRunner` (`isolation_mode=docker`).
2. Set `allow_host_execution: false` and `isolation_mode: docker`.
3. Re-run a single Lite task under isolation; inspect trajectories manually.
4. Only then scale collect → train → eval.

---

## Checklist

- [ ] Repo committed and cloned (full tree, not empty initial commit only)
- [ ] `pip install -e ".[dev]"`; `import fugu` works
- [ ] `pytest tests/ -v` green
- [ ] `torch.cuda.is_available()` true on H200
- [ ] Planner 4-bit load succeeds
- [ ] vLLM on 8001/8002/8003; `/v1/models` OK
- [ ] Mock / local E2E path demonstrated
- [ ] Docker isolation implemented **before** real SWE-bench
- [ ] Record `pip freeze` / CUDA stack for reproducibility

---

## Related files

- `configs/default.yaml` — endpoints, isolation flags
- `src/fugu/workers/pool.py` — assumes vLLM already up
- `src/fugu/execution/runner.py` — isolation gate / docker stub
- `task.md` — incomplete integration checkboxes
