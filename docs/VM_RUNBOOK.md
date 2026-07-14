# H200 VM runbook — Fugu smoke path

**First time on the VM?** Use the shorter [VM_START.md](VM_START.md) (install,
torch in this venv, pytest, optional CUDA/vLLM). This runbook is the full
staged checklist.

This is a **controlled smoke-test** procedure. It is **not** authorization to run
untrusted third-party tests **on the host**. For remotes, use
`isolation_mode=docker` (Fugu sandbox — not the official SWE-bench harness).

---

## Not ready for full GPU-scale collection until

1. Code is on the VM via **git clone/pull of a commit that contains this tree**
   (not the empty initial commit alone). Include the docker-isolation changes.
2. Docker isolation is enabled for remotes (`isolation_mode=docker`) and a
   single-task smoke (`-n 1`) has been inspected.
3. CUDA-compatible **PyTorch / bitsandbytes / vLLM** versions are validated on
   the H200 image (for train; optional for collect-only).
4. **collect → buffer → train → eval** has been demonstrated at least at small scale.

Default config **must keep** `allow_host_execution: false` for remote repos.

---

## Shell sessions, exports, and `screen`

### Exports do not survive a closed terminal

`export FUGU_...` only lasts for **that shell session**. If the terminal exits
or you open a new SSH session, you must export again (or `source` a small env file).

Check:

```bash
echo $FUGU_ENV__ISOLATION_MODE
echo $FUGU_WORKER__QWEN_URL
```

Both should be non-empty before `fugu-collect`.

### Optional env file (recommended)

Create something like `env_vm.sh` on the VM (do not commit secrets):

```bash
export FUGU_WORKER__QWEN_URL=http://HOST:PORT/v1
export FUGU_WORKER__GEMMA_URL=http://HOST:PORT/v1
export FUGU_WORKER__ORNITH_URL=http://HOST:PORT/v1

export FUGU_ENV__ISOLATION_MODE=docker
export FUGU_ENV__ALLOW_HOST_EXECUTION=false
export FUGU_ENV__DOCKER_IMAGE=python:3.11-slim
```

Each session:

```bash
source .venv/bin/activate
source env_vm.sh
```

Worker URLs must end in **`/v1` only** (not `/v1/chat/completions`). Fugu appends
`/models` and `/chat/completions` itself.

### Use `screen` or `tmux` for long jobs

`fugu-collect` can run a long time (HF download, git clone, worker calls, docker
tests). If SSH drops, a normal shell dies with it.

**`screen` example:**

```bash
screen -S fugu
# inside:
source .venv/bin/activate
source env_vm.sh   # or re-export by hand
# ... run fugu-collect ...
```

| Action | Keys / command |
|--------|----------------|
| Detach | `Ctrl-A` then `D` |
| Reattach | `screen -r fugu` |
| List | `screen -ls` |

**`tmux` alternative:** `tmux new -s fugu` → detach `Ctrl-B D` → `tmux attach -t fugu`.

Also useful to keep **vLLM workers** in their own `screen`/`tmux` sessions if you
started them yourself.

For `-n 1` smoke, screen is optional but still nice if SSH is flaky. For larger
`-n`, use screen/tmux.

---

## Stage 0 — Transfer code

```bash
# On the VM
git clone <your-remote-with-fugu-commits> clever-turing
cd clever-turing
git log -1 --oneline   # confirm you are past the empty initial commit
git pull               # after docker-isolation commits land
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

### Torch must be in *this* venv

System or another env’s torch is **not** visible after `source .venv/bin/activate`
unless you used `--system-site-packages`.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

| Result | Action |
|--------|--------|
| Import + CUDA `True` | OK |
| `ModuleNotFoundError` | Install torch into this venv |
| Import but CUDA `False` | Likely CPU wheel — reinstall CUDA torch before full install |

Recommended order:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124  # match driver
pip install -e ".[dev]"
```

### Optional freeze file

There is **no** project lockfile. After a good stack:

```bash
pip freeze > vm-requirements-freeze.txt
```

This is only a notebook of what worked on this VM (repro / debug). Fugu does
not read this file at runtime.

---

## Stage 2 — Unit tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

Expect all tests green (currently **44** after docker isolation tests). No
`PYTHONPATH=src` required after editable install.

---

## Stage 3 — Planner GPU smoke (optional if collecting first)

**Skip Stage 3 if your goal is data collection first.** Collection uses fixed
strategies + workers, not the small planner. Download planner weights when you
are ready for `fugu-train`.

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

Fugu **never starts** workers. It only HTTP-calls OpenAI-compatible servers.

### If workers are already running

**Do not** re-run `python -m vllm.entrypoints...`. Only:

1. Health-check existing ports  
2. Point Fugu at them with exports or config  

```bash
curl -s http://HOST:PORT/v1/models | head
```

### If you must start workers yourself

`<QWEN_MODEL_ID>` is a **placeholder**: Hugging Face id (e.g.
`Qwen/Qwen2.5-Coder-32B-Instruct`) **or** a full local path to a model directory
on disk (must contain `config.json` + weights). Same idea for Gemma/Ornith.

```bash
# Needs `import vllm` in the Python you use (often a separate venv — Fugu does not install vLLM)
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model/or/HF_ID \
  --port 8001
```

Default ports in `configs/default.yaml` (override if yours differ):

| Worker | Default URL |
|--------|-------------|
| Qwen | `http://localhost:8001/v1` |
| Gemma | `http://localhost:8002/v1` |
| Ornith | `http://localhost:8003/v1` |

```bash
export FUGU_WORKER__QWEN_URL=http://HOST:PORT/v1
export FUGU_WORKER__GEMMA_URL=http://HOST:PORT/v1
export FUGU_WORKER__ORNITH_URL=http://HOST:PORT/v1
```

Or edit `configs/default.yaml` / a VM-specific YAML and pass `-c` to the CLI.

Confirm Fugu sees overrides:

```bash
python - <<'PY'
from fugu.config import FuguConfig
c = FuguConfig.from_yaml("configs/default.yaml")
print(c.worker.qwen_url, c.worker.gemma_url, c.worker.ornith_url)
print(c.env.isolation_mode, c.env.allow_host_execution, c.env.docker_image)
PY
```

---

## Stage 5 — Collect path overview

Pipeline order (research):

```text
1. Collect trajectories  ← strategies + workers (+ docker tests)
2. Replay buffer
3. Train planner         ← needs planner weights (Stage 3)
4. Eval
```

| Step | Needs planner weights? | Needs vLLM workers? | Needs docker isolation? |
|------|------------------------|---------------------|-------------------------|
| Collect (remote SWE-bench) | No | Yes | Yes (`isolation_mode=docker`) |
| Train | Yes | No | No |
| Eval with planner | Yes | Yes | Yes for remotes |

---

## Stage 6 — Real SWE-bench collect (docker sandbox)

### Fugu docker isolation vs official harness

| Fugu (`isolation_mode=docker`) | Official SWE-bench harness |
|--------------------------------|----------------------------|
| Sandbox for *our* test command / reward | Leaderboard-true resolve grading |
| Generic or user-chosen image (`docker_image`) | Per-repo/env prebuilt images |
| Enough to collect planner trajectories safely | Required for paper-faithful eval metrics |

### Do not run untrusted tests on the host

```yaml
isolation_mode: host
allow_host_execution: true   # unsafe for untrusted repos — never for GitHub clones
```

### After pulling docker-isolation code

```bash
cd <repo>
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v          # expect 44+

docker pull python:3.11-slim

# same shell / screen as collect
source env_vm.sh          # or re-export workers + isolation
```

### What `fugu-collect` does

```bash
fugu-collect \
  -c configs/default.yaml \
  -d swebench-lite \
  -s single-qwen \
  -n 1 \
  -o data/buffer_smoke
```

| Flag | Meaning |
|------|---------|
| `fugu-collect` | Drive coding env with fixed strategies; write transitions to a buffer |
| `-c configs/default.yaml` | Load YAML defaults; **env exports still override** workers / isolation |
| `-d swebench-lite` | Dataset: SWE-bench Lite (Hugging Face) |
| `-s single-qwen` | Strategy: CALL_QWEN → RETRY* → STOP (not all workers / not `all`). Must be this CLI name (not `single_worker_call_qwen`) |
| `-n 1` | **One task only** (smoke). Scale only after this looks sane |
| `-o data/buffer_smoke` | Output dir for `buffer.jsonl` |

Under the hood for that one task:

1. Load one Lite instance (issue text, repo, commit, test patch, …)  
2. Clone the GitHub repo at the given commit  
3. Baseline tests (**inside Docker** if `isolation_mode=docker`)  
4. Call the Qwen worker for a patch  
5. Apply patch → tests again in Docker  
6. Maybe RETRY, then STOP  
7. Write transitions into the buffer  

This is **behavioural-cloning data** for the planner — not official leaderboard eval.

Other strategies: `single-gemma`, `single-ornith`, `round-robin`, `retry-on-fail`,
`verify-first`, `all`.

### Recommended: run collect inside `screen`

```bash
screen -S fugu
source .venv/bin/activate
source env_vm.sh

fugu-collect \
  -c configs/default.yaml \
  -d swebench-lite \
  -s single-qwen \
  -n 1 \
  -o data/buffer_smoke
```

### After it finishes — inspect

```bash
ls -la data/buffer_smoke/
wc -l data/buffer_smoke/buffer.jsonl
head -c 500 data/buffer_smoke/buffer.jsonl
```

| Good signs | Not necessarily a bug |
|------------|------------------------|
| No `IsolationError` / no `NotImplementedError` | Tests fail inside container (slim image missing deps) |
| Logs show docker test execution | Worker returns a weak/empty patch |
| `buffer.jsonl` exists with lines | Low “solved” rate on first task |

### Then scale carefully

```bash
fugu-collect ... -n 5 -o data/buffer_v1
```

Keep `allow_host_execution=false`. Do not jump to full Lite until several tasks
produce clean trajectories.

### Common failures

| Symptom | Check |
|---------|--------|
| Empty exports after new terminal | Re-export or `source env_vm.sh` |
| `Docker binary not found` | `docker` on PATH; daemon running; `docker info` |
| Still host `IsolationError` | `echo $FUGU_ENV__ISOLATION_MODE` → must be `docker` |
| Worker timeouts / empty patches | `curl $FUGU_WORKER__QWEN_URL/models`; URL ends with `/v1` |
| `No module named vllm` when *starting* servers | Install/activate a venv that has vLLM (not required if workers already up) |
| HF / disk errors | network, `HF_HOME`, free disk |
| Mount permission errors | `export FUGU_ENV__DOCKER_USER="$(id -u):$(id -g)"` |

---

## Stage 7 — Train / eval (later)

Only after you have a non-empty buffer:

```bash
fugu-train -c configs/default.yaml -b data/buffer_smoke
# then, with adapter path:
# fugu-eval -a outputs/planner/final_adapter -d swebench-lite -n 1
```

Planner load needs Stage 3 (CUDA torch + base model weights).

---

## Checklist

- [ ] Repo pulled (includes docker isolation)
- [ ] `pip install -e ".[dev]"`; `import fugu` works
- [ ] `pytest tests/ -v` green (44+)
- [ ] Worker URLs exported or in config; `/v1/models` OK
- [ ] `isolation_mode=docker`; `allow_host_execution=false`
- [ ] `docker pull` for chosen image
- [ ] Exports (or `env_vm.sh`) re-applied in current shell / screen
- [ ] `fugu-collect ... -n 1` completed; buffer inspected
- [ ] (Optional) planner CUDA load for train
- [ ] Record `pip freeze` / CUDA stack if useful

---

## Related files

- [VM_START.md](VM_START.md) — shorter start + torch notes
- `configs/default.yaml` — endpoints, isolation, docker knobs
- `src/fugu/workers/pool.py` / `vllm.py` — assume servers already up; base URL ends in `/v1`
- `src/fugu/execution/runner.py` — host gate + docker executor
- `src/fugu/cli/collect.py` — `fugu-collect` entry point
- `task.md` / `implementation_issues.md` — open work
