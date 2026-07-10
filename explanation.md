# Fugu Orchestrator: Project Explanation & Methodology

## 1. What We Are Trying to Do

We are building a **Coding-Oriented LLM Orchestrator** inspired by the Sakana Fugu and TRINITY architectures. 

Instead of relying on a single massive language model or hard-coded "if/else" routing rules to solve complex software engineering tasks (like those in SWE-bench), we are training a **small, lightweight planner model (1B–3B parameters)**. 

This planner does **not** write code itself. Its sole job is to act as an intelligent project manager. It learns a policy to coordinate a pool of larger, specialized "worker" models (like Qwen 27B, Gemma 27B/31B, and Ornith). The orchestrator decides:
- **Who** to delegate work to (which worker model is best suited for the current state).
- **When** to run tests or verify the code.
- **When** to retry a failed attempt.
- **When** to stop because the task is complete.

The ultimate goal is to maximize coding benchmark performance (tests passed) while minimizing latency, token usage, and computational cost.

---

## 2. Theory: Orchestration as a Markov Decision Process (MDP)

Traditional multi-agent frameworks often use static prompt chains. We formulate orchestration dynamically as a **Markov Decision Process (MDP)**, bringing principles from Reinforcement Learning (RL) into LLM orchestration.

An MDP consists of States ($\mathcal{S}$), Actions ($\mathcal{A}$), Transitions ($\mathcal{T}$), and Rewards ($\mathcal{R}$).

### State ($\mathcal{S}$)
The "state" is the current context of the problem, which the planner uses to make decisions. It includes:
- The initial issue / bug description.
- A summary of the repository (file tree, relevant files).
- The history of previous actions and their outcomes.
- Current test results (pass/fail/error counts) and compilation status.
- Remaining computational budget (tokens/time).

### Action Space ($\mathcal{A}$)
At each step, the planner chooses one discrete action from a defined space:
- `CALL_QWEN`, `CALL_GEMMA`, `CALL_ORNITH` (Delegate to a worker to generate a patch).
- `RUN_TESTS` (Execute the test suite).
- `VERIFY` (Run full tests plus syntax/compilation checks).
- `RETRY` (Send the exact same context + the recent error trace back to the last worker).
- `STOP` (Terminate the workflow).

### Transitions ($\mathcal{T}$)
When an action is taken, the environment transitions to a new state. For example, if the planner chooses `CALL_QWEN`, the Qwen worker generates a patch, the patch is applied to the codebase, tests are run, and the state updates with the new repository diff and test results.

### Reward Function ($\mathcal{R}$)
The system evaluates the planner's decisions using a composite reward function:
- **Positive rewards:** Increasing the test pass rate; achieving a successful compilation; a large terminal bonus for fully resolving the issue.
- **Negative penalties:** Token usage costs; latency delays; excessive retries.

---

## 3. Methodology & Implementation Architecture

The project is structured into three main phases: Environment Construction, Data Collection, and Planner Fine-Tuning.

### Phase 1: The Coding Environment (Gymnasium-style)
We treat the code repository as an interactive environment, similar to OpenAI Gym used in RL.
- **Repo Manager:** Handles highly efficient git operations. To stay within disk limits (~10 GB), it clones only one repository at a time, checks out the specific bug commit, applies patches, and deletes the repo when the task is done.
- **Test Runner:** Executes test commands in isolated subprocesses and parses outputs (like `pytest` results) to update the state.
- **Worker Pool:** Abstracts the LLMs behind a common interface. Workers are accessed via OpenAI-compatible HTTP APIs (running on vLLM/H200s).

### Phase 2: Trajectory Generation & Replay Buffer
Before we can train the planner, we need data showing *how* to orchestrate. 
- We use **fixed strategies** (e.g., a "Retry-on-Fail" strategy, or a "Multi-Worker Fallback" strategy) to run tasks from datasets like SWE-bench Lite and HumanEval.
- As these strategies interact with the environment, we record every step as a `Transition` (State, Action, Reward, Next State).
- These transitions are heavily compressed (using `zstandard` and `msgpack`) and stored in a fixed-size, memory-mapped **Replay Buffer** to keep disk usage strictly under 2–4 GB.

### Phase 3: Training the Planner (QLoRA)
We use a small instruction-tuned model (`Qwen/Qwen2.5-3B-Instruct`) as the base for our planner.
- **State Tokenization:** The complex data structures of the `PlannerState` are serialized into a clean, XML-tagged text prompt (e.g., `<|tests|>passed: 5, failed: 2<|/tests|>`).
- **Supervised Fine-Tuning (SFT):** Initially, we train the planner using standard cross-entropy loss. We frame it as a sequence classification or constrained generation task where the model learns to predict the optimal `PlannerAction` based on the successful trajectories in the replay buffer.
- **Efficiency:** Training utilizes **QLoRA** (Quantized Low-Rank Adaptation). By quantizing the base model to 4-bit (NF4) and only training small adapter weights, we can comfortably fine-tune a 3B model on a single GPU.

### Future: Reinforcement Learning
Once the planner has a solid supervised baseline, the architecture supports moving to online Reinforcement Learning (like PPO or REINFORCE). The planner will explore the action space itself, generating its own trajectories, and updating its policy to maximize the custom reward function directly.

---

## 4. Summary of Benefits

By separating the **orchestration logic** (the Planner) from the **coding capability** (the Workers), this system achieves:
1. **Cost Efficiency:** Simple tasks are routed to cheaper/faster models. Tests are run only when necessary.
2. **Resilience:** If one worker fails or hallucinates, the planner can learn a policy to retry or seamlessly fallback to a different model.
3. **Modularity:** Worker models can be swapped out, upgraded, or expanded without having to retrain the core orchestration logic.

---

## 5. Project Structure

The codebase is organized modularly to support the MDP and Phase 1-3 workflows:

```text
src/fugu/
├── core/             # Core MDP formalisms (PlannerAction, PlannerState, RewardCalculator)
├── workers/          # LLM integrations (BaseWorker, VLLMWorker, WorkerPool)
├── repo/             # Git and disk management (RepoManager, RepoContext)
├── execution/        # TestRunner (authoritative test_command; isolation gate)
├── env/              # Gymnasium-style CodingEnvironment (auto-terminate on solve)
├── datasets/         # SWE-bench (primary); HumanEval/MBPP inspection-only for now
├── trajectory/       # Fixed strategies + TrajectoryCollector
├── buffer/           # zstd + msgpack ReplayBuffer
├── planner/          # QLoRA planner, tokenizer, shared prompts.py
├── training/         # SFT Trainer, TransitionDataset, episode filter
└── cli/              # fugu-collect, fugu-train, fugu-eval (SWE-bench only)
```

**Status note:** Core scaffold and correctness fixes (issues 1–9, 11 light, 12 gate,
14 basic, 15–17) are in place; see `implementation_issues.md` and `task.md`.
Real third-party SWE-bench still needs a container executor (`isolation_mode=docker`).
