"""CLI entry-point: ``fugu-eval`` — planner or fixed-strategy evaluation.

Two modes:

* **Planner** (after train)::

      fugu-eval -a outputs/planner/final_adapter -d livecodebench-val -n 50

* **Baseline strategy** (before train; no adapter)::

      fugu-eval -s single-ornith -d livecodebench-test -n 50 \\
        -o results/baseline_ornith_test.json

Primary dataset: LiveCodeBench (Python). SWE-bench variants remain available.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Protocol

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

console = Console()

_DATASET_MAP: dict[str, tuple[str, str, dict]] = {
    "livecodebench": (
        "fugu.datasets.livecodebench",
        "LiveCodeBenchDataset",
        {"split": "all", "python_only": True},
    ),
    "livecodebench-train": (
        "fugu.datasets.livecodebench",
        "LiveCodeBenchDataset",
        {"split": "train", "python_only": True},
    ),
    "livecodebench-val": (
        "fugu.datasets.livecodebench",
        "LiveCodeBenchDataset",
        {"split": "val", "python_only": True},
    ),
    "livecodebench-test": (
        "fugu.datasets.livecodebench",
        "LiveCodeBenchDataset",
        {"split": "test", "python_only": True},
    ),
    "swebench-lite": (
        "fugu.datasets.swebench",
        "SWEBenchDataset",
        {"split": "lite"},
    ),
    "swebench-full": (
        "fugu.datasets.swebench",
        "SWEBenchDataset",
        {"split": "full"},
    ),
    "swebench-verified": (
        "fugu.datasets.swebench",
        "SWEBenchDataset",
        {"split": "verified"},
    ),
}

_SUPPORTED_DATASETS = tuple(_DATASET_MAP.keys())

_STRATEGY_CLI_CHOICES = (
    "single-qwen",
    "single-ornith",
    "round-robin",
    "retry-on-fail",
    "verify-first",
)


class _ActionPolicy(Protocol):
    def select(self, state: Any, step: int) -> Any: ...


def _load_dataset(name: str):
    """Dynamically import and instantiate a benchmark dataset."""
    import importlib

    module_path, class_name, kwargs = _DATASET_MAP[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def _load_strategy(strategy_name: str):
    """Build one fixed strategy from a CLI name (same mapping as collect)."""
    from fugu.core.actions import PlannerAction
    from fugu.trajectory.strategies import (
        RoundRobinStrategy,
        RetryOnFailStrategy,
        SingleWorkerStrategy,
        VerifyFirstStrategy,
    )

    key = strategy_name.strip().lower().replace("_", "-")
    factories = {
        "single-qwen": lambda: SingleWorkerStrategy(
            PlannerAction.CALL_QWEN, max_retries=2
        ),
        "single-ornith": lambda: SingleWorkerStrategy(
            PlannerAction.CALL_ORNITH, max_retries=2
        ),
        "round-robin": RoundRobinStrategy,
        "retry-on-fail": lambda: RetryOnFailStrategy(
            primary_action=PlannerAction.CALL_ORNITH,
            max_retries=2,
            fallback_actions=[PlannerAction.CALL_QWEN],
        ),
        "verify-first": lambda: VerifyFirstStrategy(PlannerAction.CALL_ORNITH),
    }
    if key not in factories:
        raise click.ClickException(
            f"Unknown strategy {strategy_name!r}. "
            f"Choose from: {', '.join(_STRATEGY_CLI_CHOICES)}"
        )
    return factories[key]()


class _PlannerPolicy:
    """Wraps PlannerModel.predict for the shared episode loop."""

    def __init__(self, planner: Any) -> None:
        self._planner = planner

    def select(self, state: Any, step: int) -> Any:
        return self._planner.predict(state)


class _StrategyPolicy:
    """Wraps a fixed BaseStrategy."""

    def __init__(self, strategy: Any) -> None:
        self._strategy = strategy

    def select(self, state: Any, step: int) -> Any:
        return self._strategy.select_action(state, step)


async def _evaluate_task(env: Any, policy: _ActionPolicy, task: Any) -> dict:
    """Run one episode with *policy* choosing actions."""
    from collections import Counter

    state = env.reset(task)
    total_reward = 0.0
    steps = 0
    actions_taken: list[str] = []
    all_tests_passed = False
    max_steps = getattr(env, "_max_steps", 20)

    while steps < max_steps:
        action = policy.select(state, steps)
        actions_taken.append(action.name)

        state, reward, done, info = await env.step(action)
        total_reward += reward
        steps += 1
        all_tests_passed = info.get("all_tests_passed", False)

        if done:
            break

    env.close()

    action_counts = dict(Counter(actions_taken))
    return {
        "task_id": getattr(task, "task_id", "unknown"),
        "steps": steps,
        "total_reward": total_reward,
        "all_tests_passed": all_tests_passed,
        "actions": actions_taken,
        "action_counts": action_counts,
        "n_call_qwen": action_counts.get("CALL_QWEN", 0),
        "n_call_ornith": action_counts.get("CALL_ORNITH", 0),
        "final_pass_rate": (
            state.test_results.pass_rate
            if state is not None and state.test_results is not None
            else 0.0
        ),
    }


@click.command()
@click.option(
    "--config",
    "-c",
    default="configs/default.yaml",
    help="Path to YAML config file.",
)
@click.option(
    "--adapter",
    "-a",
    default=None,
    help="Path to trained LoRA adapter (planner mode). Omit for baseline -s.",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(list(_STRATEGY_CLI_CHOICES), case_sensitive=False),
    default=None,
    help="Fixed baseline strategy (no adapter). e.g. single-ornith, single-qwen.",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(_SUPPORTED_DATASETS, case_sensitive=False),
    default="livecodebench-val",
    help="Dataset: livecodebench[-train|-val|-test] or swebench-*.",
)
@click.option(
    "--max-tasks",
    "-n",
    type=int,
    default=None,
    help="Maximum number of tasks to evaluate (default: all).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output JSON path (default under results/ with mode + dataset stamp).",
)
def main(
    config: str,
    adapter: str | None,
    strategy: str | None,
    dataset: str,
    max_tasks: int | None,
    output: str | None,
) -> None:
    """Evaluate a planner adapter or a fixed baseline strategy."""
    if adapter and strategy:
        raise click.ClickException(
            "Pass either -a/--adapter (planner) or -s/--strategy (baseline), not both."
        )
    if not adapter and not strategy:
        raise click.ClickException(
            "Pass -a/--adapter for the learned planner, or -s/--strategy for a "
            "baseline (e.g. -s single-ornith)."
        )

    mode = "planner" if adapter else "baseline"
    console.print(
        Panel.fit(
            f"[bold blue]Fugu Evaluator[/bold blue] ({mode})",
            border_style="blue",
        )
    )

    from fugu.config import FuguConfig

    cfg = FuguConfig.from_yaml(config)

    if output is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        tag = "planner" if adapter else (strategy or "baseline")
        output = f"results/eval_{tag}_{dataset}_{stamp}.json"

    console.print(f"  Config    : [green]{config}[/green]")
    console.print(f"  Mode      : [green]{mode}[/green]")
    if adapter:
        console.print(f"  Adapter   : [green]{adapter}[/green]")
    if strategy:
        console.print(f"  Strategy  : [green]{strategy}[/green]")
    console.print(f"  Dataset   : [green]{dataset}[/green]")
    console.print(f"  Max tasks : [green]{max_tasks or 'all'}[/green]")
    console.print(f"  Output    : [green]{output}[/green]")
    console.print()

    # ── Policy ──────────────────────────────────────────────────────
    policy: _ActionPolicy
    if adapter:
        from fugu.planner.model import PlannerModel

        with console.status(
            f"[bold yellow]Loading model: {cfg.planner.base_model}…[/bold yellow]"
        ):
            planner = PlannerModel(cfg.planner)
            planner.load()
        with console.status(
            f"[bold yellow]Loading adapter: {adapter}…[/bold yellow]"
        ):
            planner.load_adapter(adapter)
        console.print("[green]✓[/green] Model and adapter loaded\n")
        policy = _PlannerPolicy(planner)
    else:
        strat = _load_strategy(strategy or "single-ornith")
        console.print(
            f"[green]✓[/green] Baseline strategy: "
            f"[cyan]{getattr(strat, 'name', strategy)}[/cyan]\n"
        )
        policy = _StrategyPolicy(strat)

    # ── Environment ─────────────────────────────────────────────────
    with console.status("[bold yellow]Initialising environment…[/bold yellow]"):
        from fugu.core.reward import RewardCalculator
        from fugu.env.coding_env import CodingEnvironment
        from fugu.execution.runner import TestRunner
        from fugu.repo.manager import RepoManager
        from fugu.workers.pool import WorkerPool

        worker_pool = WorkerPool.from_config(cfg.worker)
        repo_manager = RepoManager(
            workspace_dir=cfg.repo.workspace_dir,
            max_disk_mb=cfg.repo.max_disk_mb,
        )
        test_runner = TestRunner(
            timeout_seconds=cfg.env.test_timeout_seconds,
            isolation_mode=cfg.env.isolation_mode,
            allow_host_execution=cfg.env.allow_host_execution,
            docker_image=cfg.env.docker_image,
            docker_network=cfg.env.docker_network,
            docker_memory=cfg.env.docker_memory,
            docker_cpus=cfg.env.docker_cpus,
            docker_workdir=cfg.env.docker_workdir,
            docker_user=cfg.env.docker_user,
            docker_extra_args=cfg.env.docker_extra_args,
        )
        env = CodingEnvironment(
            worker_pool=worker_pool,
            repo_manager=repo_manager,
            test_runner=test_runner,
            reward_calculator=RewardCalculator(),
            max_steps=cfg.env.max_steps,
            cleanup_on_done=cfg.repo.cleanup_on_done,
        )

    console.print("[green]✓[/green] Environment initialised\n")

    # ── Dataset ─────────────────────────────────────────────────────
    with console.status(f"[bold yellow]Loading dataset '{dataset}'…[/bold yellow]"):
        ds = _load_dataset(dataset)

    tasks = list(ds)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    console.print(f"[green]✓[/green] Dataset loaded: {len(tasks)} tasks\n")

    # ── Run ─────────────────────────────────────────────────────────
    results: list[dict] = []
    start_time = time.monotonic()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        eval_task = progress.add_task("Evaluating tasks…", total=len(tasks))

        for i, task in enumerate(tasks):
            task_id = getattr(task, "task_id", f"task_{i}")
            progress.update(
                eval_task,
                description=f"Evaluating [cyan]{task_id}[/cyan]",
            )

            try:
                task_result = asyncio.run(_evaluate_task(env, policy, task))
                results.append(task_result)
            except Exception as exc:
                console.print(f"  [red]✗[/red] Task {task_id} failed: {exc}")
                results.append(
                    {
                        "task_id": task_id,
                        "steps": 0,
                        "total_reward": 0.0,
                        "all_tests_passed": False,
                        "actions": [],
                        "action_counts": {},
                        "n_call_qwen": 0,
                        "n_call_ornith": 0,
                        "final_pass_rate": 0.0,
                        "error": str(exc),
                    }
                )

            progress.advance(eval_task)

    elapsed = time.monotonic() - start_time

    # ── Aggregates ──────────────────────────────────────────────────
    total_tasks = len(results)
    passed_tasks = sum(1 for r in results if r["all_tests_passed"])
    pass_rate = passed_tasks / total_tasks if total_tasks else 0.0
    avg_reward = (
        sum(r["total_reward"] for r in results) / total_tasks if total_tasks else 0.0
    )
    avg_steps = (
        sum(r["steps"] for r in results) / total_tasks if total_tasks else 0.0
    )
    avg_pass_rate = (
        sum(r["final_pass_rate"] for r in results) / total_tasks
        if total_tasks
        else 0.0
    )
    avg_qwen = (
        sum(r.get("n_call_qwen", 0) for r in results) / total_tasks
        if total_tasks
        else 0.0
    )
    avg_ornith = (
        sum(r.get("n_call_ornith", 0) for r in results) / total_tasks
        if total_tasks
        else 0.0
    )

    action_counts: dict[str, int] = {}
    for r in results:
        for a in r.get("actions", []):
            action_counts[a] = action_counts.get(a, 0) + 1

    aggregate = {
        "mode": mode,
        "dataset": dataset,
        "adapter": adapter,
        "strategy": strategy,
        "total_tasks": total_tasks,
        "passed_tasks": passed_tasks,
        "task_pass_rate": pass_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_test_pass_rate": avg_pass_rate,
        "avg_call_qwen": avg_qwen,
        "avg_call_ornith": avg_ornith,
        "action_distribution": action_counts,
        "elapsed_seconds": elapsed,
    }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({"aggregate": aggregate, "per_task": results}, fh, indent=2, default=str)

    summary_table = Table(title="Evaluation Results", show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")
    summary_table.add_row("Mode", mode)
    summary_table.add_row("Dataset", dataset)
    if adapter:
        summary_table.add_row("Adapter", adapter)
    if strategy:
        summary_table.add_row("Strategy", strategy)
    summary_table.add_row("Total tasks", str(total_tasks))
    summary_table.add_row(
        "Tasks solved",
        f"{passed_tasks} / {total_tasks} ({pass_rate:.1%})",
    )
    summary_table.add_row("Avg reward", f"{avg_reward:.4f}")
    summary_table.add_row("Avg steps", f"{avg_steps:.1f}")
    summary_table.add_row("Avg CALL_QWEN / task", f"{avg_qwen:.2f}")
    summary_table.add_row("Avg CALL_ORNITH / task", f"{avg_ornith:.2f}")
    summary_table.add_row("Avg test pass rate", f"{avg_pass_rate:.1%}")
    summary_table.add_row("Elapsed time", f"{elapsed:.1f}s")

    console.print()
    console.print(summary_table)

    if action_counts:
        action_table = Table(title="Action Distribution")
        action_table.add_column("Action", style="bold")
        action_table.add_column("Count", justify="right", style="cyan")
        action_table.add_column("Percentage", justify="right", style="green")
        total_actions = sum(action_counts.values())
        for action_name, count in sorted(
            action_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = count / total_actions * 100 if total_actions else 0.0
            action_table.add_row(action_name, str(count), f"{pct:.1f}%")
        console.print()
        console.print(action_table)

    console.print(f"\n  Results saved to [green]{output_path}[/green]")
    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
