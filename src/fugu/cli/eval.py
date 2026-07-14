"""CLI entry-point: ``fugu-eval`` — planner evaluation.

Evaluates a trained planner adapter against a benchmark dataset by
running episodes where the planner's predicted actions drive the coding
environment, then aggregates and reports performance metrics.

Usage::

    fugu-eval -a outputs/planner/final_adapter -d swebench-lite
    fugu-eval -a outputs/planner/final_adapter -d swebench-verified -n 50 -o results.json

Only SWE-bench variants are supported for evaluation until a standalone
Python workspace exists for HumanEval / MBPP.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

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

# SWE-bench only until standalone HumanEval/MBPP workspace support exists.
_DATASET_MAP: dict[str, tuple[str, str, dict]] = {
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


def _load_dataset(name: str):
    """Dynamically import and instantiate a benchmark dataset."""
    import importlib

    module_path, class_name, kwargs = _DATASET_MAP[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


async def _evaluate_task(env, planner, task) -> dict:
    """Run a single episode with the planner driving action selection.

    Returns a dict with task-level metrics.
    """
    state = env.reset(task)
    total_reward = 0.0
    steps = 0
    actions_taken: list[str] = []
    all_tests_passed = False

    while True:
        action = planner.predict(state)
        actions_taken.append(action.name)

        state, reward, done, info = await env.step(action)
        total_reward += reward
        steps += 1
        all_tests_passed = info.get("all_tests_passed", False)

        if done:
            break

    env.close()

    return {
        "task_id": getattr(task, "task_id", "unknown"),
        "steps": steps,
        "total_reward": total_reward,
        "all_tests_passed": all_tests_passed,
        "actions": actions_taken,
        "final_pass_rate": (
            state.test_results.pass_rate
            if state.test_results is not None
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
    required=True,
    help="Path to the trained LoRA adapter directory.",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(_SUPPORTED_DATASETS, case_sensitive=False),
    default="swebench-lite",
    help="Benchmark dataset to evaluate on (SWE-bench variants only).",
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
    default="outputs/eval_results.json",
    help="Output path for evaluation results JSON.",
)
def main(
    config: str,
    adapter: str,
    dataset: str,
    max_tasks: int | None,
    output: str,
) -> None:
    """Evaluate a trained planner adapter against a benchmark dataset."""
    console.print(
        Panel.fit(
            "[bold blue]Fugu Planner Evaluator[/bold blue]",
            border_style="blue",
        )
    )

    # ── Load configuration ──────────────────────────────────────────
    from fugu.config import FuguConfig

    cfg = FuguConfig.from_yaml(config)

    console.print(f"  Config    : [green]{config}[/green]")
    console.print(f"  Adapter   : [green]{adapter}[/green]")
    console.print(f"  Dataset   : [green]{dataset}[/green]")
    console.print(f"  Max tasks : [green]{max_tasks or 'all'}[/green]")
    console.print(f"  Output    : [green]{output}[/green]")
    console.print()

    # ── Load planner model + adapter ────────────────────────────────
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

    # ── Build environment ───────────────────────────────────────────
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
        reward_calculator = RewardCalculator()
        env = CodingEnvironment(
            worker_pool=worker_pool,
            repo_manager=repo_manager,
            test_runner=test_runner,
            reward_calculator=reward_calculator,
            max_steps=cfg.env.max_steps,
            cleanup_on_done=cfg.repo.cleanup_on_done,
        )

    console.print("[green]✓[/green] Environment initialised\n")

    # ── Load dataset ────────────────────────────────────────────────
    with console.status(f"[bold yellow]Loading dataset '{dataset}'…[/bold yellow]"):
        ds = _load_dataset(dataset)

    # Get the list of tasks
    tasks = list(ds)
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    console.print(f"[green]✓[/green] Dataset loaded: {len(tasks)} tasks\n")

    # ── Run evaluation ──────────────────────────────────────────────
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
        eval_task = progress.add_task(
            "Evaluating tasks…", total=len(tasks)
        )

        for i, task in enumerate(tasks):
            task_id = getattr(task, "task_id", f"task_{i}")
            progress.update(
                eval_task,
                description=f"Evaluating [cyan]{task_id}[/cyan]",
            )

            try:
                task_result = asyncio.run(
                    _evaluate_task(env, planner, task)
                )
                results.append(task_result)
            except Exception as exc:
                console.print(
                    f"  [red]✗[/red] Task {task_id} failed: {exc}"
                )
                results.append(
                    {
                        "task_id": task_id,
                        "steps": 0,
                        "total_reward": 0.0,
                        "all_tests_passed": False,
                        "actions": [],
                        "final_pass_rate": 0.0,
                        "error": str(exc),
                    }
                )

            progress.advance(eval_task)

    elapsed = time.monotonic() - start_time

    # ── Compute aggregate metrics ───────────────────────────────────
    total_tasks = len(results)
    passed_tasks = sum(1 for r in results if r["all_tests_passed"])
    pass_rate = passed_tasks / total_tasks if total_tasks > 0 else 0.0
    avg_reward = (
        sum(r["total_reward"] for r in results) / total_tasks
        if total_tasks > 0
        else 0.0
    )
    avg_steps = (
        sum(r["steps"] for r in results) / total_tasks
        if total_tasks > 0
        else 0.0
    )
    avg_pass_rate = (
        sum(r["final_pass_rate"] for r in results) / total_tasks
        if total_tasks > 0
        else 0.0
    )

    # Action distribution
    action_counts: dict[str, int] = {}
    for r in results:
        for a in r.get("actions", []):
            action_counts[a] = action_counts.get(a, 0) + 1

    aggregate = {
        "dataset": dataset,
        "adapter": adapter,
        "total_tasks": total_tasks,
        "passed_tasks": passed_tasks,
        "task_pass_rate": pass_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_test_pass_rate": avg_pass_rate,
        "action_distribution": action_counts,
        "elapsed_seconds": elapsed,
    }

    # ── Save results ────────────────────────────────────────────────
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    full_results = {
        "aggregate": aggregate,
        "per_task": results,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(full_results, fh, indent=2, default=str)

    # ── Display summary ─────────────────────────────────────────────
    summary_table = Table(title="Evaluation Results", show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Dataset", dataset)
    summary_table.add_row("Adapter", adapter)
    summary_table.add_row("Total tasks", str(total_tasks))
    summary_table.add_row(
        "Tasks solved",
        f"{passed_tasks} / {total_tasks} ({pass_rate:.1%})",
    )
    summary_table.add_row("Avg reward", f"{avg_reward:.4f}")
    summary_table.add_row("Avg steps", f"{avg_steps:.1f}")
    summary_table.add_row("Avg test pass rate", f"{avg_pass_rate:.1%}")
    summary_table.add_row("Elapsed time", f"{elapsed:.1f}s")

    console.print()
    console.print(summary_table)

    # Action distribution table
    if action_counts:
        action_table = Table(title="Action Distribution")
        action_table.add_column("Action", style="bold")
        action_table.add_column("Count", justify="right", style="cyan")
        action_table.add_column("Percentage", justify="right", style="green")

        total_actions = sum(action_counts.values())
        for action_name, count in sorted(
            action_counts.items(), key=lambda x: x[1], reverse=True
        ):
            pct = count / total_actions * 100 if total_actions > 0 else 0.0
            action_table.add_row(action_name, str(count), f"{pct:.1f}%")

        console.print()
        console.print(action_table)

    console.print(f"\n  Results saved to [green]{output_path}[/green]")
    console.print("\n[bold green]Evaluation complete![/bold green]")


if __name__ == "__main__":
    main()
