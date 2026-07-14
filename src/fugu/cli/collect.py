"""CLI entry-point: ``fugu-collect`` — trajectory collection.

Collects orchestration trajectories from a benchmark dataset by running
episodes in the coding environment with one or more strategies, then
stores the resulting transitions in a replay buffer.

Usage::

    fugu-collect -c configs/default.yaml -d swebench-lite -n 50
    fugu-collect -d swebench-verified --strategy round-robin -o data/swe_buffer

Only SWE-bench variants are supported for collection until a standalone
Python workspace exists for HumanEval / MBPP.
"""

from __future__ import annotations

import asyncio
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

# Dataset name → (module path, class name, constructor kwargs)
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

# Strategy name → loader key
_STRATEGY_MAP: dict[str, str] = {
    "all": "all",
    "single-qwen": "single-qwen",
    "single-gemma": "single-gemma",
    "single-ornith": "single-ornith",
    "round-robin": "round-robin",
    "retry-on-fail": "retry-on-fail",
    "verify-first": "verify-first",
}


def _load_dataset(name: str):
    """Dynamically import and instantiate a benchmark dataset."""
    import importlib

    module_path, class_name, kwargs = _DATASET_MAP[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


def _load_strategies(strategy_name: str):
    """Return one or more strategy objects based on the CLI argument."""
    from fugu.trajectory.strategies import ALL_STRATEGIES

    if strategy_name == "all":
        return ALL_STRATEGIES

    # Find the matching strategy by name
    for s in ALL_STRATEGIES:
        if getattr(s, "name", "").replace("_", "-") == strategy_name:
            return [s]

    # Fallback: return all
    console.print(
        f"[yellow]Warning:[/yellow] Strategy '{strategy_name}' not found, "
        "using all strategies.",
    )
    return ALL_STRATEGIES


@click.command()
@click.option(
    "--config",
    "-c",
    default="configs/default.yaml",
    help="Path to YAML config file.",
)
@click.option(
    "--dataset",
    "-d",
    type=click.Choice(_SUPPORTED_DATASETS, case_sensitive=False),
    default="swebench-lite",
    help="Benchmark dataset to collect from (SWE-bench variants only).",
)
@click.option(
    "--max-tasks",
    "-n",
    type=int,
    default=None,
    help="Maximum number of tasks to run (default: all).",
)
@click.option(
    "--strategy",
    "-s",
    type=click.Choice(
        [
            "all",
            "single-qwen",
            "single-gemma",
            "single-ornith",
            "round-robin",
            "retry-on-fail",
            "verify-first",
        ]
    ),
    default="all",
    help="Collection strategy to use.",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory for the replay buffer (overrides config).",
)
def main(
    config: str,
    dataset: str,
    max_tasks: int | None,
    strategy: str,
    output: str | None,
) -> None:
    """Collect orchestration trajectories from a benchmark dataset."""
    console.print(
        Panel.fit(
            "[bold cyan]Fugu Trajectory Collector[/bold cyan]",
            border_style="cyan",
        )
    )

    # ── Load configuration ──────────────────────────────────────────
    from fugu.config import FuguConfig

    cfg = FuguConfig.from_yaml(config)
    console.print(f"  Config      : [green]{config}[/green]")
    console.print(f"  Dataset     : [green]{dataset}[/green]")
    console.print(f"  Strategy    : [green]{strategy}[/green]")
    console.print(f"  Max tasks   : [green]{max_tasks or 'all'}[/green]")

    # ── Resolve output directory ────────────────────────────────────
    storage_dir = output or cfg.buffer.storage_dir
    console.print(f"  Output dir  : [green]{storage_dir}[/green]")
    console.print()

    # ── Build components ────────────────────────────────────────────
    with console.status("[bold yellow]Initialising components…[/bold yellow]"):
        from fugu.buffer.replay_buffer import ReplayBuffer
        from fugu.core.reward import RewardCalculator
        from fugu.env.coding_env import CodingEnvironment
        from fugu.execution.runner import TestRunner
        from fugu.repo.manager import RepoManager
        from fugu.trajectory.collector import TrajectoryCollector
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
        buffer = ReplayBuffer(
            capacity=cfg.buffer.capacity,
            storage_dir=storage_dir,
            max_size_mb=cfg.buffer.max_size_mb,
        )
        collector = TrajectoryCollector(env=env, buffer=buffer)

    console.print("[green]✓[/green] Components initialised\n")

    # ── Load dataset ────────────────────────────────────────────────
    with console.status(f"[bold yellow]Loading dataset '{dataset}'…[/bold yellow]"):
        ds = _load_dataset(dataset)
    console.print(f"[green]✓[/green] Dataset loaded: {dataset}\n")

    # ── Load strategies ─────────────────────────────────────────────
    strategies = _load_strategies(strategy)
    strategy_names = [getattr(s, "name", str(s)) for s in strategies]
    console.print(
        f"[green]✓[/green] Strategies: {', '.join(strategy_names)}\n"
    )

    # ── Collect trajectories ────────────────────────────────────────
    start_time = time.monotonic()

    try:
        if strategy == "all":
            result = asyncio.run(
                collector.collect_multi_strategy(
                    dataset=ds,
                    strategies=strategies,
                    max_tasks=max_tasks,
                )
            )
        else:
            strat = strategies[0]
            result = asyncio.run(
                collector.collect_dataset(
                    dataset=ds,
                    strategy=strat,
                    max_tasks=max_tasks,
                )
            )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        result = None

    elapsed = time.monotonic() - start_time

    # ── Save buffer ─────────────────────────────────────────────────
    save_path = Path(storage_dir) / "buffer.jsonl"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    buffer.save(str(save_path))

    # ── Summary ─────────────────────────────────────────────────────
    summary_table = Table(title="Collection Summary", show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Dataset", dataset)
    summary_table.add_row("Strategy", strategy)
    summary_table.add_row("Transitions collected", str(len(buffer)))
    summary_table.add_row("Elapsed time", f"{elapsed:.1f}s")
    summary_table.add_row("Buffer saved to", str(save_path))

    console.print()
    console.print(summary_table)
    console.print("\n[bold green]Done![/bold green]")


if __name__ == "__main__":
    main()
