"""CLI entry-point: ``fugu-train`` — planner training.

Trains the planner model's LoRA adapter via supervised fine-tuning on
transitions stored in a replay buffer.

Usage::

    fugu-train -c configs/default.yaml -b data/buffer
    fugu-train --buffer-dir data/swebench_buffer -o outputs/planner_v2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.command()
@click.option(
    "--config",
    "-c",
    default="configs/default.yaml",
    help="Path to YAML config file.",
)
@click.option(
    "--buffer-dir",
    "-b",
    default=None,
    help="Directory containing the replay buffer (overrides config).",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory for trained adapter (overrides config).",
)
@click.option(
    "--eval-split",
    type=float,
    default=0.1,
    help="Fraction of data to hold out for evaluation (0–1).",
)
def main(
    config: str,
    buffer_dir: str | None,
    output: str | None,
    eval_split: float,
) -> None:
    """Train the planner model from collected trajectories."""
    console.print(
        Panel.fit(
            "[bold magenta]Fugu Planner Trainer[/bold magenta]",
            border_style="magenta",
        )
    )

    # ── Load configuration ──────────────────────────────────────────
    from fugu.config import FuguConfig

    cfg = FuguConfig.from_yaml(config)

    # Apply CLI overrides
    if output is not None:
        cfg.training.output_dir = output

    resolved_buffer_dir = buffer_dir or cfg.buffer.storage_dir
    resolved_output_dir = cfg.training.output_dir

    console.print(f"  Config       : [green]{config}[/green]")
    console.print(f"  Buffer dir   : [green]{resolved_buffer_dir}[/green]")
    console.print(f"  Output dir   : [green]{resolved_output_dir}[/green]")
    console.print(f"  Eval split   : [green]{eval_split:.0%}[/green]")
    console.print(f"  Base model   : [green]{cfg.planner.base_model}[/green]")
    console.print(
        f"  LoRA rank    : [green]{cfg.planner.lora_r}[/green]  "
        f"alpha: [green]{cfg.planner.lora_alpha}[/green]"
    )
    console.print(f"  Epochs       : [green]{cfg.training.num_epochs}[/green]")
    console.print(
        f"  Batch size   : [green]{cfg.training.batch_size}[/green]  "
        f"× grad_accum={cfg.training.gradient_accumulation_steps}"
    )
    console.print(
        f"  Learning rate: [green]{cfg.training.learning_rate:.2e}[/green]"
    )
    console.print()

    # ── Load replay buffer ──────────────────────────────────────────
    from fugu.buffer.replay_buffer import ReplayBuffer

    buffer_path = Path(resolved_buffer_dir)

    with console.status("[bold yellow]Loading replay buffer…[/bold yellow]"):
        buffer = ReplayBuffer(
            capacity=cfg.buffer.capacity,
            storage_dir=resolved_buffer_dir,
            max_size_mb=cfg.buffer.max_size_mb,
        )

        # Try loading from the directory (buffer.jsonl or any .jsonl)
        buffer_file = buffer_path / "buffer.jsonl"
        if buffer_file.exists():
            buffer.load(str(buffer_file))
        elif buffer_path.is_file():
            buffer.load(str(buffer_path))
        else:
            # Try loading any .jsonl file in the directory
            jsonl_files = sorted(buffer_path.glob("*.jsonl"))
            if jsonl_files:
                buffer.load(str(jsonl_files[0]))
            else:
                console.print(
                    f"[bold red]Error:[/bold red] No buffer files found in "
                    f"'{resolved_buffer_dir}'",
                )
                sys.exit(1)

    total_transitions = len(buffer)
    if total_transitions == 0:
        console.print(
            "[bold red]Error:[/bold red] Replay buffer is empty — "
            "nothing to train on.",
        )
        sys.exit(1)

    console.print(
        f"[green]✓[/green] Buffer loaded: {total_transitions:,} transitions\n"
    )

    # ── Load planner model ──────────────────────────────────────────
    from fugu.planner.model import PlannerModel

    with console.status(
        f"[bold yellow]Loading model: {cfg.planner.base_model}…[/bold yellow]"
    ):
        planner = PlannerModel(cfg.planner)
        planner.load()

    trainable, total_params = planner.model.get_nb_trainable_parameters()
    console.print(
        f"[green]✓[/green] Model loaded: "
        f"{trainable:,} / {total_params:,} trainable parameters "
        f"({100.0 * trainable / total_params:.2f}%)\n"
    )

    # ── Train ───────────────────────────────────────────────────────
    from fugu.training.trainer import PlannerTrainer

    trainer = PlannerTrainer(model=planner, config=cfg.training)

    console.print("[bold yellow]Training started…[/bold yellow]\n")
    start_time = time.monotonic()

    try:
        metrics = trainer.train_from_buffer(buffer, eval_split=eval_split)
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user.[/yellow]")
        metrics = {}

    elapsed = time.monotonic() - start_time

    # ── Save adapter ────────────────────────────────────────────────
    adapter_path = Path(resolved_output_dir) / "final_adapter"
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.save(str(adapter_path))

    # ── Summary ─────────────────────────────────────────────────────
    summary_table = Table(title="Training Summary", show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", style="cyan")

    summary_table.add_row("Total transitions", f"{total_transitions:,}")
    summary_table.add_row("Elapsed time", f"{elapsed:.1f}s")
    summary_table.add_row("Adapter saved to", str(adapter_path))

    for key, value in metrics.items():
        if isinstance(value, float):
            summary_table.add_row(key, f"{value:.6f}")
        else:
            summary_table.add_row(key, str(value))

    console.print()
    console.print(summary_table)
    console.print("\n[bold green]Training complete![/bold green]")


if __name__ == "__main__":
    main()
