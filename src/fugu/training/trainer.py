"""Supervised fine-tuning trainer for the planner model.

Wraps the HuggingFace :class:`~transformers.Trainer` to train the
planner's LoRA adapter on collected ``(state, action)`` transitions.
Supports training from an explicit :class:`TransitionDataset` or directly
from a :class:`~fugu.buffer.replay_buffer.ReplayBuffer`.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from fugu.training.data import TransitionDataset

if TYPE_CHECKING:
    from fugu.buffer.replay_buffer import ReplayBuffer
    from fugu.config import TrainingConfig
    from fugu.planner.model import PlannerModel

logger = logging.getLogger(__name__)


class PlannerTrainer:
    """SFT trainer for the planner model.

    Parameters
    ----------
    model : PlannerModel
        A loaded planner model (with LoRA adapter attached).
    config : TrainingConfig
        Training hyperparameters.
    """

    def __init__(self, model: PlannerModel, config: TrainingConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataset: TransitionDataset,
        eval_dataset: TransitionDataset | None = None,
    ) -> dict:
        """Run supervised fine-tuning.

        Creates HuggingFace ``TrainingArguments`` from the config, builds a
        ``Trainer``, and executes the training loop.

        Parameters
        ----------
        train_dataset : TransitionDataset
            Training data.
        eval_dataset : TransitionDataset | None
            Optional evaluation data.  If provided, evaluation runs every
            ``eval_steps`` steps.

        Returns
        -------
        dict
            Training metrics from the completed run.
        """
        if self.model.model is None or self.model.tokenizer is None:
            raise RuntimeError(
                "Planner model must be loaded before training. "
                "Call PlannerModel.load() first."
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=self.config.eval_steps if eval_dataset is not None else None,
            save_total_limit=3,
            load_best_model_at_end=eval_dataset is not None,
            report_to="none",
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            optim="paged_adamw_8bit",
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
        )

        logger.info(
            "Starting training: %d examples, %d epochs, lr=%.2e",
            len(train_dataset),
            self.config.num_epochs,
            self.config.learning_rate,
        )

        result = trainer.train()
        metrics = result.metrics
        metrics["train_samples"] = len(train_dataset)

        logger.info("Training complete — metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Train from buffer
    # ------------------------------------------------------------------

    def train_from_buffer(
        self,
        buffer: ReplayBuffer,
        eval_split: float = 0.1,
    ) -> dict:
        """Load transitions from a replay buffer, split, and train.

        Parameters
        ----------
        buffer : ReplayBuffer
            A replay buffer containing collected transitions.
        eval_split : float
            Fraction of transitions to hold out for evaluation (0–1).

        Returns
        -------
        dict
            Training metrics.
        """
        all_transitions = buffer.get_all()
        if not all_transitions:
            raise ValueError("Replay buffer is empty — nothing to train on.")

        # Light quality filter: drop non-positive-return episodes by default.
        from fugu.training.filter import filter_transitions

        filtered = filter_transitions(
            all_transitions,
            min_return=0.0,
            require_solved=False,
        )
        if not filtered:
            logger.warning(
                "Quality filter removed all transitions; falling back to "
                "unfiltered buffer for training."
            )
            filtered = all_transitions
        else:
            logger.info(
                "Using %d / %d transitions after quality filter",
                len(filtered),
                len(all_transitions),
            )

        total = len(filtered)
        eval_size = max(1, int(math.floor(total * eval_split)))
        train_size = total - eval_size

        if train_size < 1:
            raise ValueError(
                f"Not enough transitions ({total}) for the requested "
                f"eval split ({eval_split:.0%})."
            )

        train_transitions = filtered[:train_size]
        eval_transitions = filtered[train_size:]

        tokenizer_name = self.model.config.base_model
        max_length = self.model.config.max_seq_length

        train_dataset = TransitionDataset(
            train_transitions,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )
        eval_dataset = TransitionDataset(
            eval_transitions,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
        )

        logger.info(
            "Train/eval split: %d / %d transitions",
            len(train_transitions),
            len(eval_transitions),
        )

        return self.train(train_dataset, eval_dataset)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, eval_dataset: TransitionDataset) -> dict:
        """Evaluate the model on *eval_dataset* and return metrics.

        Parameters
        ----------
        eval_dataset : TransitionDataset
            The evaluation dataset.

        Returns
        -------
        dict
            Evaluation metrics including ``eval_loss``.
        """
        if self.model.model is None or self.model.tokenizer is None:
            raise RuntimeError(
                "Planner model must be loaded before evaluation. "
                "Call PlannerModel.load() first."
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_eval_batch_size=self.config.batch_size,
            report_to="none",
            remove_unused_columns=False,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.model.tokenizer,
            mlm=False,
        )

        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.model.tokenizer,
        )

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """Save the LoRA adapter and tokenizer.

        Parameters
        ----------
        path : str | None
            Destination directory.  Defaults to ``config.output_dir``.
        """
        save_path = path or self.config.output_dir
        self.model.save_adapter(save_path)
        logger.info("Model and tokenizer saved to %s", save_path)
