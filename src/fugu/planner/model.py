"""Planner model — a small LLM with QLoRA adapters for action prediction.

Wraps a HuggingFace causal-LM (default: Qwen2.5-3B-Instruct) in 4-bit
quantisation with a LoRA adapter on the attention projections.  The model
predicts the next orchestration action given the current
:class:`~fugu.core.state.PlannerState`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, TaskType

from fugu.core.actions import (
    ACTION_NAMES,
    DISABLED_WORKER_ACTIONS,
    NUM_ACTIONS,
    PlannerAction,
)
from fugu.core.state import PlannerState
from fugu.planner.prompts import SYSTEM_PROMPT, build_planner_prompt

if TYPE_CHECKING:
    from fugu.config import PlannerConfig

logger = logging.getLogger(__name__)


class PlannerModel:
    """Wraps a small LLM with QLoRA adapters for action prediction.

    Parameters
    ----------
    config : PlannerConfig
        Model and LoRA hyperparameters (base model name, rank, alpha, etc.).
    """

    def __init__(self, config: PlannerConfig) -> None:
        self.config = config
        self.model: AutoModelForCausalLM | None = None
        self.tokenizer: AutoTokenizer | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the base model with 4-bit quantisation and attach a LoRA adapter.

        Uses ``BitsAndBytesConfig`` for NF4 quantisation with bfloat16
        compute dtype, then wraps the model with a ``LoraConfig`` targeting
        the configured attention projection modules.
        """
        logger.info("Loading base model: %s", self.config.base_model)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(
            "LoRA adapter attached — trainable: %s / %s (%.2f%%)",
            f"{trainable:,}",
            f"{total:,}",
            100.0 * trainable / total if total > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, state: PlannerState) -> PlannerAction:
        """Predict the best next action given the current state.

        Builds a chat-style prompt from the state, generates up to 10 new
        tokens, and parses the output for a valid action name.

        Falls back to ``STOP`` if the output cannot be parsed.
        """
        self._ensure_loaded()

        prompt = self.get_action_prompt(state)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        generated_text = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        ).strip()

        return self._parse_action(generated_text)

    def predict_with_probs(
        self, state: PlannerState
    ) -> tuple[PlannerAction, dict[PlannerAction, float]]:
        """Predict action *and* return a probability distribution over all actions.

        Uses a single forward pass to extract logits at the first generated
        position, then computes softmax probabilities for each action token.

        Returns
        -------
        tuple[PlannerAction, dict[PlannerAction, float]]
            The argmax action and a mapping from every ``PlannerAction`` to
            its softmax probability.
        """
        self._ensure_loaded()

        prompt = self.get_action_prompt(state)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_seq_length,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Logits for the *next* token (last position in the sequence)
        next_token_logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Map each *active* action name to its first token ID / logit.
        # Disabled workers (CALL_GEMMA) get -inf so they cannot win.
        action_logits: dict[PlannerAction, float] = {}
        for action in PlannerAction:
            if action in DISABLED_WORKER_ACTIONS:
                action_logits[action] = float("-inf")
                continue
            action_name = ACTION_NAMES[action]
            token_ids = self.tokenizer.encode(
                action_name, add_special_tokens=False
            )
            if token_ids:
                action_logits[action] = next_token_logits[token_ids[0]].item()
            else:
                action_logits[action] = float("-inf")

        # Softmax over active actions only (disabled stay 0 after renormalize)
        active = [a for a in PlannerAction if a not in DISABLED_WORKER_ACTIONS]
        logit_values = torch.tensor(
            [action_logits[a] for a in active],
            dtype=torch.float32,
        )
        probs = torch.softmax(logit_values, dim=0)

        action_probs: dict[PlannerAction, float] = {
            a: 0.0 for a in PlannerAction
        }
        for i, action in enumerate(active):
            action_probs[action] = probs[i].item()

        best_action = max(action_probs, key=action_probs.get)  # type: ignore[arg-type]
        return best_action, action_probs

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def get_action_prompt(self, state: PlannerState) -> str:
        """Build the full prompt via the shared train/infer builder."""
        return build_planner_prompt(
            state,
            tokenizer=self.tokenizer,
            add_generation_prompt=True,
            system_prompt=SYSTEM_PROMPT,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_adapter(self, path: str) -> None:
        """Save LoRA adapter weights and tokenizer to *path*."""
        self._ensure_loaded()
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Adapter saved to %s", path)

    def load_adapter(self, path: str) -> None:
        """Load a previously saved LoRA adapter from *path*.

        Replaces the current adapter weights.  The base model must already
        be loaded (call :meth:`load` first).
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Base model must be loaded before loading an adapter. "
                "Call load() first."
            )

        # Unwrap existing PEFT wrapper if present
        base_model = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )

        self.model = PeftModel.from_pretrained(
            base_model,
            path,
            is_trainable=True,
        )
        logger.info("Adapter loaded from %s", path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Raise if the model/tokenizer have not been loaded."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call PlannerModel.load() first."
            )

    @staticmethod
    def _parse_action(text: str) -> PlannerAction:
        """Extract a PlannerAction from generated text.

        Scans the text for any known **active** action name (case-insensitive).
        Disabled workers (e.g. ``CALL_GEMMA``) are ignored. Falls back to
        ``STOP`` if nothing matches.
        """
        normalized = text.strip().upper()

        # Try exact match first (only if enabled)
        try:
            action = PlannerAction.from_string(normalized)
            if action not in DISABLED_WORKER_ACTIONS:
                return action
            logger.warning(
                "Model emitted disabled action %s — defaulting to STOP",
                action.name,
            )
            return PlannerAction.STOP
        except ValueError:
            pass

        # Scan for any active action name embedded in the text
        for action in PlannerAction:
            if action in DISABLED_WORKER_ACTIONS:
                continue
            if ACTION_NAMES[action] in normalized:
                return action

        logger.warning(
            "Could not parse action from model output: %r — defaulting to STOP",
            text,
        )
        return PlannerAction.STOP
