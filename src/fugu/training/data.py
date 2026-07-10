"""PyTorch Dataset for supervised fine-tuning on collected transitions.

Wraps a list of :class:`~fugu.core.state.Transition` objects, tokenising
each ``(state → action)`` pair with the planner tokenizer.  Prompt tokens
are masked in the labels (set to ``-100``) so the loss is computed only
over the action-name completion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from fugu.planner.tokenizer import StateTokenizer

if TYPE_CHECKING:
    from fugu.core.state import Transition

logger = logging.getLogger(__name__)


class TransitionDataset(Dataset):
    """PyTorch ``Dataset`` wrapping transitions for SFT training.

    Parameters
    ----------
    transitions : list[Transition]
        Ordered list of MDP transitions to train on.
    tokenizer_name : str
        HuggingFace model id whose tokenizer will be used.
    max_length : int
        Maximum sequence length (prompt + completion).
    """

    def __init__(
        self,
        transitions: list[Transition],
        tokenizer_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_length: int = 4096,
    ) -> None:
        self.transitions = transitions
        self.state_tokenizer = StateTokenizer()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_length = max_length

        logger.info(
            "TransitionDataset initialised: %d transitions, max_length=%d",
            len(self.transitions),
            self.max_length,
        )

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a tokenised training example.

        The returned dict contains:

        * ``input_ids``      — full sequence (prompt + completion)
        * ``attention_mask`` — standard attention mask
        * ``labels``         — same as ``input_ids`` but with prompt tokens
          set to ``-100`` so the cross-entropy loss only covers the action
          completion tokens.
        """
        transition = self.transitions[idx]
        # Same prompt construction as PlannerModel.predict / get_action_prompt.
        pair = self.state_tokenizer.format_for_training(
            transition, tokenizer=self.tokenizer
        )

        prompt_text = pair["prompt"]
        completion_text = pair["completion"]

        # Tokenise prompt and completion separately so we know the boundary
        prompt_encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length - 20,  # reserve room for completion
        )
        completion_encoding = self.tokenizer(
            completion_text,
            add_special_tokens=False,
        )

        # Concatenate and add EOS
        input_ids = (
            prompt_encoding["input_ids"]
            + completion_encoding["input_ids"]
            + [self.tokenizer.eos_token_id]
        )

        prompt_len = len(prompt_encoding["input_ids"])

        # Truncate if exceeding max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        attention_mask = [1] * len(input_ids)

        # Labels: mask the prompt portion with -100
        labels = [-100] * min(prompt_len, len(input_ids))
        if len(input_ids) > prompt_len:
            labels += input_ids[prompt_len:]

        # Pad to max_length
        pad_length = self.max_length - len(input_ids)
        if pad_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * pad_length
            attention_mask += [0] * pad_length
            labels += [-100] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
