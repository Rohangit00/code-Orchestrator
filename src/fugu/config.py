"""Fugu configuration system.

Uses ``pydantic-settings`` so that every value can be overridden via
environment variables prefixed with ``FUGU_`` (e.g. ``FUGU_PLANNER__BASE_MODEL``).

A convenience ``FuguConfig.from_yaml()`` class method loads defaults from a
YAML file and merges them with env-var overrides.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-configs (plain BaseModel – they don't read env vars on their own)
# ---------------------------------------------------------------------------


class WorkerConfig(BaseModel):
    """Connection settings for the three worker LLMs."""

    qwen_url: str = "http://localhost:8001/v1"
    gemma_url: str = "http://localhost:8002/v1"
    ornith_url: str = "http://localhost:8003/v1"
    timeout: float = 120.0
    max_tokens: int = 4096
    temperature: float = 0.2


class RepoConfig(BaseModel):
    """Local repository / workspace management."""

    workspace_dir: str = "/tmp/fugu_workspaces"
    max_disk_mb: int = 2048
    cleanup_on_done: bool = True


class PlannerConfig(BaseModel):
    """Planner model and LoRA adapter settings."""

    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    load_in_4bit: bool = True
    max_seq_length: int = 4096


class TrainingConfig(BaseModel):
    """Training hyper-parameters."""

    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    save_steps: int = 500
    eval_steps: int = 250
    output_dir: str = "outputs/planner"
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10


class BufferConfig(BaseModel):
    """Replay buffer settings."""

    capacity: int = 100_000
    max_size_mb: int = 512
    storage_dir: str = "data/buffer"


class EnvConfig(BaseModel):
    """Coding environment limits and execution safety."""

    max_steps: int = 20
    test_timeout_seconds: int = 300
    # "host" for trusted local/mock only; "docker" for third-party remote
    # repos (runs tests/compile inside a container — not the official
    # SWE-bench harness).
    isolation_mode: str = "host"
    # Default False: refuse untrusted remote test execution on the host.
    # Set True only for trusted local path / mock fixtures.
    allow_host_execution: bool = False
    # Docker executor settings (used when isolation_mode == "docker")
    docker_image: str = "python:3.11-slim"
    docker_network: str = "none"
    docker_memory: str = "4g"
    docker_cpus: str = "2"
    docker_workdir: str = "/workspace"
    docker_user: str = ""
    docker_extra_args: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Root config (reads env vars)
# ---------------------------------------------------------------------------


class FuguConfig(BaseSettings):
    """Top-level configuration for the Fugu orchestrator.

    Values are resolved in order (highest priority first):

    1. Environment variables with ``FUGU_`` prefix (nested fields use ``__``,
       e.g. ``FUGU_ENV__ISOLATION_MODE``).
    2. Values from YAML when using :meth:`from_yaml` (passed as init kwargs).
    3. Field defaults declared above.

    Note: pydantic-settings defaults to init-over-env; we invert that via
    ``settings_customise_sources`` so ``FUGU_*`` always wins over YAML.
    """

    model_config = SettingsConfigDict(
        env_prefix="FUGU_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    repo: RepoConfig = Field(default_factory=RepoConfig)
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    buffer: BufferConfig = Field(default_factory=BufferConfig)
    env: EnvConfig = Field(default_factory=EnvConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Env must beat YAML/init so FUGU_ENV__ISOLATION_MODE=docker works
        # even when configs/default.yaml still says isolation_mode: host.
        return (
            env_settings,
            init_settings,
            dotenv_settings,
            file_secret_settings,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> FuguConfig:
        """Load configuration from a YAML file, merged with env-var overrides.

        Args:
            path: Filesystem path to a YAML config file. Missing keys fall
                back to defaults; environment variables always win.

        Returns:
            A fully validated ``FuguConfig`` instance.
        """
        yaml_path = Path(path)
        data: dict[str, Any] = {}
        if yaml_path.exists():
            with open(yaml_path) as fh:
                raw = yaml.safe_load(fh)
                if isinstance(raw, dict):
                    data = raw
        return cls(**data)
