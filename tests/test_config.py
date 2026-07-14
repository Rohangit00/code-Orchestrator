"""Config loading: env vars must override YAML (isolation_mode, workers)."""

from __future__ import annotations

from pathlib import Path

from fugu.config import FuguConfig


def test_env_isolation_mode_overrides_yaml(tmp_path: Path, monkeypatch):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "env:\n  isolation_mode: host\n  allow_host_execution: false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FUGU_ENV__ISOLATION_MODE", "docker")
    monkeypatch.setenv("FUGU_ENV__ALLOW_HOST_EXECUTION", "false")

    cfg = FuguConfig.from_yaml(yaml_path)
    assert cfg.env.isolation_mode == "docker"
    assert cfg.env.allow_host_execution is False


def test_env_worker_url_overrides_yaml(tmp_path: Path, monkeypatch):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        'worker:\n  qwen_url: "http://localhost:8001/v1"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("FUGU_WORKER__QWEN_URL", "http://10.0.0.1:9000/v1")

    cfg = FuguConfig.from_yaml(yaml_path)
    assert cfg.worker.qwen_url == "http://10.0.0.1:9000/v1"


def test_yaml_used_when_env_unset(tmp_path: Path, monkeypatch):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "env:\n  isolation_mode: docker\n  docker_image: myimage:latest\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("FUGU_ENV__ISOLATION_MODE", raising=False)
    monkeypatch.delenv("FUGU_ENV__DOCKER_IMAGE", raising=False)

    cfg = FuguConfig.from_yaml(yaml_path)
    assert cfg.env.isolation_mode == "docker"
    assert cfg.env.docker_image == "myimage:latest"
