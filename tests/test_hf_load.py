"""Script-free Hub loading helpers (datasets 4+)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fugu.datasets.hf_load import load_hub_jsonl_rows, resolve_hf_token


def test_resolve_hf_token_from_env(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("FUGU_HF_TOKEN", raising=False)
    assert resolve_hf_token() is None
    monkeypatch.setenv("FUGU_HF_TOKEN", "hf_test_secret_token")
    tok = resolve_hf_token()
    assert tok == "hf_test_secret_token"
    # normalized for hub/datasets
    import os

    assert os.environ.get("HF_TOKEN") == "hf_test_secret_token"


def test_resolve_hf_token_ignores_placeholder(monkeypatch):
    monkeypatch.delenv("FUGU_HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "hf_YOUR_TOKEN_PLACEHOLDER")
    assert resolve_hf_token() is None


def test_load_hub_jsonl_rows_local_mock(tmp_path: Path, monkeypatch):
    f1 = tmp_path / "test.jsonl"
    f2 = tmp_path / "test2.jsonl"
    rows1 = [
        {"question_id": "a", "question_content": "A"},
        {"question_id": "b", "question_content": "B"},
    ]
    rows2 = [
        {"question_id": "b", "question_content": "B-dup"},
        {"question_id": "c", "question_content": "C"},
    ]
    f1.write_text("\n".join(json.dumps(r) for r in rows1) + "\n", encoding="utf-8")
    f2.write_text("\n".join(json.dumps(r) for r in rows2) + "\n", encoding="utf-8")

    def fake_list(repo_id, repo_type="dataset", **kwargs):
        return ["test.jsonl", "test2.jsonl", "code_generation_lite.py", "README.md"]

    def fake_dl(repo_id, filename, repo_type="dataset", **kwargs):
        return str(tmp_path / filename)

    with patch("huggingface_hub.list_repo_files", side_effect=fake_list), patch(
        "huggingface_hub.hf_hub_download", side_effect=fake_dl
    ):
        rows = load_hub_jsonl_rows("livecodebench/code_generation_lite")

    ids = [r["question_id"] for r in rows]
    assert ids == ["a", "b", "c"]  # b deduped (first wins)


def test_load_hub_jsonl_explicit_files(tmp_path: Path):
    f1 = tmp_path / "only.jsonl"
    f1.write_text(
        json.dumps({"question_id": "x", "question_content": "X"}) + "\n",
        encoding="utf-8",
    )

    def fake_dl(repo_id, filename, repo_type="dataset", **kwargs):
        assert filename == "only.jsonl"
        return str(f1)

    with patch("huggingface_hub.hf_hub_download", side_effect=fake_dl):
        rows = load_hub_jsonl_rows(
            "dummy/repo", filenames=["only.jsonl"], dedupe_key="question_id"
        )
    assert len(rows) == 1
    assert rows[0]["question_id"] == "x"
