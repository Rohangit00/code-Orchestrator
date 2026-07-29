"""HuggingFace Hub loading without dataset *scripts*.

``datasets`` 4+ removed support for remote loading scripts
(``trust_remote_code``). Prefer parquet/jsonl data files via Hub download
or the generic ``json``/``parquet`` builders.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def resolve_hf_token() -> str | None:
    """Return a Hub token from the environment (never logged).

    Checked in order:

    1. ``HF_TOKEN`` (Hugging Face standard)
    2. ``HUGGING_FACE_HUB_TOKEN`` (legacy alias)
    3. ``FUGU_HF_TOKEN`` (Fugu-specific alias)

    Set only in your private shell/screen, e.g.::

        export HF_TOKEN=hf_xxxxxxxx
        # or: export FUGU_HF_TOKEN=hf_xxxxxxxx

    Do not commit real tokens. See ``env_vm.sh.example``.
    """
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "FUGU_HF_TOKEN"):
        val = os.environ.get(key)
        if val and val.strip() and "your_token" not in val.lower() and "placeholder" not in val.lower():
            token = val.strip()
            # Normalize so huggingface_hub / datasets both see it
            os.environ.setdefault("HF_TOKEN", token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
            return token
    return None


def _hub_kwargs() -> dict[str, Any]:
    """Keyword args for hub APIs that accept ``token=``."""
    token = resolve_hf_token()
    if token:
        logger.info("Hugging Face Hub: using authenticated token from env")
        return {"token": token}
    logger.warning(
        "Hugging Face Hub: unauthenticated requests "
        "(set HF_TOKEN or FUGU_HF_TOKEN in this shell for higher rate limits)"
    )
    return {}


def load_hub_jsonl_rows(
    repo_id: str,
    *,
    filenames: Iterable[str] | None = None,
    filename_glob_suffix: str = ".jsonl",
    dedupe_key: str | None = "question_id",
) -> list[dict[str, Any]]:
    """Download JSONL files from a Hub *dataset* repo and parse rows.

    Parameters
    ----------
    repo_id:
        e.g. ``livecodebench/code_generation_lite``.
    filenames:
        Explicit file list. If ``None``, all ``*filename_glob_suffix`` files
        at the repo root (non-hidden) are used, sorted by name.
    dedupe_key:
        If set, keep first row per key value.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    hub = _hub_kwargs()

    if filenames is None:
        all_files = list_repo_files(repo_id, repo_type="dataset", **hub)
        # Top-level data files only (ignore nested paths and the .py script).
        filenames = sorted(
            f
            for f in all_files
            if f.endswith(filename_glob_suffix)
            and "/" not in f
            and not f.startswith(".")
        )
    else:
        filenames = list(filenames)

    if not filenames:
        raise FileNotFoundError(
            f"No {filename_glob_suffix} files found in dataset repo {repo_id!r}. "
            "The Hub layout may have changed."
        )

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fn in filenames:
        logger.info("Downloading %s from %s …", fn, repo_id)
        path = hf_hub_download(repo_id, fn, repo_type="dataset", **hub)
        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skip %s:%d (%s)", fn, line_no, exc)
                    continue
                if not isinstance(row, dict):
                    continue
                if dedupe_key:
                    key = row.get(dedupe_key)
                    if key is not None:
                        sk = str(key)
                        if sk in seen:
                            continue
                        seen.add(sk)
                rows.append(row)

    logger.info("Loaded %d rows from %s (%d files)", len(rows), repo_id, len(filenames))
    return rows


def load_dataset_script_free(
    path: str,
    *,
    split: str = "test",
    name: str | None = None,
) -> Any:
    """Load a Hub dataset preferring script-free data files.

    1. Try normal ``load_dataset`` (works for parquet-only repos like SWE-bench).
    2. On "dataset scripts are no longer supported", fall back to JSONL Hub files.

    Uses ``HF_TOKEN`` / ``FUGU_HF_TOKEN`` when set (see :func:`resolve_hf_token`).
    """
    from datasets import load_dataset

    token = resolve_hf_token()
    load_kw: dict[str, Any] = {}
    if token:
        load_kw["token"] = token

    try:
        if name:
            return load_dataset(path, name, split=split, **load_kw)
        return load_dataset(path, split=split, **load_kw)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "dataset scripts are no longer supported" not in msg and "scripts" not in msg:
            raise
        logger.warning(
            "load_dataset(%s) failed due to removed dataset scripts (%s); "
            "falling back to Hub JSONL files.",
            path,
            exc,
        )
        rows = load_hub_jsonl_rows(path)
        # Return a list-like of dicts; callers should accept Iterable[dict]
        return rows
    except Exception as exc:
        # datasets 4/5 may raise other errors for script repos
        msg = str(exc).lower()
        if "script" in msg or "trust_remote_code" in msg:
            logger.warning(
                "load_dataset(%s) failed (%s); falling back to Hub JSONL.",
                path,
                exc,
            )
            return load_hub_jsonl_rows(path)
        raise
