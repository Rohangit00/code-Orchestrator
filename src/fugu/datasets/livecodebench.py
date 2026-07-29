"""LiveCodeBench adapter (Python code generation).

Loads contest-style problems from HuggingFace
``livecodebench/code_generation_lite`` (or a configurable dataset id),
filters to Python-friendly tasks, materialises a small pytest workspace,
and exposes train/val/test splits for collection and evaluation.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from typing import Any, Iterator, Literal

from fugu.datasets.base import BaseDataset, CodingTask

logger = logging.getLogger(__name__)

SplitName = Literal["train", "val", "test", "all"]

# Default: lite generation set used by LiveCodeBench releases.
_DEFAULT_HF = "livecodebench/code_generation_lite"

# Optional release → Hub JSONL files (top-level in code_generation_lite).
# cumulative: later releases include earlier problems via multiple files.
_RELEASE_TO_JSONL: dict[str, list[str] | None] = {
    "": None,  # all top-level jsonl
    "all": None,
    "release_v1": ["test.jsonl"],
    "v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "v5": [
        "test.jsonl",
        "test2.jsonl",
        "test3.jsonl",
        "test4.jsonl",
        "test5.jsonl",
    ],
    "release_v6": None,  # all files including test6
    "v6": None,
}

# Dual-mode public-test harness written into each LCB workspace.
# 1) stdin  — run solution.py, compare stdout (Codeforces/AtCoder-style)
# 2) functional — call Solution.method(*args) or module function (LeetCode-style)
_PYTEST_HARNESS = r'''
"""Auto-generated public-test harness for LiveCodeBench-style tasks."""
from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / "public_tests.json"
META_PATH = ROOT / "meta.json"


def _load_cases():
    raw = CASES_PATH.read_text(encoding="utf-8")
    data = json.loads(raw)
    if isinstance(data, dict) and "inputs" in data:
        inputs = data.get("inputs") or []
        outputs = data.get("outputs") or []
        return [{"input": i, "output": o} for i, o in zip(inputs, outputs)]
    if not isinstance(data, list):
        return []
    return data


def _load_meta():
    if not META_PATH.exists():
        return {}
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_value(s: str):
    s = (s or "").strip()
    if s == "":
        return ""
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _parse_functional_args(inp: str) -> list:
    """LCB functional inputs: one JSON/Python literal per line = one argument."""
    text = (inp or "").strip()
    if not text:
        return []
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        return [_parse_value(text)]
    return [_parse_value(ln) for ln in lines]


def _run_stdin(inp: str, timeout: float = 5.0) -> tuple[str, int, str]:
    data = inp if (inp.endswith("\n") or inp == "") else inp + "\n"
    proc = subprocess.run(
        [sys.executable, "solution.py"],
        input=data,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(ROOT),
    )
    return (proc.stdout or "").strip(), proc.returncode, (proc.stderr or "")


def _load_solution_module():
    path = ROOT / "solution.py"
    spec = importlib.util.spec_from_file_location("solution", path)
    assert spec and spec.loader, "cannot load solution.py"
    mod = importlib.util.module_from_spec(spec)
    # Ensure fresh load each case (worker overwrites file between runs)
    sys.modules.pop("solution", None)
    spec.loader.exec_module(mod)
    return mod


def _call_functional(func_name: str, args: list):
    mod = _load_solution_module()
    if hasattr(mod, "Solution"):
        inst = mod.Solution()
        if not hasattr(inst, func_name):
            raise AssertionError(
                f"Solution has no method {func_name!r}; "
                f"attrs={[a for a in dir(inst) if not a.startswith('_')]}"
            )
        fn = getattr(inst, func_name)
        return fn(*args)
    if hasattr(mod, func_name):
        return getattr(mod, func_name)(*args)
    # single public function fallback
    publics = [
        n
        for n, v in vars(mod).items()
        if callable(v) and not n.startswith("_") and n[0].islower()
    ]
    if len(publics) == 1:
        return getattr(mod, publics[0])(*args)
    raise AssertionError(
        f"no Solution.{func_name} or function {func_name!r} in solution.py"
    )


def test_public_cases():
    cases = _load_cases()
    assert cases, "no public test cases"
    meta = _load_meta()
    default_func = meta.get("func_name") or meta.get("function_name") or ""

    for i, case in enumerate(cases):
        if isinstance(case, str):
            continue
        inp = case.get("input", case.get("stdin", ""))
        expected_raw = case.get("output", case.get("stdout", ""))
        testtype = str(
            case.get("testtype", case.get("type", "stdin"))
        ).lower()

        if testtype in {"functional", "function", "leetcode"}:
            func_name = (
                case.get("func_name")
                or case.get("method_name")
                or default_func
            )
            assert func_name, f"case {i}: functional test missing func_name"
            args = _parse_functional_args(str(inp))
            expected = _parse_value(str(expected_raw))
            got = _call_functional(str(func_name), args)
            assert got == expected, (
                f"case {i} functional: got {got!r} expected {expected!r} "
                f"args={args!r}"
            )
            continue

        # stdin / default contest style
        out, rc, err = _run_stdin(str(inp))
        assert rc == 0, f"case {i} crashed rc={rc} stderr={err[:500]!r}"
        expected = str(expected_raw).strip()
        assert out == expected, f"case {i}: got {out!r} expected {expected!r}"
'''


class LiveCodeBenchDataset(BaseDataset):
    """Python LiveCodeBench code-generation problems.

    Parameters
    ----------
    hf_path:
        HuggingFace dataset id.
    release_version:
        Optional LCB release tag / config name (e.g. ``release_v5``).
        Passed as the HF ``name``/config when set.
    split:
        ``train`` / ``val`` / ``test`` / ``all`` subset after filtering.
    split_ratios:
        Fractions for train/val (test gets the remainder). Used when
        ``split_mode="random"``.
    split_mode:
        ``time`` — older contests → train, mid → val, newest → test.
        ``random`` — seeded shuffle then ratios.
    seed:
        RNG seed for random splits.
    python_only:
        Drop rows that look non-Python when platform/language fields exist.
    """

    def __init__(
        self,
        hf_path: str = _DEFAULT_HF,
        release_version: str | None = None,
        split: SplitName = "all",
        split_ratios: tuple[float, float] = (0.7, 0.15),
        split_mode: Literal["time", "random"] = "time",
        seed: int = 42,
        python_only: bool = True,
        max_problems: int | None = None,
    ) -> None:
        self._hf_path = hf_path
        self._release_version = release_version
        self._split = split
        self._split_ratios = split_ratios
        self._split_mode = split_mode
        self._seed = seed
        self._python_only = python_only
        self._max_problems = max_problems
        self._tasks: list[CodingTask] | None = None

    @property
    def name(self) -> str:
        return f"livecodebench_{self._split}"

    @property
    def size(self) -> int:
        self._ensure_loaded()
        assert self._tasks is not None
        return len(self._tasks)

    def __iter__(self) -> Iterator[CodingTask]:
        self._ensure_loaded()
        assert self._tasks is not None
        yield from self._tasks

    def _ensure_loaded(self) -> None:
        if self._tasks is not None:
            return
        rows = self._load_rows()
        tasks = [t for t in (self._row_to_task(r) for r in rows) if t is not None]
        if self._python_only:
            # Prefer rows that look Python-friendly (starter or no cpp markers)
            pass  # filtering done in _row_to_task
        tasks = self._apply_split(tasks)
        if self._max_problems is not None:
            tasks = tasks[: self._max_problems]
        self._tasks = tasks
        logger.info(
            "LiveCodeBench loaded split=%s n=%d (mode=%s)",
            self._split,
            len(tasks),
            self._split_mode,
        )

    def _load_rows(self) -> list[dict[str, Any]]:
        """Load LCB rows without dataset scripts (datasets 4+ compatible).

        Hub layout (code_generation_lite): top-level ``test.jsonl``,
        ``test2.jsonl``, … plus a legacy ``code_generation_lite.py`` script
        that ``datasets>=4`` refuses to run. We read the JSONL files directly.
        """
        from fugu.datasets.hf_load import load_hub_jsonl_rows

        logger.info(
            "Loading LiveCodeBench %s release=%s (script-free JSONL) …",
            self._hf_path,
            self._release_version,
        )
        # Map optional release tags to cumulative JSONL files when known.
        # Unknown / None → all top-level *.jsonl (deduped by question_id).
        release_files = _RELEASE_TO_JSONL.get(
            (self._release_version or "").strip().lower()
        )
        rows = load_hub_jsonl_rows(
            self._hf_path,
            filenames=release_files,
            dedupe_key="question_id",
        )
        return rows

    def _apply_split(self, tasks: list[CodingTask]) -> list[CodingTask]:
        if self._split == "all" or not tasks:
            return tasks

        if self._split_mode == "time":
            def _key(t: CodingTask) -> str:
                return str(t.metadata.get("contest_date") or t.task_id)

            ordered = sorted(tasks, key=_key)
        else:
            ordered = list(tasks)
            rng = random.Random(self._seed)
            rng.shuffle(ordered)

        n = len(ordered)
        tr, vr = self._split_ratios
        n_train = max(1, int(n * tr)) if n > 3 else max(1, n - 2)
        n_val = max(0, int(n * vr)) if n > 3 else (1 if n > 1 else 0)
        n_test = max(0, n - n_train - n_val)

        # ensure test non-empty when possible
        if n_test == 0 and n > n_train + n_val:
            n_test = n - n_train - n_val
        if n_train + n_val + n_test < n:
            n_test = n - n_train - n_val

        train = ordered[:n_train]
        val = ordered[n_train : n_train + n_val]
        test = ordered[n_train + n_val :]

        return {"train": train, "val": val, "test": test}[self._split]

    @staticmethod
    def _row_to_task(row: dict[str, Any]) -> CodingTask | None:
        # Field names vary slightly across LCB releases.
        qid = (
            row.get("question_id")
            or row.get("id")
            or row.get("problem_id")
            or row.get("instance_id")
        )
        if qid is None:
            return None
        task_id = str(qid)

        title = str(row.get("question_title") or row.get("title") or "")
        content = str(
            row.get("question_content")
            or row.get("problem_statement")
            or row.get("prompt")
            or ""
        )
        statement = f"{title}\n\n{content}".strip() if title else content
        if not statement:
            return None

        platform = str(row.get("platform") or row.get("source") or "").lower()
        # Soft filter: skip obvious non-python markers in title/content if tagged
        lang = str(row.get("language") or row.get("lang") or "python").lower()
        if lang and lang not in {"python", "py", "python3", ""}:
            if lang in {"cpp", "c++", "java", "javascript", "go", "rust"}:
                return None

        starter = str(row.get("starter_code") or row.get("starter") or "")
        # Default starter if empty
        if not starter.strip():
            starter = (
                "# Write a complete Python solution.\n"
                "# Read from stdin and write to stdout if required.\n"
            )

        public_raw = row.get("public_test_cases") or row.get("public_tests") or "[]"
        public_cases = _parse_json_field(public_raw)
        # Store only list-like public cases for the harness
        if not isinstance(public_cases, list):
            public_cases = []

        # Drop binary/encrypted private blobs from workspace
        public_json = json.dumps(public_cases, ensure_ascii=False)

        # LeetCode-style metadata often holds func_name as a JSON string field.
        raw_meta = _parse_json_field(row.get("metadata") or {})
        if not isinstance(raw_meta, dict):
            raw_meta = {}
        func_name = (
            raw_meta.get("func_name")
            or raw_meta.get("function_name")
            or raw_meta.get("method_name")
            or ""
        )
        # Infer method from starter: "def foo(self," inside class Solution
        if not func_name and "def " in starter:
            import re as _re

            m = _re.search(
                r"class\s+Solution\s*:[^\n]*\n(?:.*\n)*?\s+def\s+(\w+)\s*\(",
                starter,
            )
            if m:
                func_name = m.group(1)

        # Ensure typing imports for common LeetCode starters
        if "List[" in starter and "from typing" not in starter:
            starter = "from typing import List\n\n" + starter

        meta_json = json.dumps(
            {"func_name": func_name, "platform": platform},
            ensure_ascii=False,
        )

        workspace_files = {
            "solution.py": starter,
            "public_tests.json": public_json,
            "meta.json": meta_json,
            "test_solution.py": _PYTEST_HARNESS,
        }

        contest_date = str(
            row.get("contest_date") or row.get("date") or row.get("contestDate") or ""
        )
        difficulty = str(row.get("difficulty") or "")

        return CodingTask(
            task_id=f"lcb-{task_id}",
            problem_statement=statement,
            repo_url=None,
            starter_code=starter,
            entry_point=func_name or None,
            test_command="python -m pytest test_solution.py -q --tb=line",
            gold_patch=None,
            metadata={
                "workspace_files": workspace_files,
                "solution_file": "solution.py",
                "platform": platform,
                "contest_date": contest_date,
                "difficulty": difficulty,
                "source": "livecodebench",
                "func_name": func_name,
                "raw_question_id": task_id,
            },
        )


def _parse_json_field(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            # some LCB dumps double-encode
            try:
                return json.loads(json.loads(f'"{value}"'))
            except Exception:
                return []
    return []
