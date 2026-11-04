"""Prompt source dispatcher.

Bench `prompt_source` syntax:
    synthetic              — token-count-controlled filler (default; uses
                              StaticScenario.prompt_tokens)
    builtin:<name>         — load `pping_lang/bench/prompts/data/<name>.jsonl`
                              (mixed-short / mixed-long / code currently)
    file:<path>            — load a user-provided JSONL file
                              (each line: `{"prompt": "..."}`)

Loaded prompts are a list[str]. The runner cycles through them in order, so a
50-entry dataset run for 200 requests reuses each entry 4×. This is fine for
shape-controlled benching; for trace replay see v0.2 docs/bench-design §8.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

# Bundled datasets — listed for `available_builtins()` and for the UI dropdown.
# Anything in pping_lang/bench/prompts/data/<NAME>.jsonl is loadable as
# `builtin:<NAME>` regardless of this list; this constant only governs what
# the UI / discovery surface advertises.
BUILTIN_DATASETS: tuple[str, ...] = ("mixed-short", "mixed-long", "code")


def load_prompts(source: str, target_tokens: int = 500) -> list[str]:
    """Resolve a prompt_source string to a non-empty list of prompts.

    Raises ValueError on unknown source kind, missing builtin name, missing
    file, or empty file. Always returns at least one prompt on success.
    """
    if not source or source == "synthetic":
        # 惰性导入:client.py 顶层要 httpx(仅 extras 提供),别连坐无 extras 的安装
        from pping_lang.bench.client import synthesize_prompt  # noqa: PLC0415

        return [synthesize_prompt(target_tokens)]
    if source.startswith("builtin:"):
        name = source[len("builtin:"):].strip()
        return _load_builtin(name)
    if source.startswith("file:"):
        path = source[len("file:"):].strip()
        return _load_jsonl_file(Path(path))
    raise ValueError(
        f"unknown prompt_source {source!r}; expected one of: "
        f"'synthetic', 'builtin:<name>', 'file:<path>'"
    )


def available_builtins() -> tuple[str, ...]:
    """For UI dropdown / `/api/bench/prompt-sources` style discovery."""
    return BUILTIN_DATASETS


def _load_builtin(name: str) -> list[str]:
    if not name:
        raise ValueError("builtin name is empty (use e.g. 'builtin:mixed-short')")
    try:
        ref = resources.files("pping_lang.bench.prompts.data").joinpath(f"{name}.jsonl")
    except (ModuleNotFoundError, FileNotFoundError) as e:
        raise ValueError(f"builtin dataset {name!r} not found: {e}")
    if not ref.is_file():
        raise ValueError(
            f"builtin dataset {name!r} not found; available: {BUILTIN_DATASETS}"
        )
    with resources.as_file(ref) as p:
        return _parse_jsonl(Path(p), source_label=f"builtin:{name}")


def _load_jsonl_file(path: Path) -> list[str]:
    if not path.is_file():
        raise ValueError(f"prompt file does not exist: {path}")
    return _parse_jsonl(path, source_label=f"file:{path}")


def _parse_jsonl(path: Path, *, source_label: str) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{source_label}:{lineno}: not valid JSON ({e})"
                )
            if not isinstance(obj, dict) or "prompt" not in obj:
                raise ValueError(
                    f"{source_label}:{lineno}: each line must be a JSON object "
                    f"with a 'prompt' key"
                )
            text = obj["prompt"]
            if not isinstance(text, str) or not text:
                raise ValueError(
                    f"{source_label}:{lineno}: 'prompt' must be a non-empty string"
                )
            prompts.append(text)
    if not prompts:
        raise ValueError(f"{source_label}: file has zero valid prompts")
    return prompts
