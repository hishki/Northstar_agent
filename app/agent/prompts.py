"""Loads model-facing prompt text from the `prompts/` directory at the repo
root, so prompt content can be edited without touching Python code.

`app/agent/system_prompt.py` exposes the loaded system prompt as the plain
`SYSTEM_PROMPT` string constant every other module already imports;
`load_tool_descriptions()` is applied to each tool's `.description` in
`app/agent/tools.py` before it's bound to the chat model.
"""
from __future__ import annotations

from pathlib import Path

import yaml

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_system_prompt() -> str:
    return (_PROMPTS_DIR / "system_prompt.md").read_text().strip() + "\n"


def load_tool_descriptions() -> dict[str, str]:
    raw = yaml.safe_load((_PROMPTS_DIR / "tool_descriptions.yaml").read_text()) or {}
    return {name: text.strip() for name, text in raw.items()}
