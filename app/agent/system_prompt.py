"""The grounding contract, loaded from `prompts/system_prompt.md`.

The actual text -- citations, abstention, conflict handling, overrides,
injection resistance, follow-ups -- lives in that file, not here, so it can
be edited without touching Python code. See it for the content; this module
just exposes it as the `SYSTEM_PROMPT` string constant every other module
imports (`app/agent/graph.py`, `evals/run_eval.py`, `tests/test_09_*.py`).
"""
from __future__ import annotations

from app.agent.prompts import load_system_prompt

SYSTEM_PROMPT = load_system_prompt()
