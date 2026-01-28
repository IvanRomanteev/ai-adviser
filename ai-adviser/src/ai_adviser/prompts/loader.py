from __future__ import annotations

from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=16)
def load_prompt(name: str) -> str:
    """
    Load prompt text from ai_adviser/prompts as a package resource.
    """
    return resources.files("ai_adviser.prompts") \
        .joinpath(name) \
        .read_text(encoding="utf-8")
