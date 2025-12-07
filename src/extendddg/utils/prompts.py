from __future__ import annotations

from copy import deepcopy
from importlib import resources
from typing import Any, Dict

import yaml
from autoddg.utils import load_prompts as load_autoddg_prompts
from beartype import beartype

_PROMPT_CACHE: Dict[str, Any] | None = None


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base.get(key, {}), value)
        else:
            base[key] = value
    return base


@beartype
def load_prompts() -> Dict[str, Any]:
    """Load AutoDDG prompts and merge ExtendDDG overrides."""

    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE

    base = deepcopy(load_autoddg_prompts())
    override: Dict[str, Any] = {}

    try:
        with (
            resources.files("extendddg.config")
            .joinpath("prompts.yaml")
            .open("r", encoding="utf-8") as stream
        ):
            override = yaml.safe_load(stream) or {}
    except FileNotFoundError:
        override = {}

    _PROMPT_CACHE = _deep_update(base, override)
    return _PROMPT_CACHE
