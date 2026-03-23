from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(base_path: str | Path, override_path: str | Path | None = None) -> Dict[str, Any]:
    config = load_yaml(base_path)
    if override_path is not None and Path(override_path).exists():
        config = _deep_update(config, load_yaml(override_path))
    return config
