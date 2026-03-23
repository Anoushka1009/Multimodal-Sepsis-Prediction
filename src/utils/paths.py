from __future__ import annotations

from pathlib import Path
from typing import Dict


def resolve_project_paths(config: dict) -> Dict[str, Path]:
    path_config = config["paths"]
    project_root = Path(path_config["project_root"]).resolve()

    resolved = {"project_root": project_root}
    for key, value in path_config.items():
        if key == "project_root":
            continue
        path = Path(value)
        resolved[key] = path if path.is_absolute() else (project_root / path)
    return resolved


def ensure_directories(paths: Dict[str, Path]) -> None:
    for path in paths.values():
        if path.suffix:
            continue
        path.mkdir(parents=True, exist_ok=True)
