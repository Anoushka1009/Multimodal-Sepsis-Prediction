from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.utils.config import load_config
from src.utils.paths import ensure_directories, resolve_project_paths


@dataclass(frozen=True)
class ProjectRuntime:
    in_colab: bool
    project_root: Path
    config: Dict[str, Any]
    paths: Dict[str, Path]


def is_running_in_colab() -> bool:
    return "google.colab" in sys.modules


def find_project_root(start: str | Path | None = None) -> Path:
    candidates = []

    env_root = os.environ.get("MESP_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())

    if start is not None:
        start_path = Path(start).expanduser().resolve()
        candidates.extend([start_path, *start_path.parents])

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    module_path = Path(__file__).resolve()
    candidates.extend([module_path.parent, *module_path.parents])

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "configs" / "base.yaml").exists() and (candidate / "src").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the repository root. Set MESP_PROJECT_ROOT or run from inside the project."
    )


def ensure_project_on_sys_path(project_root: str | Path | None = None) -> Path:
    resolved_root = find_project_root(project_root)
    root_str = str(resolved_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return resolved_root


def maybe_mount_colab_drive(mount_point: str = "/content/drive") -> None:
    if not is_running_in_colab():
        return

    from google.colab import drive

    drive.mount(mount_point)


def load_project_runtime(
    *,
    start: str | Path | None = None,
    mount_colab_drive: bool = True,
    override_path: str | Path | None = None,
) -> ProjectRuntime:
    in_colab = is_running_in_colab()
    if in_colab and mount_colab_drive:
        maybe_mount_colab_drive()

    project_root = ensure_project_on_sys_path(start)
    base_config_path = project_root / "configs" / "base.yaml"

    if override_path is None and in_colab:
        colab_override = project_root / "configs" / "colab.yaml"
        override_path = colab_override if colab_override.exists() else None

    config = load_config(base_config_path, override_path)
    config.setdefault("paths", {})
    config["paths"]["project_root"] = str(project_root)

    paths = resolve_project_paths(config)
    ensure_directories(paths)
    return ProjectRuntime(
        in_colab=in_colab,
        project_root=project_root,
        config=config,
        paths=paths,
    )
