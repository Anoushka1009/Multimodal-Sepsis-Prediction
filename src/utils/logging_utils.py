from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(payload: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_run_manifest(
    path: str | Path,
    stage: str,
    config: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> None:
    payload = {
        "created_at_utc": utc_now_iso(),
        "stage": stage,
        "config": config,
        "extra": extra or {},
    }
    write_json(payload, path)
