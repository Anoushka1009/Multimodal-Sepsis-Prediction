from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Iterable, List, Optional


def find_zip_candidates(search_root: str | Path, filename_patterns: Optional[Iterable[str]] = None) -> List[Path]:
    search_root = Path(search_root)
    patterns = list(filename_patterns or [])
    candidates = []
    for path in search_root.rglob("*.zip"):
        if not patterns or any(pattern in path.name for pattern in patterns):
            candidates.append(path)
    return sorted(candidates)


def unzip_dataset(zip_path: str | Path, destination_dir: str | Path, overwrite: bool = False) -> List[str]:
    zip_path = Path(zip_path)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    extracted_members: List[str] = []
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target_path = destination_dir / member.filename
            if target_path.exists() and not overwrite:
                extracted_members.append(member.filename)
                continue
            archive.extract(member, destination_dir)
            extracted_members.append(member.filename)
    return extracted_members
