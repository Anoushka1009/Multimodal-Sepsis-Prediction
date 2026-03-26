from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd


def _normalize_table_name(path: Path) -> str | None:
    name = path.name
    if name.endswith(".csv.gz"):
        return name[:-3]
    if name.endswith(".csv"):
        return name
    return None


def resolve_table_path(extracted_dir: str | Path, table_name: str) -> Path:
    extracted_dir = Path(extracted_dir)
    direct_candidates = [extracted_dir / table_name]
    if not table_name.endswith(".gz"):
        direct_candidates.append(extracted_dir / f"{table_name}.gz")

    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    matches = []
    for pattern in [table_name, f"{table_name}.gz"]:
        matches.extend(path for path in extracted_dir.rglob(pattern) if path.is_file())

    if matches:
        matches = sorted(
            set(matches),
            key=lambda path: (len(path.relative_to(extracted_dir).parts), str(path)),
        )
        return matches[0]

    available = list_available_tables(extracted_dir)
    raise FileNotFoundError(
        f"Could not find table {table_name} under {extracted_dir}. "
        f"Available tables include: {available[:10]}"
    )


def list_available_tables(extracted_dir: str | Path) -> List[str]:
    extracted_dir = Path(extracted_dir)
    if not extracted_dir.exists():
        return []
    available = {
        normalized
        for path in extracted_dir.rglob("*")
        if path.is_file()
        for normalized in [_normalize_table_name(path)]
        if normalized is not None
    }
    return sorted(available)


def validate_required_tables(extracted_dir: str | Path, required_tables: Iterable[str]) -> Dict[str, bool]:
    available = set(list_available_tables(extracted_dir))
    return {table: table in available for table in required_tables}


def load_table(
    extracted_dir: str | Path,
    table_name: str,
    usecols: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    chunksize: Optional[int] = None,
    low_memory: bool = True,
    **kwargs,
):
    table_path = resolve_table_path(extracted_dir, table_name)
    return pd.read_csv(
        table_path,
        usecols=usecols,
        nrows=nrows,
        chunksize=chunksize,
        low_memory=low_memory,
        **kwargs,
    )


def iter_table_chunks(
    extracted_dir: str | Path,
    table_name: str,
    usecols: Optional[List[str]] = None,
    chunksize: int = 100000,
    low_memory: bool = True,
    **kwargs,
) -> Iterator[pd.DataFrame]:
    return load_table(
        extracted_dir=extracted_dir,
        table_name=table_name,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=low_memory,
        **kwargs,
    )
