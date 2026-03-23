from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd


def list_available_tables(extracted_dir: str | Path) -> List[str]:
    extracted_dir = Path(extracted_dir)
    if not extracted_dir.exists():
        return []
    return sorted(path.name for path in extracted_dir.glob("*.csv"))


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
    table_path = Path(extracted_dir) / table_name
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
