from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def save_dataframe_artifact(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=index)
    elif path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=index)
    else:
        raise ValueError(f"Unsupported dataframe artifact format: {path.suffix}")


def save_dataframe_bundle(bundle: Dict[str, pd.DataFrame], output_dir: str | Path) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, str] = {}
    for artifact_name, df in bundle.items():
        artifact_path = output_dir / f"{artifact_name}.csv"
        df.to_csv(artifact_path, index=False)
        saved_paths[artifact_name] = str(artifact_path)
    return saved_paths
