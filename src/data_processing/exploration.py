from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
from pandas.errors import DtypeWarning

from .io import iter_table_chunks, load_table


def _safe_value_count(series: pd.Series, top_k: int = 10) -> Dict[str, int]:
    counts = series.astype("string").fillna("<MISSING>").value_counts(dropna=False).head(top_k)
    return {str(index): int(value) for index, value in counts.items()}


@contextmanager
def _suppress_dtype_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DtypeWarning)
        yield


def infer_schema_preview(
    extracted_dir: str | Path,
    table_name: str,
    preview_rows: int = 5,
    low_memory: bool = True,
) -> pd.DataFrame:
    with _suppress_dtype_warning():
        preview = load_table(
            extracted_dir=extracted_dir,
            table_name=table_name,
            nrows=preview_rows,
            low_memory=low_memory,
        )
    rows = []
    for column in preview.columns:
        rows.append(
            {
                "table_name": table_name,
                "column_name": column,
                "dtype_preview": str(preview[column].dtype),
                "non_null_preview": int(preview[column].notna().sum()),
                "preview_example": "" if preview.empty else str(preview[column].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def summarize_table_basic(
    extracted_dir: str | Path,
    table_name: str,
    chunksize: int = 100000,
    low_memory: bool = True,
    id_columns: Optional[Iterable[str]] = None,
) -> Dict[str, object]:
    row_count = 0
    chunk_count = 0
    unique_tracker: Dict[str, set] = {column: set() for column in (id_columns or [])}
    column_count = None

    with _suppress_dtype_warning():
        for chunk in iter_table_chunks(
            extracted_dir=extracted_dir,
            table_name=table_name,
            chunksize=chunksize,
            low_memory=low_memory,
        ):
            chunk_count += 1
            row_count += len(chunk)
            if column_count is None:
                column_count = len(chunk.columns)

            for column in unique_tracker:
                if column in chunk.columns:
                    unique_tracker[column].update(chunk[column].dropna().astype(str).unique().tolist())

    summary = {
        "table_name": table_name,
        "row_count": int(row_count),
        "column_count": int(column_count or 0),
        "chunk_count": int(chunk_count),
    }
    for column, values in unique_tracker.items():
        summary[f"unique_{column.lower()}"] = int(len(values))
    return summary


def estimate_missingness(
    extracted_dir: str | Path,
    table_name: str,
    sample_rows: int = 50000,
    low_memory: bool = True,
) -> pd.DataFrame:
    with _suppress_dtype_warning():
        sample = load_table(
            extracted_dir=extracted_dir,
            table_name=table_name,
            nrows=sample_rows,
            low_memory=low_memory,
        )
    if sample.empty:
        return pd.DataFrame(
            columns=["table_name", "column_name", "missing_fraction_sample", "non_null_count_sample"]
        )

    rows = []
    sample_size = len(sample)
    for column in sample.columns:
        non_null = int(sample[column].notna().sum())
        rows.append(
            {
                "table_name": table_name,
                "column_name": column,
                "missing_fraction_sample": float(1.0 - (non_null / sample_size)),
                "non_null_count_sample": non_null,
            }
        )
    return pd.DataFrame(rows)


def summarize_note_categories(
    extracted_dir: str | Path,
    table_name: str = "NOTEEVENTS.csv",
    category_column: str = "CATEGORY",
    text_column: str = "TEXT",
    time_column: str = "CHARTTIME",
    chunksize: int = 100000,
    low_memory: bool = True,
    top_k: int = 15,
) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    note_rows = 0
    timed_rows = 0
    text_rows = 0

    with _suppress_dtype_warning():
        for chunk in iter_table_chunks(
            extracted_dir=extracted_dir,
            table_name=table_name,
            usecols=[column for column in [category_column, text_column, time_column] if column],
            chunksize=chunksize,
            low_memory=low_memory,
        ):
            note_rows += len(chunk)
            if time_column in chunk.columns:
                timed_rows += int(chunk[time_column].notna().sum())
            if text_column in chunk.columns:
                text_rows += int(chunk[text_column].notna().sum())

            if category_column in chunk.columns:
                category_counts = chunk[category_column].astype("string").fillna("<MISSING>").value_counts()
                for key, value in category_counts.items():
                    counts[str(key)] = counts.get(str(key), 0) + int(value)

    rows = [
        {
            "category": category,
            "note_count": count,
            "fraction_of_notes": (count / note_rows) if note_rows else 0.0,
            "timed_note_fraction": (timed_rows / note_rows) if note_rows else 0.0,
            "non_empty_text_fraction": (text_rows / note_rows) if note_rows else 0.0,
        }
        for category, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_k]
    ]
    return pd.DataFrame(rows)


def summarize_demographics(
    extracted_dir: str | Path,
    patients_table: str = "PATIENTS.csv",
    admissions_table: str = "ADMISSIONS.csv",
    low_memory: bool = True,
) -> Dict[str, object]:
    with _suppress_dtype_warning():
        patients = load_table(
            extracted_dir=extracted_dir,
            table_name=patients_table,
            usecols=["SUBJECT_ID", "GENDER", "DOB"],
            low_memory=low_memory,
        )
        admissions = load_table(
            extracted_dir=extracted_dir,
            table_name=admissions_table,
            usecols=["SUBJECT_ID", "HADM_ID", "ETHNICITY", "ADMITTIME"],
            low_memory=low_memory,
        )

    merged = admissions.merge(patients, on="SUBJECT_ID", how="left")
    gender_counts = _safe_value_count(merged["GENDER"], top_k=10) if "GENDER" in merged else {}
    ethnicity_counts = _safe_value_count(merged["ETHNICITY"], top_k=10) if "ETHNICITY" in merged else {}

    return {
        "patient_count": int(patients["SUBJECT_ID"].nunique()) if "SUBJECT_ID" in patients else 0,
        "admission_count": int(admissions["HADM_ID"].nunique()) if "HADM_ID" in admissions else 0,
        "gender_distribution": gender_counts,
        "ethnicity_distribution": ethnicity_counts,
    }


def build_exploration_bundle(
    extracted_dir: str | Path,
    table_names: Iterable[str],
    preview_rows: int,
    sample_rows: int,
    chunksize: int,
    low_memory: bool,
    id_columns: Optional[Iterable[str]] = None,
    note_category_top_k: int = 15,
) -> Dict[str, pd.DataFrame]:
    schema_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []
    missingness_frames: List[pd.DataFrame] = []

    for table_name in table_names:
        schema_frames.append(
            infer_schema_preview(
                extracted_dir=extracted_dir,
                table_name=table_name,
                preview_rows=preview_rows,
                low_memory=low_memory,
            )
        )
        summary_rows.append(
            summarize_table_basic(
                extracted_dir=extracted_dir,
                table_name=table_name,
                chunksize=chunksize,
                low_memory=low_memory,
                id_columns=id_columns,
            )
        )
        missingness_frames.append(
            estimate_missingness(
                extracted_dir=extracted_dir,
                table_name=table_name,
                sample_rows=sample_rows,
                low_memory=low_memory,
            )
        )

    bundle = {
        "schema_preview": pd.concat(schema_frames, ignore_index=True) if schema_frames else pd.DataFrame(),
        "table_summary": pd.DataFrame(summary_rows),
        "missingness_sample": pd.concat(missingness_frames, ignore_index=True) if missingness_frames else pd.DataFrame(),
        "note_category_summary": summarize_note_categories(
            extracted_dir=extracted_dir,
            chunksize=chunksize,
            low_memory=low_memory,
            top_k=note_category_top_k,
        ),
    }
    return bundle
