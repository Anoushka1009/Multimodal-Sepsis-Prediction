from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .cohort import ID_COLUMNS
from .io import iter_table_chunks
from .sepsis3 import attach_icustay_ids


WHITESPACE_RE = re.compile(r"\s+")


def clean_note_text(text: str, max_characters: int = 4000) -> str:
    if text is None:
        return ""
    cleaned = WHITESPACE_RE.sub(" ", str(text)).strip()
    return cleaned[:max_characters]


def load_and_filter_notes(
    extracted_dir: str | Path,
    cohort: pd.DataFrame,
    note_categories: Iterable[str],
    note_columns: Iterable[str],
    min_note_characters: int = 20,
    max_note_characters: int = 4000,
    max_notes_per_stay: int = 200,
    chunksize: int = 100000,
    low_memory: bool = True,
) -> pd.DataFrame:
    category_set = {category.lower() for category in note_categories}
    note_frames: List[pd.DataFrame] = []

    for chunk in iter_table_chunks(
        extracted_dir=extracted_dir,
        table_name='NOTEEVENTS.csv',
        usecols=list(note_columns),
        chunksize=chunksize,
        low_memory=low_memory,
    ):
        if 'CATEGORY' in chunk.columns:
            category_mask = chunk['CATEGORY'].astype(str).str.lower().isin(category_set)
            chunk = chunk.loc[category_mask].copy()
        if chunk.empty:
            continue

        time_source = 'CHARTTIME' if 'CHARTTIME' in chunk.columns else 'CHARTDATE'
        chunk['note_time'] = pd.to_datetime(chunk[time_source], errors='coerce')
        chunk = chunk.dropna(subset=['SUBJECT_ID', 'HADM_ID', 'note_time', 'TEXT']).copy()
        chunk['clean_text'] = chunk['TEXT'].astype(str).map(lambda x: clean_note_text(x, max_characters=max_note_characters))
        chunk['text_length'] = chunk['clean_text'].str.len()
        chunk = chunk.loc[chunk['text_length'] >= int(min_note_characters)].copy()
        if chunk.empty:
            continue

        note_frames.append(chunk)

    if not note_frames:
        return pd.DataFrame(columns=ID_COLUMNS + ['note_time', 'CATEGORY', 'DESCRIPTION', 'clean_text', 'text_length'])

    notes = pd.concat(note_frames, ignore_index=True)
    notes = attach_icustay_ids(notes.drop(columns=['TEXT'], errors='ignore'), cohort, time_column='note_time')
    notes = notes.sort_values(ID_COLUMNS + ['note_time']).reset_index(drop=True)
    notes['note_rank_within_stay'] = notes.groupby('ICUSTAY_ID').cumcount() + 1
    notes = notes.loc[notes['note_rank_within_stay'] <= int(max_notes_per_stay)].copy()
    return notes


def build_horizon_note_rows(
    notes: pd.DataFrame,
    horizon_structured_rows: pd.DataFrame,
    aggregation_window_hours: int = 6,
) -> pd.DataFrame:
    if notes.empty or horizon_structured_rows.empty:
        return pd.DataFrame()

    stay_cutoffs = horizon_structured_rows.groupby(ID_COLUMNS, as_index=False).agg(
        prediction_time=('prediction_time', 'max'),
        sepsis3_label=('sepsis3_label', 'max'),
        split=('split', 'first') if 'split' in horizon_structured_rows.columns else ('prediction_time', 'size'),
    )
    merged = notes.merge(stay_cutoffs, on=ID_COLUMNS, how='inner')
    merged = merged.loc[merged['note_time'] <= merged['prediction_time']].copy()
    if merged.empty:
        return merged

    merged['hours_before_prediction'] = (merged['prediction_time'] - merged['note_time']).dt.total_seconds() / 3600.0
    merged['note_window_index'] = (merged['hours_before_prediction'] // aggregation_window_hours).astype(int)
    merged = merged.sort_values(ID_COLUMNS + ['note_time']).reset_index(drop=True)
    return merged


def aggregate_notes_by_window(note_rows: pd.DataFrame) -> pd.DataFrame:
    if note_rows.empty:
        return pd.DataFrame()

    grouped = note_rows.groupby(ID_COLUMNS + ['prediction_time', 'note_window_index', 'split', 'sepsis3_label'], dropna=False)
    rows = []
    for key, group in grouped:
        subject_id, hadm_id, icustay_id, prediction_time, window_index, split, label = key
        ordered = group.sort_values('note_time')
        rows.append(
            {
                'SUBJECT_ID': subject_id,
                'HADM_ID': hadm_id,
                'ICUSTAY_ID': icustay_id,
                'prediction_time': prediction_time,
                'note_window_index': int(window_index),
                'split': split,
                'sepsis3_label': int(label),
                'note_count': int(len(ordered)),
                'first_note_time': ordered['note_time'].min(),
                'last_note_time': ordered['note_time'].max(),
                'categories': ' | '.join(ordered['CATEGORY'].astype(str).fillna('UNKNOWN').tolist()),
                'aggregated_text': ' '.join(ordered['clean_text'].tolist()),
            }
        )
    return pd.DataFrame(rows).sort_values(ID_COLUMNS + ['note_window_index']).reset_index(drop=True)


def summarize_note_coverage(horizon_notes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in horizon_notes.items():
        rows.append(
            {
                'dataset_name': name,
                'note_row_count': int(len(df)),
                'icu_stay_count': int(df['ICUSTAY_ID'].nunique()) if not df.empty and 'ICUSTAY_ID' in df else 0,
                'positive_stay_count': int(df.loc[df['sepsis3_label'] == 1, 'ICUSTAY_ID'].nunique()) if not df.empty and 'sepsis3_label' in df else 0,
                'mean_notes_per_stay': float(df.groupby('ICUSTAY_ID').size().mean()) if not df.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)
