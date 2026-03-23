from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from .cohort import ID_COLUMNS
from .io import iter_table_chunks, load_table
from .sepsis3 import attach_icustay_ids, resolve_itemids_by_keywords


STATIC_COLUMNS = ["age_at_icu_intime", "GENDER", "ETHNICITY", "FIRST_CAREUNIT", "LAST_CAREUNIT"]


def extract_feature_measurements(
    extracted_dir: str | Path,
    cohort: pd.DataFrame,
    chart_feature_keywords: Dict[str, List[str]],
    lab_feature_keywords: Dict[str, List[str]],
    chunksize: int = 100000,
    low_memory: bool = True,
) -> Dict[str, pd.DataFrame]:
    chart_itemids = resolve_itemids_by_keywords(extracted_dir, 'D_ITEMS.csv', chart_feature_keywords)
    lab_itemids = resolve_itemids_by_keywords(extracted_dir, 'D_LABITEMS.csv', lab_feature_keywords)

    chart_events = _extract_feature_events(
        extracted_dir=extracted_dir,
        table_name='CHARTEVENTS.csv',
        itemids_by_feature=chart_itemids,
        time_column='CHARTTIME',
        value_column='VALUENUM',
        chunksize=chunksize,
        low_memory=low_memory,
        has_icustay_id=True,
    )
    if not chart_events.empty:
        missing = chart_events['ICUSTAY_ID'].isna()
        if missing.any():
            repaired = attach_icustay_ids(chart_events.loc[missing].drop(columns=['ICUSTAY_ID']), cohort, time_column='charttime')
            chart_events = pd.concat([chart_events.loc[~missing], repaired], ignore_index=True)

    lab_events = _extract_feature_events(
        extracted_dir=extracted_dir,
        table_name='LABEVENTS.csv',
        itemids_by_feature=lab_itemids,
        time_column='CHARTTIME',
        value_column='VALUENUM',
        chunksize=chunksize,
        low_memory=low_memory,
        has_icustay_id=False,
    )
    if not lab_events.empty:
        lab_events = attach_icustay_ids(lab_events.drop(columns=['ICUSTAY_ID']), cohort, time_column='charttime')

    return {
        'chart_events': chart_events,
        'lab_events': lab_events,
        'chart_itemids': pd.DataFrame([
            {'feature_name': feature_name, 'itemid': itemid}
            for feature_name, itemids in chart_itemids.items()
            for itemid in itemids
        ]),
        'lab_itemids': pd.DataFrame([
            {'feature_name': feature_name, 'itemid': itemid}
            for feature_name, itemids in lab_itemids.items()
            for itemid in itemids
        ]),
    }


def _extract_feature_events(
    extracted_dir: str | Path,
    table_name: str,
    itemids_by_feature: Dict[str, List[int]],
    time_column: str,
    value_column: str,
    chunksize: int,
    low_memory: bool,
    has_icustay_id: bool,
) -> pd.DataFrame:
    all_itemids = sorted({itemid for itemids in itemids_by_feature.values() for itemid in itemids})
    if not all_itemids:
        return pd.DataFrame(columns=ID_COLUMNS + ['charttime', 'feature_name', 'value'])

    reverse_map = {itemid: feature_name for feature_name, itemids in itemids_by_feature.items() for itemid in itemids}
    usecols = ['SUBJECT_ID', 'HADM_ID', 'ITEMID', time_column, value_column]
    if has_icustay_id:
        usecols.insert(2, 'ICUSTAY_ID')

    event_frames: List[pd.DataFrame] = []
    for chunk in iter_table_chunks(
        extracted_dir=extracted_dir,
        table_name=table_name,
        usecols=usecols,
        chunksize=chunksize,
        low_memory=low_memory,
    ):
        filtered = chunk.loc[chunk['ITEMID'].isin(all_itemids)].copy()
        if filtered.empty:
            continue
        if 'ICUSTAY_ID' not in filtered.columns:
            filtered['ICUSTAY_ID'] = np.nan
        filtered['charttime'] = pd.to_datetime(filtered[time_column], errors='coerce')
        filtered['value'] = pd.to_numeric(filtered[value_column], errors='coerce')
        filtered = filtered.dropna(subset=['charttime', 'value'])
        filtered['feature_name'] = filtered['ITEMID'].map(reverse_map)
        event_frames.append(filtered[ID_COLUMNS + ['charttime', 'feature_name', 'value']])

    if not event_frames:
        return pd.DataFrame(columns=ID_COLUMNS + ['charttime', 'feature_name', 'value'])
    return pd.concat(event_frames, ignore_index=True)


def aggregate_events_hourly(
    events: pd.DataFrame,
    aggregate_functions: Iterable[str],
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=ID_COLUMNS + ['hour'])

    events = events.copy()
    events['hour'] = pd.to_datetime(events['charttime'], errors='coerce').dt.floor('H')
    events = events.dropna(subset=['hour', 'ICUSTAY_ID'])

    grouped = events.groupby(ID_COLUMNS + ['hour', 'feature_name'])['value']
    frames = []
    for func in aggregate_functions:
        if func == 'last':
            series = grouped.last().rename('value')
        elif func == 'mean':
            series = grouped.mean().rename('value')
        elif func == 'min':
            series = grouped.min().rename('value')
        elif func == 'max':
            series = grouped.max().rename('value')
        else:
            continue
        frame = series.reset_index()
        frame['feature_stat'] = frame['feature_name'] + '__' + func
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ID_COLUMNS + ['hour', 'feature_stat', 'value'])
    if combined.empty:
        return pd.DataFrame(columns=ID_COLUMNS + ['hour'])

    wide = combined.pivot_table(
        index=ID_COLUMNS + ['hour'],
        columns='feature_stat',
        values='value',
        aggfunc='first',
    ).reset_index()
    wide.columns.name = None
    return wide


def build_hourly_feature_table(
    cohort: pd.DataFrame,
    chart_hourly: pd.DataFrame,
    lab_hourly: pd.DataFrame,
) -> pd.DataFrame:
    hourly = pd.merge(
        chart_hourly,
        lab_hourly,
        on=ID_COLUMNS + ['hour'],
        how='outer',
    )
    if hourly.empty:
        return pd.DataFrame(columns=ID_COLUMNS + ['hour'])

    static_cols = [column for column in STATIC_COLUMNS if column in cohort.columns]
    cohort_static = cohort[ID_COLUMNS + ['INTIME', 'OUTTIME'] + static_cols].drop_duplicates().copy()
    hourly = hourly.merge(cohort_static, on=ID_COLUMNS, how='left')
    hourly = hourly.sort_values(ID_COLUMNS + ['hour']).reset_index(drop=True)
    hourly['hours_since_icu_admit'] = (hourly['hour'] - hourly['INTIME']).dt.total_seconds() / 3600.0
    return hourly


def build_horizon_prediction_rows(
    hourly_features: pd.DataFrame,
    labels: pd.DataFrame,
    horizons_hours: Iterable[int],
    history_window_hours: int = 48,
    min_history_hours: int = 6,
) -> Dict[str, pd.DataFrame]:
    if hourly_features.empty:
        return {f'horizon_{int(h)}h': pd.DataFrame() for h in horizons_hours}

    labels = labels.copy()
    labels['sepsis_onset_time'] = pd.to_datetime(labels['sepsis_onset_time'], errors='coerce')
    merged = hourly_features.merge(
        labels[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'sepsis_onset_time', 'sepsis3_label']],
        on=ID_COLUMNS,
        how='left',
    )

    outputs: Dict[str, pd.DataFrame] = {}
    for horizon in horizons_hours:
        horizon = int(horizon)
        frame = merged.copy()
        frame['prediction_time'] = np.where(
            frame['sepsis3_label'] == 1,
            frame['sepsis_onset_time'] - pd.to_timedelta(horizon, unit='h'),
            frame['OUTTIME'] - pd.to_timedelta(horizon, unit='h'),
        )
        frame['prediction_time'] = pd.to_datetime(frame['prediction_time'], errors='coerce')
        frame = frame.loc[frame['prediction_time'].notna()].copy()
        frame = frame.loc[frame['prediction_time'] >= frame['INTIME'] + pd.to_timedelta(min_history_hours, unit='h')].copy()
        frame = frame.loc[frame['hour'] <= frame['prediction_time']].copy()
        frame = frame.loc[frame['hour'] >= frame['prediction_time'] - pd.to_timedelta(history_window_hours, unit='h')].copy()
        frame['hours_to_prediction'] = (frame['prediction_time'] - frame['hour']).dt.total_seconds() / 3600.0
        frame['prediction_horizon_hours'] = horizon
        frame = frame.sort_values(ID_COLUMNS + ['hour']).reset_index(drop=True)
        outputs[f'horizon_{horizon}h'] = frame
    return outputs


def summarize_horizon_rows(horizon_rows: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in horizon_rows.items():
        rows.append({
            'dataset_name': name,
            'row_count': int(len(df)),
            'icu_stay_count': int(df['ICUSTAY_ID'].nunique()) if not df.empty and 'ICUSTAY_ID' in df else 0,
            'positive_stay_count': int(df.loc[df['sepsis3_label'] == 1, 'ICUSTAY_ID'].nunique()) if not df.empty and 'sepsis3_label' in df else 0,
            'feature_column_count': int(len([c for c in df.columns if '__' in c])),
        })
    return pd.DataFrame(rows)
