from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def _safe_abs_correlation(feature: pd.Series, target: pd.Series) -> float:
    feature = pd.to_numeric(feature, errors='coerce')
    target = pd.to_numeric(target, errors='coerce')

    valid = feature.notna() & target.notna()
    if valid.sum() < 2:
        return 0.0

    feature_valid = feature.loc[valid]
    target_valid = target.loc[valid]
    if feature_valid.nunique(dropna=True) < 2 or target_valid.nunique(dropna=True) < 2:
        return 0.0

    filled = feature.fillna(feature_valid.median())
    aligned_target = target.loc[filled.index]
    valid = filled.notna() & aligned_target.notna()
    if valid.sum() < 2:
        return 0.0

    filled_valid = filled.loc[valid]
    aligned_target_valid = aligned_target.loc[valid]
    if filled_valid.nunique(dropna=True) < 2 or aligned_target_valid.nunique(dropna=True) < 2:
        return 0.0

    importance = filled_valid.corr(aligned_target_valid)
    return 0.0 if pd.isna(importance) else float(abs(importance))


def compute_surrogate_feature_importance(tabular_df: pd.DataFrame) -> pd.DataFrame:
    if tabular_df.empty:
        return pd.DataFrame(columns=['feature_name', 'importance'])
    feature_columns = [
        column for column in tabular_df.columns
        if column not in {'SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'split', 'sepsis3_label'}
        and pd.api.types.is_numeric_dtype(tabular_df[column])
    ]
    if not feature_columns:
        return pd.DataFrame(columns=['feature_name', 'importance'])

    correlations = []
    target = tabular_df['sepsis3_label'].astype(float)
    for column in feature_columns:
        importance = _safe_abs_correlation(tabular_df[column], target)
        correlations.append({'feature_name': column, 'importance': float(importance)})
    return pd.DataFrame(correlations).sort_values('importance', ascending=False).reset_index(drop=True)


def derive_temporal_feature_importance(horizon_tables: Dict[str, pd.DataFrame], top_k: int = 20) -> pd.DataFrame:
    rows = []
    for dataset_name, df in horizon_tables.items():
        if df.empty:
            continue
        feature_columns = [
            column for column in df.columns
            if '__' in column and pd.api.types.is_numeric_dtype(df[column])
        ]
        target = df['sepsis3_label'].astype(float)
        for column in feature_columns:
            importance = _safe_abs_correlation(df[column], target)
            rows.append({'dataset_name': dataset_name, 'feature_name': column, 'importance': float(importance)})
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(['dataset_name', 'importance'], ascending=[True, False]).groupby('dataset_name').head(top_k).reset_index(drop=True)


def build_attention_phrase_table(note_windows_df: pd.DataFrame, top_k_phrases: int = 10) -> pd.DataFrame:
    if note_windows_df.empty or 'aggregated_text' not in note_windows_df.columns:
        return pd.DataFrame(columns=['phrase', 'pseudo_attention_score'])

    tokens = []
    for text in note_windows_df['aggregated_text'].astype(str).fillna(''):
        parts = [token.strip('.,:;()[]').lower() for token in text.split()]
        tokens.extend([token for token in parts if len(token) > 3])
    if not tokens:
        return pd.DataFrame(columns=['phrase', 'pseudo_attention_score'])

    counts = pd.Series(tokens).value_counts().head(top_k_phrases)
    total = counts.sum()
    rows = [
        {'phrase': phrase, 'pseudo_attention_score': float(count / total)}
        for phrase, count in counts.items()
    ]
    return pd.DataFrame(rows)


def build_clinical_narrative_table(feature_importance_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    if feature_importance_df.empty:
        return pd.DataFrame(columns=['feature_name', 'clinical_interpretation'])

    narratives = []
    for _, row in feature_importance_df.head(top_k).iterrows():
        feature = row['feature_name']
        if 'heart_rate' in feature:
            interpretation = 'Heart rate changes may reflect early hemodynamic stress and systemic inflammatory response.'
        elif 'map' in feature or 'sbp' in feature or 'dbp' in feature:
            interpretation = 'Blood pressure instability is clinically consistent with shock progression and organ hypoperfusion risk.'
        elif 'creatinine' in feature or 'bun' in feature:
            interpretation = 'Renal dysfunction markers often increase as sepsis affects kidney perfusion and filtration.'
        elif 'bilirubin' in feature:
            interpretation = 'Bilirubin elevations may indicate hepatic dysfunction contributing to SOFA increase.'
        elif 'lactate' in feature:
            interpretation = 'Lactate can indicate tissue hypoperfusion and metabolic stress during sepsis evolution.'
        elif 'wbc' in feature:
            interpretation = 'White blood cell abnormalities are a common signal of infection and inflammatory response.'
        else:
            interpretation = 'This feature may reflect physiologic instability associated with impending sepsis onset.'
        narratives.append({'feature_name': feature, 'clinical_interpretation': interpretation})
    return pd.DataFrame(narratives)
