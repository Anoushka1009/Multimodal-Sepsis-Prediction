from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


ID_COLUMNS = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']
EXCLUDE_COLUMNS = set(ID_COLUMNS + ['hour', 'prediction_time', 'split', 'sepsis3_label', 'sepsis_onset_time'])


def make_stay_level_tabular_dataset(
    horizon_df: pd.DataFrame,
    aggregations: Iterable[str] = ('mean', 'min', 'max', 'last'),
) -> pd.DataFrame:
    if horizon_df.empty:
        return pd.DataFrame()

    feature_columns = [
        column for column in horizon_df.columns
        if column not in EXCLUDE_COLUMNS
        and pd.api.types.is_numeric_dtype(horizon_df[column])
    ]
    if not feature_columns:
        return horizon_df[ID_COLUMNS + ['split', 'sepsis3_label']].drop_duplicates().copy()

    grouped = horizon_df.groupby(ID_COLUMNS, dropna=False)
    frames = []
    for agg in aggregations:
        if agg == 'last':
            frame = grouped[feature_columns].last().reset_index()
        elif agg == 'mean':
            frame = grouped[feature_columns].mean().reset_index()
        elif agg == 'min':
            frame = grouped[feature_columns].min().reset_index()
        elif agg == 'max':
            frame = grouped[feature_columns].max().reset_index()
        else:
            continue
        renamed = {column: f'{column}__{agg}' for column in feature_columns}
        frame = frame.rename(columns=renamed)
        frames.append(frame)

    tabular = frames[0]
    for frame in frames[1:]:
        tabular = tabular.merge(frame, on=ID_COLUMNS, how='inner')

    labels = horizon_df[ID_COLUMNS + ['split', 'sepsis3_label']].drop_duplicates()
    tabular = tabular.merge(labels, on=ID_COLUMNS, how='left')
    return tabular


def split_tabular_dataset(tabular_df: pd.DataFrame) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    result = {}
    for split_name in ['train', 'val', 'test']:
        split_df = tabular_df.loc[tabular_df['split'] == split_name].copy()
        y = split_df['sepsis3_label'].astype(int) if not split_df.empty else pd.Series(dtype=int)
        X = split_df.drop(columns=['sepsis3_label', 'split']) if not split_df.empty else pd.DataFrame()
        result[split_name] = (X, y)
    return result


def build_baseline_models(config: dict) -> Dict[str, object]:
    random_state = config['baselines']['random_state']
    models = {}

    lr_cfg = config['baselines']['logistic_regression']
    models['logistic_regression'] = Pipeline([
        ('imputer', SimpleImputer(strategy=config['baselines']['imputation_strategy'])),
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            max_iter=lr_cfg['max_iter'],
            class_weight=lr_cfg.get('class_weight'),
            random_state=random_state,
        )),
    ])

    rf_cfg = config['baselines']['random_forest']
    models['random_forest'] = Pipeline([
        ('imputer', SimpleImputer(strategy=config['baselines']['imputation_strategy'])),
        ('model', RandomForestClassifier(
            n_estimators=rf_cfg['n_estimators'],
            max_depth=rf_cfg['max_depth'],
            min_samples_leaf=rf_cfg['min_samples_leaf'],
            class_weight=rf_cfg.get('class_weight'),
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    if XGBClassifier is not None:
        xgb_cfg = config['baselines']['xgboost']
        models['xgboost'] = Pipeline([
            ('imputer', SimpleImputer(strategy=config['baselines']['imputation_strategy'])),
            ('model', XGBClassifier(
                n_estimators=xgb_cfg['n_estimators'],
                max_depth=xgb_cfg['max_depth'],
                learning_rate=xgb_cfg['learning_rate'],
                subsample=xgb_cfg['subsample'],
                colsample_bytree=xgb_cfg['colsample_bytree'],
                random_state=random_state,
                eval_metric='logloss',
                n_jobs=2,
            )),
        ])

    allowed = set(config['baselines']['models'])
    return {name: model for name, model in models.items() if name in allowed}


def fit_and_predict_baseline(model, train_X, train_y, eval_X):
    feature_cols = [col for col in train_X.columns if col not in ID_COLUMNS]
    model.fit(train_X[feature_cols], train_y)
    proba = model.predict_proba(eval_X[feature_cols])[:, 1]
    return proba, feature_cols
