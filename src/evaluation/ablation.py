from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.baselines import (
    build_baseline_models,
    fit_and_predict_baseline,
    make_stay_level_tabular_dataset,
    split_tabular_dataset,
)


VITAL_KEYWORDS = ['heart_rate', 'sbp', 'dbp', 'map', 'respiratory_rate', 'temperature_c', 'spo2', 'glucose_chart']
LAB_KEYWORDS = ['wbc', 'hemoglobin', 'creatinine', 'platelet', 'bilirubin', 'lactate', 'sodium', 'potassium', 'bicarbonate', 'bun']


def classify_feature_family(column_name: str) -> str:
    lower = column_name.lower()
    if any(keyword in lower for keyword in VITAL_KEYWORDS):
        return 'vitals'
    if any(keyword in lower for keyword in LAB_KEYWORDS):
        return 'labs'
    if lower.startswith('age_') or lower in {'hours_since_icu_admit'} or 'careunit' in lower:
        return 'static_or_context'
    return 'other'


def select_variant_columns(horizon_df: pd.DataFrame, variant_name: str) -> List[str]:
    base_columns = ['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'hour', 'prediction_time', 'split', 'sepsis3_label']
    available = [column for column in horizon_df.columns if column not in base_columns]

    selected = []
    for column in available:
        family = classify_feature_family(column)
        if variant_name == 'vitals_only' and family == 'vitals':
            selected.append(column)
        elif variant_name == 'vitals_labs' and family in {'vitals', 'labs'}:
            selected.append(column)
        elif variant_name == 'structured_full':
            selected.append(column)
    return base_columns + selected


def build_variant_dataset(horizon_df: pd.DataFrame, variant_name: str) -> pd.DataFrame:
    keep_columns = [column for column in select_variant_columns(horizon_df, variant_name) if column in horizon_df.columns]
    return horizon_df[keep_columns].copy()


def run_structured_ablation_suite(horizon_tables: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
    models = build_baseline_models(config)
    models = {name: model for name, model in models.items() if name in set(config['ablation']['baseline_models'])}

    rows = []
    artifacts: Dict[str, pd.DataFrame] = {}
    for dataset_name, horizon_df in horizon_tables.items():
        for variant_name in config['ablation']['executable_variants']:
            variant_df = build_variant_dataset(horizon_df, variant_name)
            tabular_df = make_stay_level_tabular_dataset(variant_df, aggregations=config['baselines']['tabular_aggregations'])
            artifacts[f'{dataset_name}_{variant_name}_tabular'] = tabular_df
            splits = split_tabular_dataset(tabular_df)
            train_X, train_y = splits['train']
            test_X, test_y = splits['test']

            if train_X.empty or test_X.empty:
                continue

            for model_name, model in models.items():
                test_prob, feature_cols = fit_and_predict_baseline(model, train_X, train_y, test_X)
                metrics = compute_binary_classification_metrics(test_y, test_prob)
                rows.append({
                    'dataset_name': dataset_name,
                    'variant_name': variant_name,
                    'model_name': model_name,
                    **metrics,
                    'n_features': len(feature_cols),
                    'n_examples': int(len(test_y)),
                })
                pred_df = test_X[['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID']].copy()
                pred_df['y_true'] = test_y.to_numpy()
                pred_df['y_prob'] = test_prob
                pred_df['variant_name'] = variant_name
                pred_df['dataset_name'] = dataset_name
                pred_df['model_name'] = model_name
                artifacts[f'{dataset_name}_{variant_name}_{model_name}_predictions'] = pred_df

    artifacts['ablation_results'] = pd.DataFrame(rows)
    return artifacts


def build_planned_ablation_matrix(config: dict) -> pd.DataFrame:
    description_map = {
        'vitals_only': 'Structured vitals subset only',
        'vitals_labs': 'Vitals plus laboratory subset',
        'structured_full': 'All structured EHR features',
        'text_only': 'Clinical notes without structured features',
        'multimodal_fusion': 'Structured plus text with multiple fusion strategies',
    }
    rows = []
    for variant in config['ablation']['planned_variants']:
        rows.append({
            'variant_name': variant,
            'description': description_map.get(variant, variant),
            'implemented_now': variant in set(config['ablation']['executable_variants']),
        })
    return pd.DataFrame(rows)


def build_fusion_strategy_table(experiment_plan_df: pd.DataFrame) -> pd.DataFrame:
    if experiment_plan_df.empty:
        return pd.DataFrame(columns=['fusion_strategy', 'structured_encoder', 'dry_run_mean_probability'])
    keep_columns = [column for column in ['fusion_strategy', 'structured_encoder', 'dry_run_mean_probability', 'dataset_name'] if column in experiment_plan_df.columns]
    return experiment_plan_df[keep_columns].copy().sort_values(['dataset_name', 'fusion_strategy']).reset_index(drop=True)
