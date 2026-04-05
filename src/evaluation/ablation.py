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
from src.training.tabular_multimodal import train_tabular_multimodal_models
from src.utils.paths import resolve_project_paths


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
    model_source = str(config.get('ablation', {}).get('model_source', 'baseline')).lower()
    if model_source == 'tabular_multimodal':
        return _run_tabular_multimodal_ablation_suite(horizon_tables, config)

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


def _run_tabular_multimodal_ablation_suite(horizon_tables: Dict[str, pd.DataFrame], config: dict) -> Dict[str, pd.DataFrame]:
    paths = resolve_project_paths(config)
    processed_dir = paths['processed_data_dir']
    extracted_dir = paths['extracted_data_dir']

    rows = []
    artifacts: Dict[str, pd.DataFrame] = {}
    for dataset_name, horizon_df in horizon_tables.items():
        text_path = processed_dir / '05_text_processing' / f'{dataset_name}_note_windows.csv'
        if not text_path.exists():
            continue
        text_df = pd.read_csv(
            text_path,
            parse_dates=['prediction_time', 'first_note_time', 'last_note_time'],
            low_memory=False,
        )

        for variant_name in config['ablation']['executable_variants']:
            variant_df = build_variant_dataset(horizon_df, variant_name)
            dataset_tag = f'{dataset_name}_{variant_name}'
            output = train_tabular_multimodal_models(
                structured_df=variant_df,
                text_df=text_df,
                config=config,
                extracted_dir=extracted_dir,
                dataset_name=dataset_tag,
                device=config.get('multimodal', {}).get('device', 'auto'),
            )

            for artifact_name, artifact_df in output['artifacts'].items():
                artifacts[artifact_name] = artifact_df

            result_key = f'{dataset_tag}_tabular_multimodal_results'
            result_df = output['artifacts'].get(result_key, pd.DataFrame()).copy()
            if not result_df.empty:
                result_df['dataset_name'] = dataset_name
                result_df['variant_name'] = variant_name
                rows.append(result_df)

    artifacts['ablation_results'] = (
        pd.concat(rows, ignore_index=True).sort_values(['dataset_name', 'variant_name', 'model_name', 'split']).reset_index(drop=True)
        if rows
        else pd.DataFrame()
    )
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
        return pd.DataFrame(columns=['fusion_strategy', 'structured_encoder', 'dataset_name'])
    keep_columns = [
        column
        for column in [
            'fusion_strategy',
            'structured_encoder',
            'dataset_name',
            'split',
            'auprc',
            'auroc',
            'loss',
            'dry_run_mean_probability',
            'text_embedding_backend',
        ]
        if column in experiment_plan_df.columns
    ]
    sort_columns = [column for column in ['dataset_name', 'split', 'fusion_strategy'] if column in keep_columns]
    return experiment_plan_df[keep_columns].copy().sort_values(sort_columns or keep_columns[:1]).reset_index(drop=True)
