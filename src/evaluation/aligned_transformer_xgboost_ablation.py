from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from src.evaluation.ablation import build_variant_dataset
from src.training.aligned_transformer_xgboost import train_aligned_transformer_xgboost


VARIANT_DESCRIPTION_MAP = {
    "vitals_only": "Structured vitals subset only",
    "vitals_labs": "Vitals plus laboratory subset",
    "structured_full": "All structured EHR features",
}


def build_aligned_transformer_xgboost_ablation_plan(config: dict) -> pd.DataFrame:
    cfg = config.get("aligned_transformer_xgboost_ablation", {})
    planned_variants = list(cfg.get("planned_variants", cfg.get("executable_variants", [])))
    executable_variants = set(cfg.get("executable_variants", []))
    return pd.DataFrame(
        [
            {
                "variant_name": variant_name,
                "description": VARIANT_DESCRIPTION_MAP.get(variant_name, variant_name),
                "implemented_now": variant_name in executable_variants,
            }
            for variant_name in planned_variants
        ]
    )


def run_aligned_transformer_xgboost_ablation_suite(
    *,
    horizon_tables: Dict[str, pd.DataFrame],
    config: dict,
    processed_dir: str | Path,
    extracted_dir: str | Path,
    output_dir: str | Path,
    device=None,
) -> Dict[str, pd.DataFrame]:
    cfg = config.get("aligned_transformer_xgboost_ablation", {})
    executable_variants = list(cfg.get("executable_variants", []))
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    artifacts: Dict[str, pd.DataFrame] = {}
    hybrid_rows: list[pd.DataFrame] = []
    encoder_rows: list[pd.DataFrame] = []

    for dataset_name, horizon_df in horizon_tables.items():
        text_path = processed_dir / "05_text_processing" / f"{dataset_name}_note_windows.csv"
        if not text_path.exists():
            continue
        text_df = pd.read_csv(
            text_path,
            parse_dates=["prediction_time", "first_note_time", "last_note_time"],
            low_memory=False,
        )

        for variant_name in executable_variants:
            variant_df = build_variant_dataset(horizon_df, variant_name)
            dataset_tag = f"{dataset_name}_{variant_name}"

            output = train_aligned_transformer_xgboost(
                structured_df=variant_df,
                text_df=text_df,
                config=config,
                extracted_dir=extracted_dir,
                output_dir=output_dir,
                dataset_name=dataset_tag,
                device=device,
            )

            for artifact_name, artifact_df in output["artifacts"].items():
                artifacts[artifact_name] = artifact_df

            hybrid_key = f"{dataset_tag}_aligned_transformer_xgboost_results"
            hybrid_df = output["artifacts"].get(hybrid_key, pd.DataFrame()).copy()
            if not hybrid_df.empty:
                hybrid_df["dataset_name"] = dataset_name
                hybrid_df["variant_name"] = variant_name
                hybrid_rows.append(hybrid_df)

            encoder_key = f"{dataset_tag}_aligned_transformer_encoder_results"
            encoder_df = output["artifacts"].get(encoder_key, pd.DataFrame()).copy()
            if not encoder_df.empty:
                encoder_df["dataset_name"] = dataset_name
                encoder_df["variant_name"] = variant_name
                encoder_rows.append(encoder_df)

    artifacts["aligned_transformer_xgboost_ablation_results"] = (
        pd.concat(hybrid_rows, ignore_index=True).sort_values(["dataset_name", "variant_name", "split"]).reset_index(drop=True)
        if hybrid_rows
        else pd.DataFrame()
    )
    artifacts["aligned_transformer_encoder_ablation_results"] = (
        pd.concat(encoder_rows, ignore_index=True).sort_values(["dataset_name", "variant_name", "split"]).reset_index(drop=True)
        if encoder_rows
        else pd.DataFrame()
    )
    artifacts["aligned_transformer_xgboost_ablation_plan"] = build_aligned_transformer_xgboost_ablation_plan(config)
    return artifacts
