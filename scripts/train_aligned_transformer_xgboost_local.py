from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.aligned_transformer_xgboost import train_aligned_transformer_xgboost
from src.utils.io_utils import save_dataframe_bundle
from src.utils.logging_utils import write_run_manifest
from src.utils.runtime import load_project_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the aligned Transformer + BERT + XGBoost hybrid model locally.")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon in hours. Defaults to the first configured horizon.")
    parser.add_argument("--device", default=None, help="Device override for the alignment encoder: auto, cuda, or cpu.")
    parser.add_argument("--config-override", default=None, help="Optional YAML override file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_project_runtime(mount_colab_drive=False, override_path=args.config_override)
    config = runtime.config
    paths = runtime.paths

    horizon = int(args.horizon or config["prediction"]["horizons_hours"][0])
    dataset_name = f"horizon_{horizon}h"

    structured_path = paths["processed_data_dir"] / "04_feature_engineering" / f"{dataset_name}.csv"
    text_path = paths["processed_data_dir"] / "05_text_processing" / f"{dataset_name}_note_windows.csv"
    if not structured_path.exists():
        raise FileNotFoundError(f"Missing structured dataset: {structured_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"Missing text dataset: {text_path}")

    structured_df = pd.read_csv(structured_path, parse_dates=["hour", "prediction_time", "INTIME", "OUTTIME"])
    text_df = pd.read_csv(text_path, parse_dates=["prediction_time", "first_note_time", "last_note_time"])

    output_dir = paths["processed_data_dir"] / config.get("aligned_transformer_xgboost", {}).get(
        "output_stage",
        "10_aligned_transformer_xgboost",
    )
    training_output = train_aligned_transformer_xgboost(
        structured_df=structured_df,
        text_df=text_df,
        config=config,
        extracted_dir=paths["extracted_data_dir"],
        output_dir=output_dir,
        dataset_name=dataset_name,
        device=args.device,
    )

    saved_paths = save_dataframe_bundle(training_output["artifacts"], output_dir)
    manifest_path = paths["manifests_dir"] / f"10_aligned_transformer_xgboost_{dataset_name}_manifest.json"
    write_run_manifest(
        path=manifest_path,
        stage="10_aligned_transformer_xgboost",
        config=config,
        extra={
            "dataset_name": dataset_name,
            "saved_artifacts": saved_paths,
            "device": training_output["device"],
            "text_embedding_backend": training_output["text_embedding_backend"],
            "checkpoint_path": training_output["checkpoint_path"],
            "xgboost_model_path": training_output["xgboost_model_path"],
        },
    )

    print(f"Saved aligned Transformer + XGBoost artifacts to: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
