from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training import prepare_multimodal_dataset, train_multimodal_models
from src.utils.io_utils import save_dataframe_bundle
from src.utils.logging_utils import write_run_manifest
from src.utils.runtime import load_project_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multimodal sepsis model locally.")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon in hours. Defaults to the first configured horizon.")
    parser.add_argument("--device", default=None, help="Device override: auto, cuda, or cpu.")
    parser.add_argument("--config-override", default=None, help="Optional YAML override file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_project_runtime(mount_colab_drive=False, override_path=args.config_override)
    config = runtime.config
    paths = runtime.paths
    local_zip_path = paths["project_root"] / config["dataset"].get("default_local_zip_path", "mimic.zip")

    if args.device:
        config["multimodal"]["device"] = args.device

    horizon = int(args.horizon or config["prediction"]["horizons_hours"][0])
    dataset_name = f"horizon_{horizon}h"

    structured_path = paths["processed_data_dir"] / "04_feature_engineering" / f"{dataset_name}.csv"
    text_path = paths["processed_data_dir"] / "05_text_processing" / f"{dataset_name}_note_windows.csv"
    if not structured_path.exists():
        raise FileNotFoundError(
            f"Missing structured dataset: {structured_path}. "
            f"Run notebooks 01, 03, 04, and 05 first. "
            f"The local dataset zip is configured as: {local_zip_path}"
        )
    if not text_path.exists():
        raise FileNotFoundError(
            f"Missing text dataset: {text_path}. "
            f"Run notebooks 01, 03, 04, and 05 first. "
            f"The local dataset zip is configured as: {local_zip_path}"
        )

    structured_df = pd.read_csv(structured_path, parse_dates=["hour", "prediction_time", "INTIME", "OUTTIME"])
    text_df = pd.read_csv(text_path, parse_dates=["prediction_time", "first_note_time", "last_note_time"])

    prepared = prepare_multimodal_dataset(
        structured_df=structured_df,
        text_df=text_df,
        config=config,
    )
    training_output = train_multimodal_models(
        prepared=prepared,
        config=config,
        output_dir=paths["processed_data_dir"] / "07_multimodal_models",
        dataset_name=dataset_name,
    )

    saved_paths = save_dataframe_bundle(
        training_output["artifacts"],
        paths["processed_data_dir"] / "07_multimodal_models",
    )

    manifest_path = paths["manifests_dir"] / f"07_multimodal_models_{dataset_name}_manifest.json"
    write_run_manifest(
        path=manifest_path,
        stage="07_multimodal_models",
        config=config,
        extra={
            "dataset_name": dataset_name,
            "saved_artifacts": saved_paths,
            "checkpoint_paths": training_output["checkpoint_paths"],
            "device": training_output["device"],
            "text_embedding_backend": training_output["text_embedding_backend"],
        },
    )

    print(f"Configured local dataset zip: {local_zip_path}")
    print(f"Saved multimodal artifacts to: {paths['processed_data_dir'] / '07_multimodal_models'}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
