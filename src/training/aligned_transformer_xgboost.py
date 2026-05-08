from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import compute_binary_classification_metrics
from src.models.aligned_transformer_xgboost import AlignedTransformerXGBoostEncoder
from src.models.baselines import ID_COLUMNS, build_baseline_models
from src.training.multimodal import (
    PreparedMultimodalDataset,
    prepare_multimodal_dataset,
    resolve_device,
    set_random_seed,
)
from src.training.tabular_multimodal import (
    FEATURE_METADATA_COLUMNS,
    build_clinical_event_feature_table,
    build_note_feature_table,
    build_structured_augmented_tabular_dataset,
)


def _split_indices(stay_index: pd.DataFrame, split_name: str) -> np.ndarray:
    return np.flatnonzero(stay_index["split"].astype(str).to_numpy() == split_name)


def _build_loader(
    prepared: PreparedMultimodalDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(prepared.structured_sequences[indices], dtype=torch.float32),
        torch.tensor(prepared.text_embeddings[indices], dtype=torch.float32),
        torch.tensor(prepared.text_token_ids[indices], dtype=torch.long),
        torch.tensor(prepared.text_token_attention_masks[indices], dtype=torch.long),
        torch.tensor(prepared.structured_summary_features[indices], dtype=torch.float32),
        torch.tensor(prepared.structured_padding_masks[indices], dtype=torch.bool),
        torch.tensor(prepared.text_padding_masks[indices], dtype=torch.bool),
        torch.tensor(prepared.labels[indices], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _select_decision_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    metric_name: str = "f1",
    default_threshold: float = 0.5,
) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return float(default_threshold)

    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(0.05, 0.95, 19, dtype=np.float32),
                y_prob.astype(np.float32),
                np.asarray([default_threshold], dtype=np.float32),
            ]
        )
    )

    best_threshold = float(default_threshold)
    best_score = float("-inf")
    for threshold in candidate_thresholds:
        metrics = compute_binary_classification_metrics(y_true, y_prob, threshold=float(threshold))
        score = float(metrics.get(metric_name, float("nan")))
        if np.isnan(score):
            continue
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _build_prediction_frame(
    ids: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    dataset_name: str,
    model_name: str,
    decision_threshold: float,
) -> pd.DataFrame:
    frame = ids.reset_index(drop=True).copy()
    frame["y_true"] = np.asarray(y_true).astype(int)
    frame["y_prob"] = np.asarray(y_prob).astype(float)
    frame["y_pred"] = (frame["y_prob"] >= float(decision_threshold)).astype(int)
    frame["decision_threshold"] = float(decision_threshold)
    frame["dataset_name"] = dataset_name
    frame["model_name"] = model_name
    return frame


def _predict_alignment_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs = []
    labels = []
    losses = []

    with torch.no_grad():
        for structured_batch, text_batch, token_ids_batch, token_mask_batch, summary_batch, structured_mask_batch, text_mask_batch, label_batch in loader:
            structured_batch = structured_batch.to(device)
            text_batch = text_batch.to(device)
            token_ids_batch = token_ids_batch.to(device)
            token_mask_batch = token_mask_batch.to(device)
            summary_batch = summary_batch.to(device)
            structured_mask_batch = structured_mask_batch.to(device)
            text_mask_batch = text_mask_batch.to(device)
            label_batch = label_batch.to(device)

            logits, _ = model(
                structured_batch,
                text_window_embeddings=text_batch,
                text_input_ids=token_ids_batch,
                text_token_attention_mask=token_mask_batch,
                structured_summary_features=summary_batch,
                structured_padding_mask=structured_mask_batch,
                text_padding_mask=text_mask_batch,
            )
            if criterion is not None:
                losses.append(float(criterion(logits, label_batch).detach().cpu()))
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            labels.append(label_batch.detach().cpu().numpy())

    if not probs:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32), float("nan")
    return (
        np.concatenate(probs).astype(np.float32),
        np.concatenate(labels).astype(np.float32),
        float(np.mean(losses)) if losses else float("nan"),
    )


def _extract_aligned_embeddings(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    rows = []

    with torch.no_grad():
        for structured_batch, text_batch, token_ids_batch, token_mask_batch, summary_batch, structured_mask_batch, text_mask_batch, _ in loader:
            structured_batch = structured_batch.to(device)
            text_batch = text_batch.to(device)
            token_ids_batch = token_ids_batch.to(device)
            token_mask_batch = token_mask_batch.to(device)
            summary_batch = summary_batch.to(device)
            structured_mask_batch = structured_mask_batch.to(device)
            text_mask_batch = text_mask_batch.to(device)

            _, aux = model(
                structured_batch,
                text_window_embeddings=text_batch,
                text_input_ids=token_ids_batch,
                text_token_attention_mask=token_mask_batch,
                structured_summary_features=summary_batch,
                structured_padding_mask=structured_mask_batch,
                text_padding_mask=text_mask_batch,
            )
            rows.append(aux["aligned_repr"].detach().cpu().numpy().astype(np.float32))

    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.concatenate(rows, axis=0).astype(np.float32)


def _write_live_artifacts(
    *,
    output_dir: Path,
    dataset_name: str,
    history_rows: list[dict],
    progress_payload: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{dataset_name}_aligned_transformer_xgboost_training_history_live.csv"
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    progress_path = output_dir / f"{dataset_name}_aligned_transformer_xgboost_progress.json"
    progress_path.write_text(json.dumps(progress_payload, indent=2, sort_keys=True), encoding="utf-8")


def _prepare_alignment_config(config: dict) -> dict:
    cfg = copy.deepcopy(config)
    hybrid_cfg = cfg.get("aligned_transformer_xgboost", {})
    cfg.setdefault("multimodal", {})
    cfg["multimodal"]["structured_encoder"] = str(hybrid_cfg.get("structured_encoder", "transformer"))
    cfg["multimodal"]["text_encoder_mode"] = str(hybrid_cfg.get("text_encoder_mode", cfg["multimodal"].get("text_encoder_mode", "frozen_embedding")))
    cfg["multimodal"]["max_structured_steps"] = int(hybrid_cfg.get("max_structured_steps", cfg["multimodal"].get("max_structured_steps", 48)))
    cfg["multimodal"]["max_text_windows"] = int(hybrid_cfg.get("max_text_windows", cfg["multimodal"].get("max_text_windows", 8)))
    cfg["multimodal"]["max_tokens_per_window"] = int(hybrid_cfg.get("max_tokens_per_window", cfg["multimodal"].get("max_tokens_per_window", 128)))
    cfg["multimodal"]["text_embedding_dim"] = int(hybrid_cfg.get("text_embedding_dim", cfg["multimodal"].get("text_embedding_dim", 768)))
    return cfg


def _build_split_feature_table(feature_table: pd.DataFrame) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
    splits: Dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for split_name in ["train", "val", "test"]:
        split_df = feature_table.loc[feature_table["split"] == split_name].copy()
        if split_df.empty:
            splits[split_name] = (pd.DataFrame(), pd.Series(dtype=int))
            continue
        y = split_df["sepsis3_label"].astype(int)
        drop_columns = [column for column in FEATURE_METADATA_COLUMNS if column in split_df.columns]
        X = split_df.drop(columns=drop_columns)
        splits[split_name] = (X, y)
    return splits


def train_aligned_transformer_xgboost(
    *,
    structured_df: pd.DataFrame,
    text_df: pd.DataFrame,
    config: dict,
    extracted_dir: str | Path,
    output_dir: str | Path,
    dataset_name: str,
    device=None,
) -> Dict[str, object]:
    hybrid_cfg = config.get("aligned_transformer_xgboost", {})
    seed = int(config["project"]["seed"])
    set_random_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device or hybrid_cfg.get("device", config["multimodal"].get("device", "auto")))
    alignment_config = _prepare_alignment_config(config)
    prepared = prepare_multimodal_dataset(
        structured_df=structured_df,
        text_df=text_df,
        config=alignment_config,
        device=device,
    )

    batch_size = int(hybrid_cfg.get("batch_size", config["multimodal"].get("batch_size", 16)))
    epochs = int(hybrid_cfg.get("epochs", 10))
    learning_rate = float(hybrid_cfg.get("learning_rate", config["multimodal"].get("learning_rate", 3e-4)))
    weight_decay = float(hybrid_cfg.get("weight_decay", config["multimodal"].get("weight_decay", 1e-4)))
    gradient_clip_norm = float(hybrid_cfg.get("gradient_clip_norm", config["multimodal"].get("gradient_clip_norm", 1.0)))
    scheduler_name = str(hybrid_cfg.get("scheduler", config["multimodal"].get("scheduler", "cosine"))).lower()
    min_learning_rate = float(hybrid_cfg.get("min_learning_rate", config["multimodal"].get("min_learning_rate", learning_rate * 0.1)))
    early_stopping_patience = int(hybrid_cfg.get("early_stopping_patience", config["multimodal"].get("early_stopping_patience", 5)))
    threshold_selection_metric = str(hybrid_cfg.get("threshold_selection_metric", config["multimodal"].get("threshold_selection_metric", "f1")))
    hidden_dim = int(hybrid_cfg.get("hidden_dim", config["multimodal"].get("hidden_dim", 128)))
    aligned_dim = int(hybrid_cfg.get("aligned_dim", 256))
    dropout = float(hybrid_cfg.get("dropout", config["multimodal"].get("dropout", 0.2)))

    train_indices = _split_indices(prepared.stay_index, "train")
    val_indices = _split_indices(prepared.stay_index, "val")
    test_indices = _split_indices(prepared.stay_index, "test")
    if len(train_indices) == 0 or len(test_indices) == 0:
        raise ValueError("Aligned transformer + XGBoost training requires non-empty train and test splits.")

    selection_indices = val_indices if len(val_indices) > 0 else test_indices
    train_loader = _build_loader(prepared, train_indices, batch_size=batch_size, shuffle=True)
    selection_loader = _build_loader(prepared, selection_indices, batch_size=batch_size, shuffle=False)
    test_loader = _build_loader(prepared, test_indices, batch_size=batch_size, shuffle=False)

    positive_count = float(prepared.labels[train_indices].sum())
    negative_count = float(len(train_indices) - positive_count)
    pos_weight_value = negative_count / positive_count if positive_count > 0 and negative_count > 0 else 1.0
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    )

    encoder_model = AlignedTransformerXGBoostEncoder(
        structured_input_dim=len(prepared.feature_columns),
        text_embedding_dim=prepared.text_embedding_dim,
        structured_summary_dim=int(prepared.structured_summary_features.shape[1]),
        hidden_dim=hidden_dim,
        aligned_dim=aligned_dim,
        dropout=dropout,
        text_encoder_mode=prepared.text_input_mode,
        text_model_name=config["text_processing"].get("pretrained_text_model_name"),
        text_local_files_only=bool(config["text_processing"].get("local_files_only", False)),
        text_finetune_unfrozen_layers=int(hybrid_cfg.get("text_finetune_unfrozen_layers", config["multimodal"].get("text_finetune_unfrozen_layers", 2))),
        text_gradient_checkpointing=bool(hybrid_cfg.get("text_gradient_checkpointing", config["multimodal"].get("text_gradient_checkpointing", False))),
        structured_num_heads=int(hybrid_cfg.get("structured_num_heads", config["multimodal"].get("structured_num_heads", 4))),
        structured_num_layers=int(hybrid_cfg.get("structured_num_layers", config["multimodal"].get("structured_num_layers", 2))),
        text_num_heads=int(hybrid_cfg.get("text_num_heads", config["multimodal"].get("text_num_heads", 4))),
        text_num_layers=int(hybrid_cfg.get("text_num_layers", config["multimodal"].get("text_num_layers", 1))),
        fusion_num_heads=int(hybrid_cfg.get("fusion_num_heads", config["multimodal"].get("fusion_num_heads", 4))),
    ).to(device)

    text_finetune_learning_rate = float(hybrid_cfg.get("text_finetune_learning_rate", config["multimodal"].get("text_finetune_learning_rate", 2e-5)))
    text_window_embedder = getattr(encoder_model, "text_window_embedder", None)
    if text_window_embedder is not None and getattr(text_window_embedder, "is_finetuning_enabled", False):
        text_params = [parameter for parameter in text_window_embedder.parameters() if parameter.requires_grad]
        text_param_ids = {id(parameter) for parameter in text_params}
        other_params = [
            parameter
            for parameter in encoder_model.parameters()
            if parameter.requires_grad and id(parameter) not in text_param_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": learning_rate},
                {"params": text_params, "lr": text_finetune_learning_rate},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(encoder_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = None
    if scheduler_name == "cosine" and epochs > 1:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=min_learning_rate,
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=min_learning_rate,
        )

    history_rows: list[dict] = []
    best_metric = float("-inf")
    best_threshold = float(config.get("evaluation", {}).get("default_threshold", 0.5))
    best_state = copy.deepcopy(encoder_model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        encoder_model.train()
        batch_losses = []
        for structured_batch, text_batch, token_ids_batch, token_mask_batch, summary_batch, structured_mask_batch, text_mask_batch, label_batch in train_loader:
            structured_batch = structured_batch.to(device)
            text_batch = text_batch.to(device)
            token_ids_batch = token_ids_batch.to(device)
            token_mask_batch = token_mask_batch.to(device)
            summary_batch = summary_batch.to(device)
            structured_mask_batch = structured_mask_batch.to(device)
            text_mask_batch = text_mask_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = encoder_model(
                structured_batch,
                text_window_embeddings=text_batch,
                text_input_ids=token_ids_batch,
                text_token_attention_mask=token_mask_batch,
                structured_summary_features=summary_batch,
                structured_padding_mask=structured_mask_batch,
                text_padding_mask=text_mask_batch,
            )
            loss = criterion(logits, label_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), gradient_clip_norm)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        selection_prob, selection_true, selection_loss = _predict_alignment_model(
            encoder_model,
            selection_loader,
            device=device,
            criterion=criterion,
        )
        selection_metrics = (
            compute_binary_classification_metrics(selection_true, selection_prob)
            if len(selection_true)
            else {}
        )
        selection_score = float(selection_metrics.get("auprc", float("nan")))
        if np.isnan(selection_score):
            selection_score = float(selection_metrics.get("auroc", float("nan")))
        if np.isnan(selection_score):
            selection_score = -float(selection_loss) if not np.isnan(selection_loss) else float("-inf")

        if selection_score > best_metric:
            best_metric = selection_score
            best_state = copy.deepcopy(encoder_model.state_dict())
            best_threshold = _select_decision_threshold(
                selection_true,
                selection_prob,
                metric_name=threshold_selection_metric,
                default_threshold=float(config.get("evaluation", {}).get("default_threshold", 0.5)),
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(selection_score)
            else:
                scheduler.step()

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "selection_loss": selection_loss,
                "selection_auroc": float(selection_metrics.get("auroc", float("nan"))),
                "selection_auprc": float(selection_metrics.get("auprc", float("nan"))),
                "selection_precision": float(selection_metrics.get("precision", float("nan"))),
                "selection_recall": float(selection_metrics.get("recall", float("nan"))),
                "selection_f1": float(selection_metrics.get("f1", float("nan"))),
                "decision_threshold": float(best_threshold),
            }
        )

        _write_live_artifacts(
            output_dir=output_dir,
            dataset_name=dataset_name,
            history_rows=history_rows,
            progress_payload={
                "dataset_name": dataset_name,
                "status": "training_alignment_encoder",
                "current_epoch": int(epoch),
                "total_epochs": int(epochs),
                "latest_train_loss": float(np.mean(batch_losses)) if batch_losses else float("nan"),
                "latest_selection_loss": float(selection_loss) if not np.isnan(selection_loss) else None,
                "latest_selection_auroc": float(selection_metrics.get("auroc", float("nan"))),
                "latest_selection_auprc": float(selection_metrics.get("auprc", float("nan"))),
                "latest_selection_f1": float(selection_metrics.get("f1", float("nan"))),
                "best_selection_score": float(best_metric),
                "best_decision_threshold": float(best_threshold),
            },
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            break

    encoder_model.load_state_dict(best_state)
    checkpoint_path = output_dir / f"{dataset_name}_aligned_transformer_encoder_best.pt"
    torch.save(
        {
            "state_dict": encoder_model.state_dict(),
            "model_name": "aligned_transformer_encoder",
            "structured_encoder": "transformer",
            "text_input_mode": prepared.text_input_mode,
            "text_embedding_backend": prepared.text_embedding_backend,
            "aligned_dim": aligned_dim,
            "feature_columns": prepared.feature_columns,
            "structured_summary_columns": prepared.structured_summary_columns,
            "config": config,
        },
        checkpoint_path,
    )

    artifact_tables: Dict[str, pd.DataFrame] = {}
    encoder_result_rows: list[dict] = []
    split_probabilities: dict[str, np.ndarray] = {}
    split_labels: dict[str, np.ndarray] = {}
    aligned_feature_frames: list[pd.DataFrame] = []

    aligned_feature_columns = [f"aligned_embedding_{index:03d}" for index in range(aligned_dim)]

    split_loaders = {
        "train": _build_loader(prepared, train_indices, batch_size=batch_size, shuffle=False),
        "val": _build_loader(prepared, val_indices, batch_size=batch_size, shuffle=False) if len(val_indices) else None,
        "test": test_loader,
    }
    split_indices_map = {"train": train_indices, "val": val_indices, "test": test_indices}

    for split_name, loader in split_loaders.items():
        indices = split_indices_map[split_name]
        if loader is None or len(indices) == 0:
            continue

        split_prob, split_true, split_loss = _predict_alignment_model(
            encoder_model,
            loader,
            device=device,
            criterion=criterion,
        )
        split_embeddings = _extract_aligned_embeddings(encoder_model, loader, device=device)
        split_probabilities[split_name] = split_prob
        split_labels[split_name] = split_true

        split_ids = prepared.stay_index.iloc[indices][ID_COLUMNS].reset_index(drop=True)
        aligned_frame = split_ids.copy()
        aligned_frame = pd.concat(
            [
                aligned_frame,
                pd.DataFrame(split_embeddings, columns=aligned_feature_columns),
            ],
            axis=1,
        )
        aligned_feature_frames.append(aligned_frame)

        if split_name in {"val", "test"}:
            metrics = compute_binary_classification_metrics(split_true, split_prob, threshold=best_threshold)
            encoder_result_rows.append(
                {
                    "dataset_name": dataset_name,
                    "split": split_name,
                    "model_name": "aligned_transformer_encoder",
                    "structured_encoder": "transformer",
                    "text_input_mode": prepared.text_input_mode,
                    "text_embedding_backend": prepared.text_embedding_backend,
                    "aligned_dim": int(aligned_dim),
                    "n_examples": int(len(split_true)),
                    "decision_threshold": float(best_threshold),
                    "loss": split_loss,
                    **metrics,
                }
            )
            artifact_tables[f"{dataset_name}_aligned_transformer_encoder_{split_name}_predictions"] = _build_prediction_frame(
                split_ids,
                split_true,
                split_prob,
                dataset_name=dataset_name,
                model_name="aligned_transformer_encoder",
                decision_threshold=best_threshold,
            )

    alignment_feature_table = pd.concat(aligned_feature_frames, ignore_index=True) if aligned_feature_frames else pd.DataFrame(columns=ID_COLUMNS)

    structured_table = build_structured_augmented_tabular_dataset(
        structured_df,
        aggregations=hybrid_cfg.get("structured_aggregations", config.get("tabular_multimodal", {}).get("structured_aggregations", ["mean", "min", "max", "last"])),
        include_missingness=bool(hybrid_cfg.get("include_missingness", config.get("tabular_multimodal", {}).get("include_missingness", True))),
        include_static_categoricals=bool(hybrid_cfg.get("include_static_categoricals", config.get("tabular_multimodal", {}).get("include_static_categoricals", True))),
    )
    stay_index = structured_table[
        ID_COLUMNS + [column for column in ["split", "sepsis3_label", "prediction_time", "INTIME", "OUTTIME"] if column in structured_table.columns]
    ].copy()

    metadata_config = copy.deepcopy(config)
    metadata_config.setdefault("tabular_multimodal", {})
    metadata_config["tabular_multimodal"]["text_embedding_aggregations"] = []
    metadata_config["tabular_multimodal"]["include_note_metadata"] = bool(hybrid_cfg.get("include_note_metadata", True))
    note_metadata_table, _ = build_note_feature_table(
        text_df,
        stay_index,
        config=metadata_config,
        device=device,
    )

    event_config = copy.deepcopy(config)
    event_config.setdefault("tabular_multimodal", {})
    event_config["tabular_multimodal"]["include_clinical_event_features"] = bool(hybrid_cfg.get("include_clinical_event_features", True))
    event_config["tabular_multimodal"]["clinical_event_lookback_hours"] = int(
        hybrid_cfg.get(
            "clinical_event_lookback_hours",
            config.get("tabular_multimodal", {}).get(
                "clinical_event_lookback_hours",
                config["feature_engineering"].get("history_window_hours", 48),
            ),
        )
    )
    event_table = build_clinical_event_feature_table(
        stay_index,
        structured_df,
        config=event_config,
        extracted_dir=extracted_dir,
    )

    feature_table = (
        structured_table.merge(note_metadata_table, on=ID_COLUMNS, how="left")
        .merge(event_table, on=ID_COLUMNS, how="left")
        .merge(alignment_feature_table, on=ID_COLUMNS, how="inner")
    )

    splits = _build_split_feature_table(feature_table)
    train_X, train_y = splits["train"]
    val_X, val_y = splits["val"]
    test_X, test_y = splits["test"]
    if train_X.empty or test_X.empty:
        raise ValueError("Aligned transformer + XGBoost stage requires non-empty train and test splits.")

    xgboost_model_name = str(hybrid_cfg.get("xgboost_model", "xgboost"))
    models = build_baseline_models(config)
    if xgboost_model_name not in models:
        raise ValueError(f"Requested XGBoost model '{xgboost_model_name}' is not available in the configured baselines.")

    xgboost_model = clone(models[xgboost_model_name])
    feature_columns = list(train_X.columns)
    aligned_columns = [column for column in feature_columns if column.startswith("aligned_embedding_")]
    note_metadata_columns = [
        column
        for column in feature_columns
        if column.startswith("note_") or column.startswith("category_window_") or column.startswith("category_note_")
    ]
    event_feature_columns = [
        column
        for column in feature_columns
        if column.startswith("antibiotic_") or column.startswith("culture_") or column.startswith("vasopressor_")
    ]
    structured_feature_columns = [
        column
        for column in feature_columns
        if column not in set(aligned_columns + note_metadata_columns + event_feature_columns)
    ]

    xgboost_model.fit(train_X[feature_columns], train_y)
    xgboost_model_path = output_dir / f"{dataset_name}_aligned_transformer_xgboost_model.joblib"
    joblib.dump(
        {
            "model": xgboost_model,
            "feature_columns": feature_columns,
            "model_name": "aligned_transformer_xgboost",
            "dataset_name": dataset_name,
        },
        xgboost_model_path,
    )
    selection_X = val_X if not val_X.empty else test_X
    selection_y = val_y if not val_y.empty else test_y
    selection_prob = xgboost_model.predict_proba(selection_X[feature_columns])[:, 1]
    xgb_threshold = _select_decision_threshold(
        selection_y.to_numpy(),
        selection_prob,
        metric_name=threshold_selection_metric,
        default_threshold=float(config.get("evaluation", {}).get("default_threshold", 0.5)),
    )

    hybrid_result_rows: list[dict] = []
    split_ids = {
        split_name: feature_table.loc[feature_table["split"] == split_name, ID_COLUMNS].reset_index(drop=True)
        for split_name in ["val", "test"]
    }
    for split_name, eval_X, eval_y in [("val", val_X, val_y), ("test", test_X, test_y)]:
        if eval_X.empty:
            continue
        eval_prob = xgboost_model.predict_proba(eval_X[feature_columns])[:, 1]
        metrics = compute_binary_classification_metrics(eval_y, eval_prob, threshold=xgb_threshold)
        hybrid_result_rows.append(
            {
                "dataset_name": dataset_name,
                "split": split_name,
                "model_name": "aligned_transformer_xgboost",
                "structured_encoder": "transformer",
                "text_input_mode": prepared.text_input_mode,
                "text_embedding_backend": prepared.text_embedding_backend,
                "decision_threshold": float(xgb_threshold),
                "n_features": int(len(feature_columns)),
                "n_aligned_features": int(len(aligned_columns)),
                "n_structured_features": int(len(structured_feature_columns)),
                "n_note_metadata_features": int(len(note_metadata_columns)),
                "n_event_features": int(len(event_feature_columns)),
                **metrics,
            }
        )
        artifact_tables[f"{dataset_name}_aligned_transformer_xgboost_{split_name}_predictions"] = _build_prediction_frame(
            split_ids[split_name],
            eval_y.to_numpy(),
            eval_prob,
            dataset_name=dataset_name,
            model_name="aligned_transformer_xgboost",
            decision_threshold=xgb_threshold,
        )

    feature_manifest_rows = []
    for column in aligned_columns:
        feature_manifest_rows.append({"feature_name": column, "feature_group": "aligned_embedding"})
    for column in structured_feature_columns:
        feature_manifest_rows.append({"feature_name": column, "feature_group": "structured"})
    for column in note_metadata_columns:
        feature_manifest_rows.append({"feature_name": column, "feature_group": "note_metadata"})
    for column in event_feature_columns:
        feature_manifest_rows.append({"feature_name": column, "feature_group": "clinical_event"})

    artifact_tables[f"{dataset_name}_aligned_transformer_encoder_training_history"] = pd.DataFrame(history_rows)
    artifact_tables[f"{dataset_name}_aligned_transformer_encoder_results"] = pd.DataFrame(encoder_result_rows)
    artifact_tables[f"{dataset_name}_aligned_transformer_xgboost_results"] = pd.DataFrame(hybrid_result_rows)
    artifact_tables[f"{dataset_name}_aligned_transformer_xgboost_feature_manifest"] = pd.DataFrame(feature_manifest_rows)
    artifact_tables[f"{dataset_name}_aligned_transformer_xgboost_stay_index"] = feature_table[
        ID_COLUMNS + [column for column in ["split", "sepsis3_label"] if column in feature_table.columns]
    ].copy()
    artifact_tables[f"{dataset_name}_aligned_transformer_xgboost_summary"] = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "structured_encoder": "transformer",
                "text_input_mode": prepared.text_input_mode,
                "text_embedding_backend": prepared.text_embedding_backend,
                "alignment_dim": int(aligned_dim),
                "n_train": int(len(train_y)),
                "n_val": int(len(val_y)),
                "n_test": int(len(test_y)),
                "n_aligned_features": int(len(aligned_columns)),
                "n_structured_features": int(len(structured_feature_columns)),
                "n_note_metadata_features": int(len(note_metadata_columns)),
                "n_event_features": int(len(event_feature_columns)),
                "encoder_checkpoint_path": str(checkpoint_path),
                "xgboost_model_path": str(xgboost_model_path),
                "device": str(device),
            }
        ]
    )
    artifact_tables[f"{dataset_name}_aligned_transformer_xgboost_experiment_plan"] = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "model_name": "aligned_transformer_xgboost",
                "structured_encoder": "transformer",
                "text_input_mode": prepared.text_input_mode,
                "text_embedding_backend": prepared.text_embedding_backend,
                "hidden_dim": int(hidden_dim),
                "aligned_dim": int(aligned_dim),
                "epochs": int(epochs),
                "batch_size": int(batch_size),
                "learning_rate": float(learning_rate),
                "weight_decay": float(weight_decay),
                "scheduler": scheduler_name,
                "threshold_selection_metric": threshold_selection_metric,
                "xgboost_model": xgboost_model_name,
                "n_train": int(len(train_y)),
                "n_val": int(len(val_y)),
                "n_test": int(len(test_y)),
            }
        ]
    )

    _write_live_artifacts(
        output_dir=output_dir,
        dataset_name=dataset_name,
        history_rows=history_rows,
        progress_payload={
            "dataset_name": dataset_name,
            "status": "completed",
            "current_epoch": int(history_rows[-1]["epoch"]) if history_rows else 0,
            "total_epochs": int(epochs),
            "encoder_checkpoint": str(checkpoint_path),
            "xgboost_model": str(xgboost_model_path),
            "xgboost_threshold": float(xgb_threshold),
            "result_row_count": int(len(hybrid_result_rows)),
        },
    )

    return {
        "artifacts": artifact_tables,
        "checkpoint_path": str(checkpoint_path),
        "xgboost_model_path": str(xgboost_model_path),
        "device": str(device),
        "text_embedding_backend": prepared.text_embedding_backend,
    }
