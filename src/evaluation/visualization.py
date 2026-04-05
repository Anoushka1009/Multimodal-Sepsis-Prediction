from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


CUSTOM_MULTIMODAL_MODELS = (
    "early_fusion",
    "late_fusion",
    "gated_fusion",
    "cross_modal_attention",
)


def _resolve_prediction_threshold(predictions: pd.DataFrame, threshold: float | None) -> float:
    if threshold is not None:
        return float(threshold)
    if "decision_threshold" in predictions.columns:
        values = pd.to_numeric(predictions["decision_threshold"], errors="coerce").dropna()
        if not values.empty:
            return float(values.iloc[0])
    return 0.5


def _available_models(
    evaluation_df: pd.DataFrame,
    dataset_name: str,
    model_names: Sequence[str],
) -> list[str]:
    available = []
    for model_name in model_names:
        mask = (evaluation_df["dataset_name"] == dataset_name) & (evaluation_df["model_name"] == model_name)
        if mask.any():
            available.append(model_name)
    return available


def _load_curve_table(
    evaluation_dir: str | Path,
    dataset_name: str,
    model_name: str,
    suffix: str,
) -> pd.DataFrame | None:
    path = Path(evaluation_dir) / f"{dataset_name}_{model_name}_{suffix}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def plot_metric_bars(
    evaluation_df: pd.DataFrame,
    dataset_name: str,
    metrics: Iterable[str] = ("auroc", "auprc", "f1"),
    model_names: Sequence[str] = CUSTOM_MULTIMODAL_MODELS,
) -> tuple[plt.Figure, plt.Axes]:
    metric_list = list(metrics)
    rows = evaluation_df.loc[
        evaluation_df["dataset_name"].astype(str) == str(dataset_name),
        ["model_name", *metric_list],
    ].copy()
    rows = rows.loc[rows["model_name"].isin(model_names)].copy()
    rows = rows.set_index("model_name").reindex(_available_models(evaluation_df, dataset_name, model_names))

    fig, ax = plt.subplots(figsize=(9, 5))
    rows.plot(kind="bar", ax=ax, rot=20)
    ax.set_title(f"Custom Multimodal Metrics: {dataset_name}")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    return fig, ax


def plot_roc_curves(
    evaluation_df: pd.DataFrame,
    evaluation_dir: str | Path,
    dataset_name: str,
    model_names: Sequence[str] = CUSTOM_MULTIMODAL_MODELS,
) -> tuple[plt.Figure, plt.Axes]:
    models = _available_models(evaluation_df, dataset_name, model_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in models:
        curve = _load_curve_table(evaluation_dir, dataset_name, model_name, "roc_curve")
        if curve is None or curve.empty:
            continue
        auc = evaluation_df.loc[
            (evaluation_df["dataset_name"] == dataset_name) & (evaluation_df["model_name"] == model_name),
            "auroc",
        ].iloc[0]
        ax.plot(curve["fpr"], curve["tpr"], label=f"{model_name} (AUROC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title(f"ROC Curves: {dataset_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_pr_curves(
    evaluation_df: pd.DataFrame,
    evaluation_dir: str | Path,
    dataset_name: str,
    model_names: Sequence[str] = CUSTOM_MULTIMODAL_MODELS,
) -> tuple[plt.Figure, plt.Axes]:
    models = _available_models(evaluation_df, dataset_name, model_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name in models:
        curve = _load_curve_table(evaluation_dir, dataset_name, model_name, "pr_curve")
        if curve is None or curve.empty:
            continue
        auprc = evaluation_df.loc[
            (evaluation_df["dataset_name"] == dataset_name) & (evaluation_df["model_name"] == model_name),
            "auprc",
        ].iloc[0]
        ax.plot(curve["recall"], curve["precision"], label=f"{model_name} (AUPRC={auprc:.3f})")

    ax.set_title(f"Precision-Recall Curves: {dataset_name}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_calibration_curves(
    evaluation_dir: str | Path,
    dataset_name: str,
    model_names: Sequence[str] = CUSTOM_MULTIMODAL_MODELS,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for model_name in model_names:
        curve = _load_curve_table(evaluation_dir, dataset_name, model_name, "calibration")
        if curve is None or curve.empty:
            continue
        plotted = True
        ax.plot(
            curve["mean_predicted_probability"],
            curve["fraction_positive"],
            marker="o",
            label=model_name,
        )

    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_title(f"Calibration Curves: {dataset_name}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction Positive")
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_confusion_matrices(
    prediction_dir: str | Path,
    dataset_name: str,
    threshold: float | None = None,
    model_names: Sequence[str] = CUSTOM_MULTIMODAL_MODELS,
) -> tuple[plt.Figure, np.ndarray]:
    paths = [
        (model_name, Path(prediction_dir) / f"{dataset_name}_{model_name}_test_predictions.csv")
        for model_name in model_names
    ]
    available = [(model_name, path) for model_name, path in paths if path.exists()]
    if not available:
        raise FileNotFoundError(f"No custom-model prediction files found in {prediction_dir} for {dataset_name}.")

    ncols = 2
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 4 * nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for axis, (model_name, path) in zip(axes.flat, available):
        predictions = pd.read_csv(path)
        y_true = predictions["y_true"].astype(int)
        resolved_threshold = _resolve_prediction_threshold(predictions, threshold)
        y_pred = (predictions["y_prob"].astype(float) >= resolved_threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(model_name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(["0", "1"])
        axis.set_yticklabels(["0", "1"])

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")

        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes.flat[len(available) :]:
        axis.axis("off")

    fig.suptitle(f"Confusion Matrices: {dataset_name}", y=1.02)
    fig.tight_layout()
    return fig, axes
