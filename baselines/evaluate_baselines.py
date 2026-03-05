"""Evaluate trained baselines and generate comparison plots / tables.

Usage:
    python -m baselines.evaluate_baselines --config configs/baseline_configs.yaml
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from torch.utils.data import DataLoader

from baselines.models import (
    CrossAttnOnlyModel,
    ECGOnlyResNet,
    EarlyFusionModel,
    LateFusionModel,
    TabularOnlyMLP,
    TextOnlyBERT,
    XGBoostBaseline,
)
from baselines.train_baselines import MODEL_REGISTRY, _build_pytorch_model
from src.evaluate import evaluate_model

logger = logging.getLogger(__name__)

# ── Friendly display names ──────────────────────────────────────────────────

DISPLAY_NAMES: dict[str, str] = {
    "B1_tabular_mlp": "B1 Tabular MLP",
    "B2_text_bert": "B2 Text BERT",
    "B3_ecg_resnet": "B3 ECG ResNet",
    "B4_early_fusion": "B4 Early Fusion",
    "B5_late_fusion": "B5 Late Fusion",
    "B6_crossattn_only": "B6 Cross-Attn",
    "B7_xgboost": "B7 XGBoost",
    "Proposed": "Proposed",
    "Optimized": "Optimized",
}


# ── Single-model evaluation ─────────────────────────────────────────────────


def evaluate_single_baseline(
    model_name: str,
    checkpoint_path: str,
    test_loader: DataLoader,
    device: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Load a baseline checkpoint and compute test-set metrics.

    Parameters
    ----------
    model_name:
        Registry key (e.g. ``"B3_ecg_resnet"``).
    checkpoint_path:
        Path to ``.pt`` (PyTorch) or ``.pkl`` (XGBoost) file.
    test_loader:
        DataLoader for the test split.
    device:
        ``"cpu"`` or ``"cuda"``.
    config:
        Model-specific hyper-parameters (used for model reconstruction).

    Returns
    -------
    dict
        ``{"accuracy": …, "f1_macro": …, "auc_roc": …, …}``
    """
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if model_name == "B7_xgboost":
        with open(checkpoint_path, "rb") as f:
            xgb_model = pickle.load(f)  # noqa: S301

        features_list, labels_list = [], []
        for batch in test_loader:
            parts: list[np.ndarray] = []
            if "tabular" in batch:
                parts.append(batch["tabular"].numpy())
            if "tfidf" in batch:
                parts.append(batch["tfidf"].numpy())
            if "ecg_features" in batch:
                parts.append(batch["ecg_features"].numpy())
            if parts:
                features_list.append(np.concatenate(parts, axis=1))
            labels_list.append(batch["label"].numpy())

        X_test = np.concatenate(features_list)
        y_test = np.concatenate(labels_list)

        preds = xgb_model.predict(X_test)
        probas = xgb_model.predict_proba(X_test)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro")),
            "auc_roc": float(
                roc_auc_score(y_test, probas, multi_class="ovr", average="macro")
            ),
        }
        return metrics

    # PyTorch models
    model = _build_pytorch_model(model_name, config)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    result = evaluate_model(model, test_loader, device)

    return {
        "accuracy": result.get("accuracy", 0.0),
        "f1_macro": result.get("f1_macro", 0.0),
        "auc_roc": result.get("auc_roc", 0.0),
    }


# ── Evaluate all baselines ──────────────────────────────────────────────────


def evaluate_all_baselines(config_path: str) -> pd.DataFrame:
    """Evaluate every baseline checkpoint and return a comparison DataFrame.

    Parameters
    ----------
    config_path:
        Path to ``baseline_configs.yaml``.

    Returns
    -------
    pd.DataFrame
        Columns: ``model``, ``accuracy``, ``f1_macro``, ``auc_roc``.
    """
    with open(config_path, "r") as f:
        all_configs: dict[str, Any] = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = Path(all_configs.get("output_dir", "outputs/baselines"))

    # Build a test loader (same preprocessing as training)
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    from src.dataset import MultiModalTriageDataset
    from src.preprocess import ECGProcessor, TextCleaner, TriageDataCleaner, filter_and_align

    shared = all_configs.get("shared", {})
    csv_path = all_configs.get("csv_path", "data/triage.csv")
    ecg_path = all_configs.get("ecg_path", "data/ecg.npy")

    df, ecg_array = filter_and_align(csv_path, ecg_path, target_classes=[0, 1, 2])
    _, test_df, _, test_ecg = train_test_split(
        df,
        ecg_array,
        test_size=0.2,
        random_state=shared.get("seed", 42),
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    test_ds = MultiModalTriageDataset(
        test_df, test_ecg, tokenizer, TextCleaner(), ECGProcessor(), TriageDataCleaner()
    )
    test_loader = DataLoader(test_ds, batch_size=shared.get("batch_size", 32), shuffle=False)

    rows: list[dict[str, Any]] = []
    for model_name in MODEL_REGISTRY:
        ext = ".pkl" if model_name == "B7_xgboost" else ".pt"
        ckpt = checkpoint_dir / f"{model_name}{ext}"
        if not ckpt.exists():
            logger.warning("Checkpoint not found for %s – skipping.", model_name)
            continue
        model_cfg = all_configs.get(model_name, {})
        metrics = evaluate_single_baseline(
            model_name, str(ckpt), test_loader, device, model_cfg
        )
        rows.append({"model": model_name, **metrics})
        logger.info("%s → %s", model_name, metrics)

    return pd.DataFrame(rows)


# ── Plotting utilities ───────────────────────────────────────────────────────


def plot_baseline_comparison(results_df: pd.DataFrame, save_path: str) -> None:
    """Grouped bar chart comparing F1-Macro, Accuracy, and AUC-ROC.

    Baselines are rendered in gray, *Proposed* in blue, *Optimized* in green.
    Exact values are annotated on top of each bar.
    """
    metric_cols = ["f1_macro", "accuracy", "auc_roc"]
    display_labels = {"f1_macro": "F1-Macro", "accuracy": "Accuracy", "auc_roc": "AUC-ROC"}
    models = results_df["model"].tolist()
    display = [DISPLAY_NAMES.get(m, m) for m in models]

    x = np.arange(len(models))
    n_metrics = len(metric_cols)
    width = 0.22

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.6), 6))

    for i, col in enumerate(metric_cols):
        values = results_df[col].values
        colors = []
        for m in models:
            if m == "Proposed":
                colors.append("#4A90D9")
            elif m == "Optimized":
                colors.append("#50C878")
            else:
                gray_level = 0.35 + 0.06 * models.index(m)
                colors.append(str(min(gray_level, 0.75)))
        bars = ax.bar(x + i * width, values, width, label=display_labels[col], color=colors)
        for bar, val in zip(bars, values):
            ax.annotate(
                f"{val:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(display, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric Value")
    ax.set_title("Baseline Comparison")
    ax.legend()
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info("Saved comparison plot → %s", save_path)


def plot_modality_contribution(results_df: pd.DataFrame, save_path: str) -> None:
    """Stacked bar chart showing metric contribution by modality group.

    Groups: Tabular Only (B1), Text Only (B2), ECG Only (B3), All Modalities
    (best of B4–B6).
    """
    groups = {
        "Tabular Only": "B1_tabular_mlp",
        "Text Only": "B2_text_bert",
        "ECG Only": "B3_ecg_resnet",
    }
    metric_cols = ["f1_macro", "accuracy", "auc_roc"]
    display_labels = {"f1_macro": "F1-Macro", "accuracy": "Accuracy", "auc_roc": "AUC-ROC"}

    rows: list[dict[str, Any]] = []
    for label, key in groups.items():
        row = results_df.loc[results_df["model"] == key]
        if row.empty:
            continue
        rows.append({"group": label, **{c: row.iloc[0][c] for c in metric_cols}})

    # "All" = best fusion baseline (B4–B6) by f1_macro
    fusion_names = {"B4_early_fusion", "B5_late_fusion", "B6_crossattn_only"}
    fusion_rows = results_df[results_df["model"].isin(fusion_names)]
    if not fusion_rows.empty:
        best = fusion_rows.sort_values("f1_macro", ascending=False).iloc[0]
        rows.append({"group": "All Modalities", **{c: best[c] for c in metric_cols}})

    plot_df = pd.DataFrame(rows).set_index("group")
    plot_df.rename(columns=display_labels, inplace=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df.plot.bar(ax=ax)
    ax.set_ylabel("Metric Value")
    ax.set_ylim(0, 1.05)
    ax.set_title("Modality Contribution")
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info("Saved modality contribution plot → %s", save_path)


def generate_comparison_table(results_df: pd.DataFrame, save_path: str) -> None:
    """Persist results DataFrame as CSV.

    Parameters
    ----------
    results_df:
        DataFrame with columns ``model``, ``accuracy``, ``f1_macro``, ``auc_roc``.
    save_path:
        Destination ``.csv`` path.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_path, index=False)
    logger.info("Saved comparison table → %s", save_path)


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for evaluating all baselines."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Evaluate baseline models (B1–B7)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_configs.yaml",
        help="Path to baseline_configs.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/baselines",
        help="Directory for plots and tables",
    )
    args = parser.parse_args()

    results_df = evaluate_all_baselines(args.config)
    if results_df.empty:
        logger.error("No baselines evaluated – aborting plots.")
        return

    out = Path(args.output_dir)
    plot_baseline_comparison(results_df, str(out / "comparison_bar.png"))
    plot_modality_contribution(results_df, str(out / "modality_contribution.png"))
    generate_comparison_table(results_df, str(out / "comparison_table.csv"))


if __name__ == "__main__":
    main()
