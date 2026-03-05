"""Ablation study (A1–A7) for the proposed multimodal triage model.

Usage:
    python -m baselines.ablation_study --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.losses import FocalLoss
from src.train import EarlyStopping, train_one_epoch, validate

logger = logging.getLogger(__name__)

# ── Ablation configurations ─────────────────────────────────────────────────

ABLATION_CONFIGS: dict[str, dict[str, Any]] = {
    "A1": {"loss_type": "cross_entropy"},
    "A2": {"use_weighted_sampler": False},
    "A3": {"bert_freeze_layers": 12},
    "A4": {"bert_freeze_layers": 0},
    "A5": {"unidirectional_attention": True},
    "A6": {"ecg_resnet_blocks": 4},
    "A7": {"ecg_preprocessing": False},
}

ABLATION_DESCRIPTIONS: dict[str, str] = {
    "A1": "Cross-entropy loss (no focal)",
    "A2": "No weighted sampler",
    "A3": "BERT fully frozen",
    "A4": "BERT fully unfrozen",
    "A5": "Unidirectional attention",
    "A6": "Shallow ECG encoder (4 blocks)",
    "A7": "No ECG preprocessing",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_loss(config: dict[str, Any]) -> nn.Module:
    """Return either FocalLoss or CrossEntropyLoss based on *config*."""
    if config.get("loss_type") == "cross_entropy":
        return nn.CrossEntropyLoss()
    return FocalLoss(
        gamma=config.get("focal_gamma", 2.0),
        alpha=config.get("focal_alpha", 0.25),
    )


def _build_model(config: dict[str, Any], device: str) -> nn.Module:
    """Build the proposed model with ablation-modified *config*."""
    from src.model import MultiModalTriageModel, MultiModalTriageModelOptimized

    model_cls = (
        MultiModalTriageModelOptimized
        if config.get("use_optimized", False)
        else MultiModalTriageModel
    )
    model = model_cls()
    return model.to(device)


def _apply_ablation(base_config: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy *base_config* and apply *overrides* on top."""
    merged = copy.deepcopy(base_config)
    merged.update(overrides)
    return merged


# ── Core routines ────────────────────────────────────────────────────────────


def run_single_ablation(
    ablation_id: str,
    config: dict[str, Any],
    base_config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
) -> dict[str, Any]:
    """Run a single ablation experiment.

    Parameters
    ----------
    ablation_id:
        Key from :data:`ABLATION_CONFIGS` (e.g. ``"A1"``).
    config:
        Ablation-specific overrides.
    base_config:
        Full baseline (proposed-model) configuration.
    train_loader / val_loader / test_loader:
        Data loaders.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict
        Test-set metrics for this ablation.
    """
    merged = _apply_ablation(base_config, config)
    logger.info("Ablation %s – %s", ablation_id, ABLATION_DESCRIPTIONS.get(ablation_id, ""))

    model = _build_model(merged, device)
    loss_fn = _build_loss(merged)
    optimizer = optim.AdamW(model.parameters(), lr=merged.get("lr", 2e-5))
    early_stopping = EarlyStopping(patience=merged.get("early_stopping_patience", 15))

    max_epochs = merged.get("max_epochs", 100)
    best_val_f1 = 0.0
    best_state: dict[str, Any] | None = None

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            torch.device(device),
            multi_task=merged.get("use_optimized", False),
        )
        val_metrics = validate(
            model,
            val_loader,
            loss_fn,
            torch.device(device),
            multi_task=merged.get("use_optimized", False),
        )

        val_f1 = val_metrics.get("f1_macro", 0.0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

        if early_stopping(val_f1):
            logger.info("%s early-stopped at epoch %d", ablation_id, epoch)
            break

    # Evaluate on test set with best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    from src.evaluate import evaluate_model

    test_metrics = evaluate_model(model, test_loader, device)
    test_metrics["ablation_id"] = ablation_id
    test_metrics["description"] = ABLATION_DESCRIPTIONS.get(ablation_id, "")
    return test_metrics


def run_all_ablations(config_path: str) -> pd.DataFrame:
    """Execute ablation experiments A1–A7 and return a results DataFrame.

    Parameters
    ----------
    config_path:
        Path to the *proposed-model* ``config.yaml``.

    Returns
    -------
    pd.DataFrame
        One row per ablation with metrics columns.
    """
    with open(config_path, "r") as f:
        base_config: dict[str, Any] = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Ablation device: %s", device)

    # ── Data loading ────────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    from transformers import AutoTokenizer

    from src.dataset import MultiModalTriageDataset, create_weighted_sampler
    from src.preprocess import ECGProcessor, TextCleaner, TriageDataCleaner, filter_and_align

    csv_path = base_config.get("csv_path", "data/triage.csv")
    ecg_path = base_config.get("ecg_path", "data/ecg.npy")
    seed = base_config.get("seed", 42)

    df, ecg = filter_and_align(csv_path, ecg_path, target_classes=[0, 1, 2])
    train_df, temp_df, train_ecg, temp_ecg = train_test_split(
        df, ecg, test_size=0.3, random_state=seed, stratify=df["label"]
    )
    val_df, test_df, val_ecg, test_ecg = train_test_split(
        temp_df, temp_ecg, test_size=0.5, random_state=seed, stratify=temp_df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    tc, ep, tdc = TextCleaner(), ECGProcessor(), TriageDataCleaner()

    train_ds = MultiModalTriageDataset(train_df, train_ecg, tokenizer, tc, ep, tdc)
    val_ds = MultiModalTriageDataset(val_df, val_ecg, tokenizer, tc, ep, tdc)
    test_ds = MultiModalTriageDataset(test_df, test_ecg, tokenizer, tc, ep, tdc)

    batch_size = base_config.get("batch_size", 32)
    sampler = create_weighted_sampler(train_df["label"].values)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Run ablations ───────────────────────────────────────────────────
    rows: list[dict[str, Any]] = []
    for ablation_id, overrides in ABLATION_CONFIGS.items():
        # A2 needs a train_loader *without* weighted sampler
        abl_train_loader = train_loader
        if ablation_id == "A2":
            abl_train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True
            )

        metrics = run_single_ablation(
            ablation_id,
            overrides,
            base_config,
            abl_train_loader,
            val_loader,
            test_loader,
            device,
        )
        rows.append(metrics)
        logger.info("%s → %s", ablation_id, metrics)

    return pd.DataFrame(rows)


# ── Visualisation ────────────────────────────────────────────────────────────


def plot_ablation_heatmap(
    results_df: pd.DataFrame,
    baseline_metrics: dict[str, float],
    save_path: str,
) -> None:
    """Annotated diverging heatmap of ablation deltas.

    Parameters
    ----------
    results_df:
        DataFrame returned by :func:`run_all_ablations`.
    baseline_metrics:
        ``{"f1_macro": …, "accuracy": …, "auc_roc": …, "critical_recall": …}``
        from the full proposed model.
    save_path:
        Image output path.
    """
    metric_cols = ["f1_macro", "accuracy", "auc_roc", "critical_recall"]
    display_labels = {
        "f1_macro": "F1-Macro",
        "accuracy": "Accuracy",
        "auc_roc": "AUC-ROC",
        "critical_recall": "Critical Recall",
    }

    available = [c for c in metric_cols if c in results_df.columns]

    delta_data: list[dict[str, Any]] = []
    for _, row in results_df.iterrows():
        entry: dict[str, Any] = {"ablation": row.get("ablation_id", "")}
        for col in available:
            entry[display_labels[col]] = row[col] - baseline_metrics.get(col, 0.0)
        delta_data.append(entry)

    delta_df = pd.DataFrame(delta_data).set_index("ablation")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        delta_df,
        annot=True,
        fmt=".3f",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Ablation Δ vs. Baseline")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    logger.info("Saved ablation heatmap → %s", save_path)


def analyze_ablation_results(
    results_df: pd.DataFrame,
    baseline_metrics: dict[str, float],
) -> str:
    """Generate a concise text summary of the ablation findings.

    Parameters
    ----------
    results_df:
        DataFrame from :func:`run_all_ablations`.
    baseline_metrics:
        Proposed-model reference metrics.

    Returns
    -------
    str
        Multi-line analysis text.
    """
    lines: list[str] = ["=" * 60, "Ablation Study – Summary", "=" * 60, ""]

    metric_cols = ["f1_macro", "accuracy", "auc_roc"]
    available = [c for c in metric_cols if c in results_df.columns]

    for _, row in results_df.iterrows():
        aid = row.get("ablation_id", "?")
        desc = row.get("description", "")
        deltas = {
            col: row[col] - baseline_metrics.get(col, 0.0)
            for col in available
        }
        sign = lambda v: "+" if v >= 0 else ""  # noqa: E731
        delta_str = ", ".join(
            f"{col} {sign(d)}{d:.4f}" for col, d in deltas.items()
        )
        lines.append(f"{aid} ({desc}): {delta_str}")

    # Identify largest drop
    if available:
        worst_col = available[0]
        worst_row = results_df.loc[
            (results_df[worst_col] - baseline_metrics.get(worst_col, 0.0)).idxmin()
        ]
        worst_aid = worst_row.get("ablation_id", "?")
        worst_delta = worst_row[worst_col] - baseline_metrics.get(worst_col, 0.0)
        lines.append("")
        lines.append(
            f"Largest {worst_col} drop: {worst_aid} ({worst_delta:+.4f})"
        )

    lines.append("")
    lines.append("=" * 60)
    summary = "\n".join(lines)
    logger.info("\n%s", summary)
    return summary


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the ablation study."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run ablation study (A1–A7)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to proposed-model config.yaml",
    )
    parser.add_argument(
        "--baseline-metrics",
        type=str,
        default=None,
        help="Path to YAML with proposed-model metrics (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ablations",
        help="Directory for plots and reports",
    )
    args = parser.parse_args()

    results_df = run_all_ablations(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load or default baseline metrics
    baseline_metrics: dict[str, float] = {
        "f1_macro": 0.0,
        "accuracy": 0.0,
        "auc_roc": 0.0,
        "critical_recall": 0.0,
    }
    if args.baseline_metrics and Path(args.baseline_metrics).exists():
        with open(args.baseline_metrics, "r") as f:
            baseline_metrics.update(yaml.safe_load(f))

    plot_ablation_heatmap(results_df, baseline_metrics, str(out / "ablation_heatmap.png"))

    summary = analyze_ablation_results(results_df, baseline_metrics)
    summary_path = out / "ablation_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    results_df.to_csv(out / "ablation_results.csv", index=False)
    logger.info("Ablation outputs saved to %s", out)


if __name__ == "__main__":
    main()
