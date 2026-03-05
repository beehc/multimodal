"""Train all baseline models (B1–B7) defined in baseline_configs.yaml.

Usage:
    python -m baselines.train_baselines --config configs/baseline_configs.yaml
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
from src.losses import FocalLoss
from src.train import EarlyStopping, train_one_epoch, validate

logger = logging.getLogger(__name__)

# ── Model registry ──────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[str, type] = {
    "B1_tabular_mlp": TabularOnlyMLP,
    "B2_text_bert": TextOnlyBERT,
    "B3_ecg_resnet": ECGOnlyResNet,
    "B4_early_fusion": EarlyFusionModel,
    "B5_late_fusion": LateFusionModel,
    "B6_crossattn_only": CrossAttnOnlyModel,
    "B7_xgboost": XGBoostBaseline,
}


def _build_pytorch_model(model_name: str, config: dict[str, Any]) -> nn.Module:
    """Instantiate a PyTorch baseline model from *config*."""
    cls = MODEL_REGISTRY[model_name]
    kwargs: dict[str, Any] = {}

    if model_name == "B1_tabular_mlp":
        kwargs["dropout"] = config.get("dropout", 0.3)
    elif model_name == "B2_text_bert":
        kwargs["freeze_layers"] = config.get("freeze_layers", 9)
        kwargs["dropout"] = config.get("dropout", 0.5)
    elif model_name == "B3_ecg_resnet":
        kwargs["dropout"] = config.get("dropout", 0.5)
    elif model_name in {"B4_early_fusion", "B5_late_fusion", "B6_crossattn_only"}:
        kwargs["dropout"] = config.get("dropout", 0.3)

    return cls(**kwargs)


# ── Training helpers ─────────────────────────────────────────────────────────


def _train_pytorch_baseline(
    model: nn.Module,
    config: dict[str, Any],
    shared: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    checkpoint_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    """Run a standard PyTorch training loop with FocalLoss + AdamW."""
    model = model.to(device)
    loss_fn = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.get("lr", 1e-3))
    early_stopping = EarlyStopping(
        patience=shared.get("early_stopping_patience", 15),
    )

    max_epochs = shared.get("max_epochs", 100)
    best_val_f1 = 0.0
    best_metrics: dict[str, Any] = {}
    checkpoint_path = checkpoint_dir / f"{model_name}.pt"

    for epoch in range(1, max_epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, torch.device(device)
        )
        val_metrics = validate(model, val_loader, loss_fn, torch.device(device))

        val_f1 = val_metrics.get("f1_macro", 0.0)
        logger.info(
            "%s epoch %d – train_loss=%.4f  val_f1=%.4f",
            model_name,
            epoch,
            train_metrics.get("loss", float("nan")),
            val_f1,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_metrics = {**val_metrics, "best_epoch": epoch}
            torch.save(model.state_dict(), checkpoint_path)

        if early_stopping(val_f1):
            logger.info("%s early-stopped at epoch %d", model_name, epoch)
            break

    best_metrics["checkpoint_path"] = str(checkpoint_path)
    return best_metrics


def _train_xgboost_baseline(
    config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    """Prepare features and fit an XGBoost model."""
    xgb_model = XGBoostBaseline(config)

    # Flatten loader batches into feature / label arrays
    def _collect(loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
        features_list: list[np.ndarray] = []
        labels_list: list[np.ndarray] = []
        for batch in loader:
            feat_parts: list[np.ndarray] = []
            if "tabular" in batch:
                feat_parts.append(batch["tabular"].numpy())
            if "tfidf" in batch:
                feat_parts.append(batch["tfidf"].numpy())
            if "ecg_features" in batch:
                feat_parts.append(batch["ecg_features"].numpy())
            if feat_parts:
                features_list.append(np.concatenate(feat_parts, axis=1))
            labels_list.append(batch["label"].numpy())
        return np.concatenate(features_list), np.concatenate(labels_list)

    X_train, y_train = _collect(train_loader)
    X_val, y_val = _collect(val_loader)

    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    checkpoint_path = checkpoint_dir / "B7_xgboost.pkl"
    with open(checkpoint_path, "wb") as f:
        pickle.dump(xgb_model, f)

    # Quick validation metrics
    from sklearn.metrics import accuracy_score, f1_score

    preds = xgb_model.predict(X_val)
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_val, preds)),
        "f1_macro": float(f1_score(y_val, preds, average="macro")),
        "checkpoint_path": str(checkpoint_path),
    }
    return metrics


# ── Public API ───────────────────────────────────────────────────────────────


def train_single_baseline(
    model_name: str,
    config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train one baseline model and return its best validation metrics.

    Parameters
    ----------
    model_name:
        Key from *baseline_configs.yaml* (e.g. ``"B1_tabular_mlp"``).
    config:
        Model-specific hyper-parameters merged with the ``shared`` section.
    train_loader / val_loader:
        PyTorch data loaders.
    device:
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    dict
        Validation metrics including ``checkpoint_path``.
    """
    shared = config.get("shared", {})
    checkpoint_dir = Path(config.get("output_dir", "outputs/baselines"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if model_name == "B7_xgboost":
        return _train_xgboost_baseline(config, train_loader, val_loader, checkpoint_dir)

    model = _build_pytorch_model(model_name, config)
    return _train_pytorch_baseline(
        model, config, shared, train_loader, val_loader, device, checkpoint_dir, model_name
    )


def train_all_baselines(config_path: str) -> dict[str, dict[str, Any]]:
    """Train every baseline listed in *config_path* sequentially.

    Parameters
    ----------
    config_path:
        Path to ``baseline_configs.yaml``.

    Returns
    -------
    dict
        ``{model_name: metrics_dict}`` for each baseline.
    """
    with open(config_path, "r") as f:
        all_configs: dict[str, Any] = yaml.safe_load(f)

    shared: dict[str, Any] = all_configs.get("shared", {})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: %s", device)

    # ── Data loading (reuses src.preprocess utilities) ───────────────────
    from src.dataset import MultiModalTriageDataset, create_weighted_sampler
    from src.preprocess import ECGProcessor, TextCleaner, TriageDataCleaner

    csv_path = all_configs.get("csv_path", "data/triage.csv")
    ecg_path = all_configs.get("ecg_path", "data/ecg.npy")

    from src.preprocess import filter_and_align

    df, ecg_array = filter_and_align(csv_path, ecg_path, target_classes=[0, 1, 2])

    text_cleaner = TextCleaner()
    ecg_processor = ECGProcessor()
    tabular_cleaner = TriageDataCleaner()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    from sklearn.model_selection import train_test_split

    train_df, val_df, train_ecg, val_ecg = train_test_split(
        df, ecg_array, test_size=0.2, random_state=shared.get("seed", 42), stratify=df["label"]
    )

    train_ds = MultiModalTriageDataset(
        train_df, train_ecg, tokenizer, text_cleaner, ecg_processor, tabular_cleaner
    )
    val_ds = MultiModalTriageDataset(
        val_df, val_ecg, tokenizer, text_cleaner, ecg_processor, tabular_cleaner
    )

    batch_size = shared.get("batch_size", 32)
    sampler = create_weighted_sampler(train_df["label"].values)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Sequential training ─────────────────────────────────────────────
    results: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_REGISTRY:
        model_config = copy.deepcopy(all_configs.get(model_name, {}))
        model_config["shared"] = shared
        model_config["output_dir"] = str(
            Path(all_configs.get("output_dir", "outputs/baselines"))
        )
        logger.info("Training %s …", model_name)
        metrics = train_single_baseline(
            model_name, model_config, train_loader, val_loader, device
        )
        results[model_name] = metrics
        logger.info("%s → %s", model_name, metrics)

    # ── Save summary ────────────────────────────────────────────────────
    summary_path = Path(all_configs.get("output_dir", "outputs/baselines")) / "results_summary.yaml"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    logger.info("Results saved to %s", summary_path)

    return results


# ── CLI entry point ──────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for training all baselines."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train baseline models (B1–B7)")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_configs.yaml",
        help="Path to baseline_configs.yaml",
    )
    args = parser.parse_args()
    train_all_baselines(args.config)


if __name__ == "__main__":
    main()
