"""Training pipeline for Multi-Modal Emergency Triage System."""
from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from src.dataset import MultiModalTriageDataset, create_weighted_sampler
from src.losses import FocalLoss, MultiTaskLoss
from src.model import MultiModalTriageModel, MultiModalTriageModelOptimized
from src.preprocess import (
    ECGProcessor,
    TextCleaner,
    TriageDataCleaner,
    filter_and_align,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Monitors a metric and signals when training should stop."""

    def __init__(
        self, patience: int = 15, min_delta: float = 0.001, mode: str = "max"
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best_score: float | None = None
        self._counter: int = 0
        self._should_save: bool = False

    # ------------------------------------------------------------------

    @property
    def best_score(self) -> float | None:
        """Return the best metric value observed so far."""
        return self._best_score

    @property
    def should_save(self) -> bool:
        """Return True when the latest metric is the new best."""
        return self._should_save

    # ------------------------------------------------------------------

    def _is_improvement(self, metric: float) -> bool:
        if self._best_score is None:
            return True
        if self.mode == "max":
            return metric > self._best_score + self.min_delta
        return metric < self._best_score - self.min_delta

    def __call__(self, metric: float) -> bool:
        """Update state and return ``True`` if training should stop."""
        if self._is_improvement(metric):
            self._best_score = metric
            self._counter = 0
            self._should_save = True
        else:
            self._counter += 1
            self._should_save = False
        return self._counter >= self.patience


class MetricTracker:
    """Accumulates per-epoch metric values."""

    def __init__(self) -> None:
        self.history: dict[str, list[float]] = {}

    def update(self, metrics: dict[str, float]) -> None:
        """Append each metric value to the corresponding history list."""
        for key, value in metrics.items():
            self.history.setdefault(key, []).append(value)

    def get_history(self) -> dict[str, list[float]]:
        """Return the full metric history."""
        return self.history


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    gradient_clip: float = 1.0,
    multi_task: bool = False,
) -> dict[str, float]:
    """Run one training epoch and return ``{'loss', 'f1_macro'}``."""
    model.train()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ecg = batch["ecg"].to(device)
        tabular = batch["tabular"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        if multi_task:
            main_logits, aux_logits_list = model(
                input_ids, attention_mask, ecg, tabular
            )
            total_loss, _ = loss_fn(main_logits, labels, aux_logits_list)
        else:
            logits = model(input_ids, attention_mask, ecg, tabular)
            total_loss = loss_fn(logits, labels)
            main_logits = logits

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        running_loss += total_loss.item() * labels.size(0)
        preds = main_logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": epoch_loss, "f1_macro": epoch_f1}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    multi_task: bool = False,
) -> dict[str, float]:
    """Run validation and return ``{'loss', 'f1_macro'}``."""
    model.eval()
    running_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ecg = batch["ecg"].to(device)
            tabular = batch["tabular"].to(device)
            labels = batch["label"].to(device)

            if multi_task:
                main_logits, aux_logits_list = model(
                    input_ids, attention_mask, ecg, tabular
                )
                total_loss, _ = loss_fn(main_logits, labels, aux_logits_list)
            else:
                logits = model(input_ids, attention_mask, ecg, tabular)
                total_loss = loss_fn(logits, labels)
                main_logits = logits

            running_loss += total_loss.item() * labels.size(0)
            preds = main_logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"loss": epoch_loss, "f1_macro": epoch_f1}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_training_curves(history: dict[str, list[float]], save_path: str) -> None:
    """Save a 2×1 figure with loss and F1-macro curves."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # -- Loss --
    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # -- F1-Macro --
    axes[1].plot(epochs, history["train_f1_macro"], label="Train F1-Macro")
    axes[1].plot(epochs, history["val_f1_macro"], label="Val F1-Macro")

    best_epoch = int(np.argmax(history["val_f1_macro"]))
    best_f1 = history["val_f1_macro"][best_epoch]
    axes[1].plot(
        best_epoch + 1, best_f1, marker="*", markersize=14, color="red",
        label=f"Best (epoch {best_epoch + 1}, {best_f1:.4f})",
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1-Macro")
    axes[1].set_title("Training / Validation F1-Macro")
    axes[1].legend()
    axes[1].grid(True)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Training curves saved to %s", save_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(config_path: str) -> None:
    """End-to-end training driven by a YAML config file."""
    # ---- logging ----------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # ---- config -----------------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ---- reproducibility --------------------------------------------------
    seed = config["reproducibility"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- data -------------------------------------------------------------
    csv_path = config["data"]["raw_csv"]
    ecg_path = config["data"]["raw_ecg"]
    target_classes = config["data"]["target_classes"]

    df, ecg_array = filter_and_align(csv_path, ecg_path, target_classes)
    logger.info("Dataset size after filtering: %d", len(df))

    from transformers import AutoTokenizer

    bert_model_name = config["model"]["bert_model"]
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    text_cleaner = TextCleaner()
    ecg_processor = ECGProcessor()
    tabular_cleaner = TriageDataCleaner()

    use_interaction = config["model"].get("tabular", {}).get(
        "interaction_features", False
    )

    # train / val split (stratified)
    labels = df["label"].values
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    train_ecg = ecg_array[train_idx]
    val_ecg = ecg_array[val_idx]

    max_length = config["model"].get("max_length", 128)

    train_ds = MultiModalTriageDataset(
        train_df, train_ecg, tokenizer, text_cleaner, ecg_processor,
        tabular_cleaner, max_length=max_length,
        use_interaction_features=use_interaction,
    )
    val_ds = MultiModalTriageDataset(
        val_df, val_ecg, tokenizer, text_cleaner, ecg_processor,
        tabular_cleaner, max_length=max_length,
        use_interaction_features=use_interaction,
    )

    batch_size = config["training"]["batch_size"]
    sampler = create_weighted_sampler(train_df["label"].values)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    # ---- model ------------------------------------------------------------
    model_version = config["model"]["version"]
    multi_task = config["model"].get("auxiliary_tasks", False)

    if model_version == "v7":
        model = MultiModalTriageModelOptimized(config)
    else:
        model = MultiModalTriageModel()

    model = model.to(device)
    logger.info("Model version: %s | multi-task: %s", model_version, multi_task)

    # ---- loss -------------------------------------------------------------
    loss_cfg = config["loss"]
    focal_loss = FocalLoss(
        gamma=loss_cfg.get("gamma", 2.0),
        alpha=loss_cfg.get("alpha", 0.25),
    )
    if multi_task:
        aux_weight = config["model"].get("auxiliary_weight", 0.1)
        loss_fn = MultiTaskLoss(
            base_loss_fn=focal_loss,
            auxiliary_weight=aux_weight,
        )
    else:
        loss_fn = focal_loss

    # ---- optimizer & scheduler -------------------------------------------
    train_cfg = config["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", train_cfg.get("lr", 2e-5)),
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )

    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get("T_0", 10),
        T_mult=sched_cfg.get("T_mult", 2),
    )

    # ---- training loop ----------------------------------------------------
    es_cfg = train_cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        min_delta=es_cfg.get("min_delta", 0.001),
        mode="max",
    )

    gradient_clip = train_cfg.get("gradient_clip", 1.0)
    max_epochs = train_cfg.get("max_epochs", 100)

    output_dir = Path(config["data"].get("processed_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best_model.pt"

    tracker = MetricTracker()

    for epoch in range(1, max_epochs + 1):
        logger.info("Epoch %d / %d", epoch, max_epochs)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            gradient_clip=gradient_clip, multi_task=multi_task,
        )
        val_metrics = validate(
            model, val_loader, loss_fn, device, multi_task=multi_task,
        )
        scheduler.step()

        tracker.update({
            "train_loss": train_metrics["loss"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_f1_macro": val_metrics["f1_macro"],
        })

        logger.info(
            "Train Loss=%.4f  F1=%.4f | Val Loss=%.4f  F1=%.4f",
            train_metrics["loss"], train_metrics["f1_macro"],
            val_metrics["loss"], val_metrics["f1_macro"],
        )

        stop = early_stopping(val_metrics["f1_macro"])

        if early_stopping.should_save:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_f1_macro": val_metrics["f1_macro"],
                    "config": config,
                },
                best_ckpt_path,
            )
            logger.info(
                "Saved best checkpoint (val F1=%.4f) → %s",
                val_metrics["f1_macro"], best_ckpt_path,
            )

        if stop:
            logger.info(
                "Early stopping triggered after %d epochs (best F1=%.4f)",
                epoch, early_stopping.best_score,
            )
            break

    # ---- post-training ----------------------------------------------------
    plot_training_curves(
        tracker.get_history(),
        save_path=str(output_dir / "training_curves.png"),
    )
    logger.info("Training complete.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train triage model")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
