"""Evaluation pipeline for the Multi-Modal Emergency Triage System.

Loads a trained checkpoint, runs inference on the held-out test set, and
produces metrics, confusion-matrix / ROC-curve visualisations, and a
detailed classification report.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset import MultiModalTriageDataset, create_weighted_sampler
from src.model import MultiModalTriageModel, MultiModalTriageModelOptimized
from src.preprocess import (
    ECGProcessor,
    TextCleaner,
    TriageDataCleaner,
    filter_and_align,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    model_class: type,
    model_kwargs: dict[str, Any],
    device: str,
) -> nn.Module:
    """Instantiate a model from *model_class*, restore weights, and prepare
    it for inference.

    Args:
        checkpoint_path: Path to the saved ``.pt`` checkpoint.
        model_class: The ``nn.Module`` subclass to instantiate.
        model_kwargs: Keyword arguments forwarded to the model constructor.
        device: Target device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        The model with loaded weights, moved to *device* and set to eval mode.
    """
    model = model_class(**model_kwargs)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Loaded model from %s onto %s", checkpoint_path, device)
    return model


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    multi_task: bool = False,
) -> dict[str, Any]:
    """Run the model on every batch in *dataloader* and compute metrics.

    Args:
        model: A trained ``nn.Module`` in eval mode.
        dataloader: Test ``DataLoader``.
        device: Device string.
        multi_task: If ``True`` the model returns
            ``(main_logits, aux_logits_list)`` instead of plain logits.

    Returns:
        Dictionary with keys ``'predictions'``, ``'probabilities'``,
        ``'labels'``, ``'accuracy'``, ``'f1_macro'``, and ``'auc_roc'``.
    """
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ecg = batch["ecg"].to(device)
            tabular = batch["tabular"].to(device)
            labels = batch["label"].to(device)

            if multi_task:
                main_logits, _ = model(input_ids, attention_mask, ecg, tabular)
            else:
                main_logits = model(input_ids, attention_mask, ecg, tabular)

            all_logits.append(main_logits.cpu())
            all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels_t = torch.cat(all_labels, dim=0)

    probabilities = torch.softmax(logits, dim=1).numpy()
    predictions = logits.argmax(dim=1).numpy()
    labels_np = labels_t.numpy()

    accuracy = float((predictions == labels_np).mean())
    f1_macro = float(
        f1_score(labels_np, predictions, average="macro", zero_division=0)
    )
    auc_roc = float(
        roc_auc_score(labels_np, probabilities, multi_class="ovr", average="macro")
    )

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "labels": labels_np,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "auc_roc": auc_roc,
    }


# ---------------------------------------------------------------------------
# Visualisations & reporting
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    """Plot a row-normalised confusion matrix and save to *save_path*."""
    cm = confusion_matrix(labels, predictions)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Normalized Confusion Matrix on Test Set")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_path)


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    """Plot One-vs-Rest ROC curves (one per class) and save to *save_path*."""
    unique_classes = sorted(np.unique(labels))
    labels_bin = label_binarize(labels, classes=unique_classes)

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)
    for idx, cls in enumerate(unique_classes):
        fpr, tpr, _ = roc_curve(labels_bin[:, idx], probabilities[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            color=colors[idx % len(colors)],
            label=f"{class_names[idx]} (AUC = {roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ROC curves saved to %s", save_path)


def generate_classification_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
    save_path: str,
) -> None:
    """Generate an ``sklearn`` classification report and write it to disk."""
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        zero_division=0,
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(report, encoding="utf-8")
    logger.info("Classification report saved to %s", save_path)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(config_path: str) -> None:
    """End-to-end evaluation driven by a YAML configuration file.

    1. Load configuration.
    2. Prepare the test split (same seed / split ratio as training).
    3. Restore the best checkpoint.
    4. Run evaluation and collect metrics.
    5. Generate confusion matrix, ROC curves, and classification report.
    6. Print a summary to the console.
    """
    # ---- logging ----------------------------------------------------------
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    # ---- config -----------------------------------------------------------
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["reproducibility"]["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ---- data -------------------------------------------------------------
    csv_path = config["data"]["raw_csv"]
    ecg_path = config["data"]["raw_ecg"]
    target_classes = config["data"]["target_classes"]

    df, ecg_array = filter_and_align(csv_path, ecg_path, target_classes)
    logger.info("Dataset size after filtering: %d", len(df))

    bert_model_name = config["model"]["bert_model"]
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    text_cleaner = TextCleaner()
    ecg_processor = ECGProcessor()
    tabular_cleaner = TriageDataCleaner()

    use_interaction = config["model"].get("tabular", {}).get(
        "interaction_features", False
    )

    # Reproduce the same train/test split used during training
    labels = df["label"].values
    _, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )

    test_df = df.iloc[test_idx].reset_index(drop=True)
    test_ecg = ecg_array[test_idx]

    max_length = config["model"].get("max_length", 128)
    test_ds = MultiModalTriageDataset(
        test_df,
        test_ecg,
        tokenizer,
        text_cleaner,
        ecg_processor,
        tabular_cleaner,
        max_length=max_length,
        use_interaction_features=use_interaction,
    )

    batch_size = config["training"]["batch_size"]
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    # ---- model ------------------------------------------------------------
    model_version = config["model"]["version"]
    multi_task = config["model"].get("auxiliary_tasks", False)

    if model_version == "v7":
        model_class = MultiModalTriageModelOptimized
        model_kwargs: dict[str, Any] = {"config": config}
    else:
        model_class = MultiModalTriageModel
        model_kwargs = {}

    checkpoint_dir = Path(
        config["output"].get("checkpoint_dir", "outputs/checkpoints/")
    )
    processed_dir = Path(config["data"].get("processed_dir", "outputs"))
    # Try the configured checkpoint dir first, fall back to processed dir
    best_ckpt = checkpoint_dir / "best_model.pt"
    if not best_ckpt.exists():
        best_ckpt = processed_dir / "best_model.pt"

    model = load_model(
        checkpoint_path=str(best_ckpt),
        model_class=model_class,
        model_kwargs=model_kwargs,
        device=str(device),
    )

    # ---- evaluation -------------------------------------------------------
    results = evaluate_model(model, test_loader, str(device), multi_task=multi_task)

    # ---- outputs ----------------------------------------------------------
    class_names = [f"Triage {c}" for c in target_classes]
    figures_dir = Path(config["output"].get("figures_dir", "outputs/figures/"))
    reports_dir = Path(config["output"].get("reports_dir", "outputs/reports/"))

    plot_confusion_matrix(
        results["labels"],
        results["predictions"],
        class_names,
        save_path=str(figures_dir / "confusion_matrix.png"),
    )
    plot_roc_curves(
        results["labels"],
        results["probabilities"],
        class_names,
        save_path=str(figures_dir / "roc_curves.png"),
    )
    generate_classification_report(
        results["labels"],
        results["predictions"],
        class_names,
        save_path=str(reports_dir / "classification_report.txt"),
    )

    # ---- summary ----------------------------------------------------------
    logger.info(
        "Test Accuracy: %.4f | F1-Macro: %.4f | AUC-ROC: %.4f",
        results["accuracy"],
        results["f1_macro"],
        results["auc_roc"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate triage model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML config file",
    )
    args = parser.parse_args()
    main(args.config)
