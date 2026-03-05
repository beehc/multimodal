"""Interpretability module for the Multi-Modal Emergency Triage System.

Provides SHAP-based feature importance for tabular inputs, attention
visualisations for text and ECG modalities, and modality-contribution
analysis for models with auxiliary heads.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.dataset import TABULAR_COLS, MultiModalTriageDataset
from src.model import MultiModalTriageModel, MultiModalTriageModelOptimized

logger = logging.getLogger(__name__)

# ── Class labels ────────────────────────────────────────────────────
CLASS_NAMES = ["ESI-1 (Immediate)", "ESI-2 (Emergent)", "ESI-3 (Urgent)"]


# ── Helpers ─────────────────────────────────────────────────────────

def _collect_batches(
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 200,
) -> dict[str, torch.Tensor]:
    """Iterate over *dataloader* and return stacked tensors up to *max_samples*."""
    collected: dict[str, list[torch.Tensor]] = {
        "input_ids": [],
        "attention_mask": [],
        "ecg": [],
        "tabular": [],
        "label": [],
    }
    n = 0
    for batch in dataloader:
        bs = batch["tabular"].size(0)
        take = min(bs, max_samples - n)
        for key in collected:
            collected[key].append(batch[key][:take].to(device))
        n += take
        if n >= max_samples:
            break

    return {k: torch.cat(v, dim=0) for k, v in collected.items()}


def _is_optimized_model(model: nn.Module) -> bool:
    """Return *True* if *model* is the v7 optimised variant with aux heads."""
    return isinstance(model, MultiModalTriageModelOptimized)


def _model_forward(model: nn.Module, **kwargs: Any) -> torch.Tensor:
    """Call *model* and always return the **main** logits tensor."""
    out = model(**kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out


def _extract_attention_weights(
    module: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ecg: torch.Tensor,
    tabular: torch.Tensor,
) -> list[torch.Tensor]:
    """Run a forward pass and capture cross-attention weights via hooks.

    Returns a list of attention-weight tensors gathered from every
    ``nn.MultiheadAttention`` layer found inside the model.
    """
    attn_weights: list[torch.Tensor] = []

    handles: list[torch.utils.hooks.RemovableHook] = []
    for submodule in module.modules():
        if isinstance(submodule, nn.MultiheadAttention):
            def _hook(
                _mod: nn.Module,
                _inp: Any,
                output: Any,
                _store: list = attn_weights,
            ) -> None:
                # nn.MultiheadAttention returns (attn_output, attn_weights)
                if isinstance(output, tuple) and len(output) >= 2:
                    weights = output[1]
                    if weights is not None:
                        _store.append(weights.detach().cpu())

            # Temporarily enable weight output
            original_flag = getattr(submodule, "need_weights", True)
            submodule.need_weights = True  # type: ignore[attr-defined]
            handle = submodule.register_forward_hook(_hook)
            handles.append(handle)

    with torch.no_grad():
        _model_forward(
            module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            ecg=ecg,
            tabular=tabular,
        )

    for h in handles:
        h.remove()

    return attn_weights


# ── 1. SHAP tabular feature importance ──────────────────────────────

def compute_shap_values(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_names: list[str] | None = None,
    save_path: str = "outputs/figures/shap_tabular.png",
    max_samples: int = 100,
    n_background: int = 50,
) -> np.ndarray:
    """Compute SHAP values for the tabular modality and save a bar chart.

    Parameters
    ----------
    model : nn.Module
        Trained triage model (v6 or v7).
    dataloader : DataLoader
        Test-set data loader.
    device : torch.device
        Computation device.
    feature_names : list[str] | None
        Names for each tabular feature.  Falls back to
        :pydata:`TABULAR_COLS` when *None*.
    save_path : str
        Destination path for the bar-chart figure.
    max_samples : int
        Maximum number of samples to explain.
    n_background : int
        Number of background samples for the KernelSHAP explainer.

    Returns
    -------
    np.ndarray
        SHAP values array of shape ``(n_samples, n_features, n_classes)``.
    """
    if feature_names is None:
        feature_names = list(TABULAR_COLS)

    model.eval()
    data = _collect_batches(dataloader, device, max_samples=max_samples)

    # Fixed context tensors (median across the sample)
    fixed_input_ids = data["input_ids"][:1].expand(1, -1)
    fixed_attention_mask = data["attention_mask"][:1].expand(1, -1)
    fixed_ecg = data["ecg"][:1]

    # Wrapper: tabular numpy → softmax probabilities numpy
    def _predict(tabular_np: np.ndarray) -> np.ndarray:
        t = torch.tensor(tabular_np, dtype=torch.float32, device=device)
        batch_size = t.size(0)
        ids = fixed_input_ids.expand(batch_size, -1)
        mask = fixed_attention_mask.expand(batch_size, -1)
        e = fixed_ecg.expand(batch_size, -1, -1)
        with torch.no_grad():
            logits = _model_forward(
                model, input_ids=ids, attention_mask=mask, ecg=e, tabular=t,
            )
        probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    tabular_np = data["tabular"].cpu().numpy()
    background = tabular_np[:n_background]
    explain = tabular_np[:max_samples]

    logger.info(
        "Computing SHAP values for %d samples (background=%d) …",
        explain.shape[0],
        background.shape[0],
    )
    explainer = shap.KernelExplainer(_predict, background)
    shap_values = explainer.shap_values(explain, silent=True)  # list of arrays

    # shap_values: list[ndarray(n, features)] per class → (n, features, classes)
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    # ── Plot mean |SHAP| per feature ────────────────────────────────
    mean_abs = np.mean(np.abs(shap_values), axis=(0, 2))  # (n_features,)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in order],
        mean_abs[order],
        color=sns.color_palette("viridis", len(feature_names)),
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Tabular Feature Importance (KernelSHAP)")
    ax.invert_yaxis()
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("SHAP bar chart saved to %s", save_path)

    return shap_values


# ── 2. Text-attention visualisation ─────────────────────────────────

def visualize_text_attention(
    model: nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ecg: torch.Tensor,
    tabular: torch.Tensor,
    device: torch.device,
    save_path: str = "outputs/figures/text_attention.png",
) -> None:
    """Visualise cross-attention weights over input text tokens.

    Extracts attention from every ``nn.MultiheadAttention`` layer
    inside *model*, averages across heads and layers, and produces a
    heatmap of per-token attention intensity.

    Parameters
    ----------
    model : nn.Module
        Trained triage model.
    tokenizer
        HuggingFace tokenizer used to decode token ids.
    input_ids, attention_mask, ecg, tabular : torch.Tensor
        A single-sample batch (leading dim = 1).
    device : torch.device
        Computation device.
    save_path : str
        Destination for the saved figure.
    """
    model.eval()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    ecg = ecg.to(device)
    tabular = tabular.to(device)

    # Ensure batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask.unsqueeze(0)
        ecg = ecg.unsqueeze(0)
        tabular = tabular.unsqueeze(0)

    attn_weights = _extract_attention_weights(
        model, input_ids, attention_mask, ecg, tabular,
    )

    if not attn_weights:
        logger.warning("No attention weights captured — skipping text attention plot.")
        return

    # Decode tokens
    token_ids = input_ids[0].cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # Trim to real (non-padding) tokens
    mask_len = int(attention_mask[0].sum().item())
    tokens = tokens[:mask_len]

    # Average attention over heads for the first captured layer (text→ECG path)
    # Shape: (batch, n_heads, query_len, key_len) or (batch, query_len, key_len)
    attn = attn_weights[0][0]  # first sample
    if attn.dim() == 2:
        attn_scores = attn[:mask_len].mean(dim=-1).numpy()
    else:
        # (n_heads, q, k) → average heads then average over key dimension
        attn_scores = attn.mean(dim=0)[:mask_len].mean(dim=-1).numpy()

    # Normalise to [0, 1]
    attn_min, attn_max = attn_scores.min(), attn_scores.max()
    if attn_max - attn_min > 0:
        attn_scores = (attn_scores - attn_min) / (attn_max - attn_min)

    # ── Heatmap ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(10, mask_len * 0.45), 3))
    sns.heatmap(
        attn_scores.reshape(1, -1),
        xticklabels=tokens,
        yticklabels=["Attention"],
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Attention weight"},
    )
    ax.set_title("Cross-Attention over Text Tokens")
    plt.xticks(rotation=60, ha="right", fontsize=7)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Text attention heatmap saved to %s", save_path)


# ── 3. ECG-attention visualisation ──────────────────────────────────

def visualize_ecg_attention(
    model: nn.Module,
    ecg_signal: torch.Tensor,
    device: torch.device,
    save_path: str = "outputs/figures/ecg_attention.png",
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    tabular: torch.Tensor | None = None,
) -> None:
    """Plot an ECG signal with a cross-attention intensity overlay.

    Parameters
    ----------
    model : nn.Module
        Trained triage model.
    ecg_signal : torch.Tensor
        ECG tensor of shape ``(12, 5000)`` or ``(1, 12, 5000)``.
    device : torch.device
        Computation device.
    save_path : str
        Destination for the saved figure.
    input_ids, attention_mask, tabular : torch.Tensor | None
        Companion inputs required for a full forward pass.  When *None*,
        zero tensors of the expected shape are used as placeholders.
    """
    model.eval()

    if ecg_signal.dim() == 2:
        ecg_signal = ecg_signal.unsqueeze(0)
    ecg_signal = ecg_signal.to(device)

    # Build placeholder tensors if not supplied
    if input_ids is None:
        input_ids = torch.zeros(1, 128, dtype=torch.long, device=device)
    else:
        input_ids = input_ids.to(device)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

    if attention_mask is None:
        attention_mask = torch.ones(1, 128, dtype=torch.long, device=device)
    else:
        attention_mask = attention_mask.to(device)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

    if tabular is None:
        tabular = torch.zeros(1, len(TABULAR_COLS), dtype=torch.float32, device=device)
    else:
        tabular = tabular.to(device)
        if tabular.dim() == 1:
            tabular = tabular.unsqueeze(0)

    attn_weights = _extract_attention_weights(
        model, input_ids, attention_mask, ecg_signal, tabular,
    )

    if not attn_weights:
        logger.warning("No attention weights captured — skipping ECG attention plot.")
        return

    # Use the second captured attention layer (ECG←Text path) if available,
    # otherwise fall back to the first.
    attn = attn_weights[1] if len(attn_weights) > 1 else attn_weights[0]
    attn = attn[0]  # first sample

    # Reduce to 1-D attention profile; average over heads and query dim
    if attn.dim() == 2:
        ecg_attn = attn.mean(dim=0).numpy()
    else:
        ecg_attn = attn.mean(dim=0).mean(dim=0).numpy()

    # Interpolate attention to match ECG time axis (5000 samples)
    n_timepoints = ecg_signal.shape[-1]
    attn_interp = np.interp(
        np.linspace(0, 1, n_timepoints),
        np.linspace(0, 1, len(ecg_attn)),
        ecg_attn,
    )
    attn_interp = (attn_interp - attn_interp.min()) / (
        attn_interp.max() - attn_interp.min() + 1e-8
    )

    # ── Plot (Lead II, index 1) ─────────────────────────────────────
    lead_idx = min(1, ecg_signal.shape[1] - 1)
    signal = ecg_signal[0, lead_idx].cpu().numpy()
    time_s = np.arange(n_timepoints) / 500.0  # 500 Hz

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(time_s, signal, color="steelblue", linewidth=0.8, label="ECG Lead II")
    ax.fill_between(
        time_s,
        signal.min(),
        signal.max(),
        where=attn_interp > np.percentile(attn_interp, 75),
        alpha=0.35,
        color="salmon",
        label="High attention (>75th pctl)",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalised)")
    ax.set_title("ECG Signal with Cross-Attention Overlay")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("ECG attention plot saved to %s", save_path)


# ── 4. Modality contribution analysis ──────────────────────────────

def analyze_modality_contributions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    save_path: str = "outputs/figures/modality_contributions.png",
    max_samples: int = 300,
) -> dict[str, Any]:
    """Compare per-modality auxiliary predictions with the fused prediction.

    Only meaningful for the v7 ``MultiModalTriageModelOptimized`` model
    that exposes auxiliary heads.  For v6 models the function logs a
    warning and returns an empty dictionary.

    Parameters
    ----------
    model : nn.Module
        Trained triage model.
    dataloader : DataLoader
        Test-set data loader.
    device : torch.device
        Computation device.
    save_path : str
        Destination for the saved figure.
    max_samples : int
        Maximum samples to analyse.

    Returns
    -------
    dict
        Dictionary with keys ``"fused"``, ``"text"``, ``"ecg"``,
        ``"tabular"`` each mapping to prediction arrays, plus a
        ``"conflict_rate"`` scalar.
    """
    if not _is_optimized_model(model):
        logger.warning(
            "Model does not have auxiliary heads — skipping modality contribution analysis."
        )
        return {}

    model.eval()
    data = _collect_batches(dataloader, device, max_samples=max_samples)

    with torch.no_grad():
        out = model(
            input_ids=data["input_ids"],
            attention_mask=data["attention_mask"],
            ecg=data["ecg"],
            tabular=data["tabular"],
        )

    if not isinstance(out, tuple) or len(out) < 2:
        logger.warning("Model output is not a (main, aux) tuple — skipping.")
        return {}

    main_logits, aux_list = out
    main_probs = torch.softmax(main_logits, dim=-1).cpu().numpy()
    main_preds = main_logits.argmax(dim=-1).cpu().numpy()

    modality_names = ["text", "ecg", "tabular"]
    aux_preds: dict[str, np.ndarray] = {}
    aux_confs: dict[str, np.ndarray] = {}

    for name, logits in zip(modality_names, aux_list):
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        aux_preds[name] = logits.argmax(dim=-1).cpu().numpy()
        aux_confs[name] = probs.max(axis=-1)

    fused_conf = main_probs.max(axis=-1)

    # ── Conflict analysis ───────────────────────────────────────────
    n = len(main_preds)
    conflicts = np.zeros(n, dtype=bool)
    for name in modality_names:
        conflicts |= aux_preds[name] != main_preds
    conflict_rate = float(conflicts.mean())
    logger.info("Modality conflict rate: %.2f%%", conflict_rate * 100)

    # ── Visualisation: confidence distributions ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Mean confidence per modality
    conf_means = {
        "Fused": float(fused_conf.mean()),
        **{name.capitalize(): float(aux_confs[name].mean()) for name in modality_names},
    }
    colours = sns.color_palette("Set2", len(conf_means))
    axes[0].bar(conf_means.keys(), conf_means.values(), color=colours)
    axes[0].set_ylabel("Mean prediction confidence")
    axes[0].set_title("Average Confidence by Modality")
    axes[0].set_ylim(0, 1)
    for i, (k, v) in enumerate(conf_means.items()):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    # (b) Conflict breakdown
    per_modality_conflict = {
        name.capitalize(): float((aux_preds[name] != main_preds).mean())
        for name in modality_names
    }
    axes[1].bar(
        per_modality_conflict.keys(),
        per_modality_conflict.values(),
        color=sns.color_palette("Reds_r", len(per_modality_conflict)),
    )
    axes[1].set_ylabel("Fraction disagreeing with fused prediction")
    axes[1].set_title(f"Modality Conflicts (overall rate {conflict_rate:.1%})")
    axes[1].set_ylim(0, 1)
    for i, (k, v) in enumerate(per_modality_conflict.items()):
        axes[1].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=9)

    fig.suptitle("Modality Contribution Analysis", fontsize=13, y=1.02)
    fig.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Modality contribution plot saved to %s", save_path)

    return {
        "fused": main_preds,
        "text": aux_preds.get("text"),
        "ecg": aux_preds.get("ecg"),
        "tabular": aux_preds.get("tabular"),
        "conflict_rate": conflict_rate,
    }


# ── 5. Main entry point ────────────────────────────────────────────

def main(config_path: str = "configs/config.yaml") -> None:
    """Run all interpretability analyses and save figures.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file used during training.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
    )

    # ── Load config ─────────────────────────────────────────────────
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    figures_dir = Path(config.get("output", {}).get("figures_dir", "outputs/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Build model ─────────────────────────────────────────────────
    model_cfg = config.get("model", {})
    model_version = model_cfg.get("version", "v6")

    if model_version == "v7":
        model = MultiModalTriageModelOptimized(config)
    else:
        model = MultiModalTriageModel()

    # ── Load checkpoint ─────────────────────────────────────────────
    ckpt_dir = Path(
        config.get("output", {}).get("checkpoint_dir", "outputs/checkpoints")
    )
    ckpt_path = ckpt_dir / "best_model.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found at %s — aborting.", ckpt_path)
        return

    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", ckpt_path)

    # ── Prepare test data ───────────────────────────────────────────
    from transformers import AutoTokenizer

    from src.preprocess import ECGProcessor, TextCleaner, TriageDataCleaner

    data_cfg = config.get("data", {})
    processed_dir = Path(data_cfg.get("processed_dir", "data/processed"))
    max_length = model_cfg.get("max_length", 128)
    use_interaction = model_cfg.get("tabular", {}).get(
        "interaction_features", False
    )

    bert_model_name = model_cfg.get("bert_model", "emilyalsentzer/Bio_ClinicalBERT")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    text_cleaner = TextCleaner()
    ecg_processor = ECGProcessor()
    tabular_cleaner = TriageDataCleaner()

    import pandas as pd

    test_csv = processed_dir / "test.csv"
    test_ecg_path = processed_dir / "test_ecg.npy"
    if not test_csv.exists() or not test_ecg_path.exists():
        logger.error(
            "Processed test data not found in %s — aborting.", processed_dir,
        )
        return

    test_df = pd.read_csv(test_csv)
    test_ecg = np.load(str(test_ecg_path))

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
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    # ── Run analyses ────────────────────────────────────────────────
    logger.info("═══ SHAP tabular feature importance ═══")
    try:
        compute_shap_values(
            model,
            test_loader,
            device,
            feature_names=list(TABULAR_COLS),
            save_path=str(figures_dir / "shap_tabular.png"),
        )
    except Exception:
        logger.exception("SHAP analysis failed")

    logger.info("═══ Text attention visualisation ═══")
    try:
        sample = next(iter(test_loader))
        visualize_text_attention(
            model,
            tokenizer,
            input_ids=sample["input_ids"][:1],
            attention_mask=sample["attention_mask"][:1],
            ecg=sample["ecg"][:1],
            tabular=sample["tabular"][:1],
            device=device,
            save_path=str(figures_dir / "text_attention.png"),
        )
    except Exception:
        logger.exception("Text attention visualisation failed")

    logger.info("═══ ECG attention visualisation ═══")
    try:
        sample = next(iter(test_loader))
        visualize_ecg_attention(
            model,
            ecg_signal=sample["ecg"][:1],
            device=device,
            save_path=str(figures_dir / "ecg_attention.png"),
            input_ids=sample["input_ids"][:1],
            attention_mask=sample["attention_mask"][:1],
            tabular=sample["tabular"][:1],
        )
    except Exception:
        logger.exception("ECG attention visualisation failed")

    logger.info("═══ Modality contribution analysis ═══")
    try:
        analyze_modality_contributions(
            model,
            test_loader,
            device,
            save_path=str(figures_dir / "modality_contributions.png"),
        )
    except Exception:
        logger.exception("Modality contribution analysis failed")

    logger.info("All interpretability analyses complete. Figures in %s", figures_dir)


if __name__ == "__main__":
    main()
