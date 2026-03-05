"""Dataset module for the Multi-Modal Emergency Triage System.

Provides PyTorch ``Dataset`` implementations for multi-modal and
single-modal training, along with a weighted-sampler utility to
address class imbalance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from src.preprocess import ECGProcessor, TextCleaner, TriageDataCleaner

logger = logging.getLogger(__name__)

# Column names expected in the incoming dataframe
TABULAR_COLS: list[str] = [
    "temperature",
    "heartrate",
    "resprate",
    "o2sat",
    "sbp",
    "dbp",
    "pain",
    "gender",
    "age",
]

TEXT_COL: str = "chiefcomplaint"


# ---------------------------------------------------------------------------
# Multi-modal dataset
# ---------------------------------------------------------------------------


class MultiModalTriageDataset(Dataset):
    """Multi-modal dataset returning text, ECG, tabular features, and labels.

    All heavy preprocessing (cleaning, filtering, tokenisation) is
    performed once in ``__init__`` so that ``__getitem__`` is a cheap
    index-based lookup.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Must contain the columns listed in :pydata:`TABULAR_COLS`, a
        ``chiefcomplaint`` text column, and a ``label`` column.
    ecg_array : np.ndarray
        Raw ECG signals with shape ``(N, 12, 5000)``.
    tokenizer
        A HuggingFace tokenizer (Bio-ClinicalBERT).
    text_cleaner : TextCleaner
        Instance used to clean chief-complaint strings.
    ecg_processor : ECGProcessor
        Processor that filters and normalises ECG signals.
    tabular_cleaner : TriageDataCleaner
        Cleaner for tabular vital-sign data.
    max_length : int
        Maximum token length for the tokenizer.
    use_interaction_features : bool
        If ``True``, append 36 pairwise interaction features to the 9
        original tabular features (yielding 45 total).
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        ecg_array: np.ndarray,
        tokenizer: Any,
        text_cleaner: TextCleaner,
        ecg_processor: ECGProcessor,
        tabular_cleaner: TriageDataCleaner,
        max_length: int = 128,
        use_interaction_features: bool = False,
    ) -> None:
        super().__init__()

        n_samples = len(dataframe)
        assert ecg_array.shape[0] == n_samples, (
            f"Row count mismatch: dataframe has {n_samples} rows but "
            f"ecg_array has {ecg_array.shape[0]} samples"
        )

        # -- Tabular ----------------------------------------------------------
        logger.info("Processing tabular features …")
        tab_df = dataframe[TABULAR_COLS].copy()
        tab_df["temperature"] = tabular_cleaner.clean_temperature(tab_df["temperature"])
        tab_df["pain"] = tabular_cleaner.clean_pain(tab_df["pain"])
        tab_df, _scaler = tabular_cleaner.impute_and_scale(tab_df)
        if use_interaction_features:
            tab_df = tabular_cleaner.generate_interaction_features(tab_df)
        self.tabular = tab_df.values.astype(np.float32)

        # -- Text --------------------------------------------------------------
        logger.info("Cleaning chief-complaint texts …")
        self.cleaned_texts = [
            text_cleaner.clean(str(t)) for t in dataframe[TEXT_COL].values
        ]

        logger.info("Tokenizing %d texts (max_length=%d) …", n_samples, max_length)
        encoding = tokenizer(
            self.cleaned_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )
        self.input_ids = encoding["input_ids"].astype(np.int64)
        self.attention_mask = encoding["attention_mask"].astype(np.int64)

        # -- ECG ---------------------------------------------------------------
        logger.info("Processing ECG signals …")
        self.ecg = ecg_processor.process(ecg_array).astype(np.float32)

        # -- Labels ------------------------------------------------------------
        self.labels = dataframe["label"].values.astype(np.int64)

        logger.info(
            "MultiModalTriageDataset ready: %d samples, tabular shape %s, "
            "ECG shape %s",
            n_samples,
            self.tabular.shape,
            self.ecg.shape,
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "ecg": torch.tensor(self.ecg[idx], dtype=torch.float),
            "tabular": torch.tensor(self.tabular[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Single-modal dataset
# ---------------------------------------------------------------------------


class SingleModalDataset(Dataset):
    """Dataset that exposes only a single modality for ablation baselines.

    Parameters
    ----------
    dataframe : pd.DataFrame
        See :class:`MultiModalTriageDataset`.
    ecg_array : np.ndarray
        Raw ECG signals with shape ``(N, 12, 5000)``.
    tokenizer
        A HuggingFace tokenizer (Bio-ClinicalBERT).
    text_cleaner : TextCleaner
        Instance used to clean chief-complaint strings.
    ecg_processor : ECGProcessor
        Processor that filters and normalises ECG signals.
    tabular_cleaner : TriageDataCleaner
        Cleaner for tabular vital-sign data.
    modality : str
        One of ``'text'``, ``'ecg'``, or ``'tabular'``.
    max_length : int
        Maximum token length for the tokenizer (used only when
        ``modality='text'``).
    use_interaction_features : bool
        If ``True`` and ``modality='tabular'``, append 36 pairwise
        interaction features.
    """

    _VALID_MODALITIES = {"text", "ecg", "tabular"}

    def __init__(
        self,
        dataframe: pd.DataFrame,
        ecg_array: np.ndarray,
        tokenizer: Any,
        text_cleaner: TextCleaner,
        ecg_processor: ECGProcessor,
        tabular_cleaner: TriageDataCleaner,
        modality: str = "text",
        max_length: int = 128,
        use_interaction_features: bool = False,
    ) -> None:
        super().__init__()

        if modality not in self._VALID_MODALITIES:
            raise ValueError(
                f"Invalid modality '{modality}'. "
                f"Must be one of {sorted(self._VALID_MODALITIES)}"
            )

        self.modality = modality
        n_samples = len(dataframe)

        # -- Process only the requested modality ------------------------------
        self.input_ids: np.ndarray | None = None
        self.attention_mask: np.ndarray | None = None
        self.ecg: np.ndarray | None = None
        self.tabular: np.ndarray | None = None

        if modality == "text":
            logger.info("SingleModalDataset: preparing text modality …")
            cleaned_texts = [
                text_cleaner.clean(str(t)) for t in dataframe[TEXT_COL].values
            ]
            encoding = tokenizer(
                cleaned_texts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            self.input_ids = encoding["input_ids"].astype(np.int64)
            self.attention_mask = encoding["attention_mask"].astype(np.int64)

        elif modality == "ecg":
            logger.info("SingleModalDataset: preparing ECG modality …")
            assert ecg_array.shape[0] == n_samples, (
                f"Row count mismatch: dataframe has {n_samples} rows but "
                f"ecg_array has {ecg_array.shape[0]} samples"
            )
            self.ecg = ecg_processor.process(ecg_array).astype(np.float32)

        elif modality == "tabular":
            logger.info("SingleModalDataset: preparing tabular modality …")
            tab_df = dataframe[TABULAR_COLS].copy()
            tab_df["temperature"] = tabular_cleaner.clean_temperature(
                tab_df["temperature"]
            )
            tab_df["pain"] = tabular_cleaner.clean_pain(tab_df["pain"])
            tab_df, _scaler = tabular_cleaner.impute_and_scale(tab_df)
            if use_interaction_features:
                tab_df = tabular_cleaner.generate_interaction_features(tab_df)
            self.tabular = tab_df.values.astype(np.float32)

        # -- Labels ------------------------------------------------------------
        self.labels = dataframe["label"].values.astype(np.int64)

        logger.info(
            "SingleModalDataset ready: %d samples, modality='%s'",
            n_samples,
            modality,
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample: dict[str, torch.Tensor] = {
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

        if self.modality == "text":
            sample["input_ids"] = torch.tensor(
                self.input_ids[idx], dtype=torch.long  # type: ignore[index]
            )
            sample["attention_mask"] = torch.tensor(
                self.attention_mask[idx], dtype=torch.long  # type: ignore[index]
            )
        elif self.modality == "ecg":
            sample["ecg"] = torch.tensor(
                self.ecg[idx], dtype=torch.float  # type: ignore[index]
            )
        elif self.modality == "tabular":
            sample["tabular"] = torch.tensor(
                self.tabular[idx], dtype=torch.float  # type: ignore[index]
            )

        return sample


# ---------------------------------------------------------------------------
# Weighted sampler utility
# ---------------------------------------------------------------------------


def create_weighted_sampler(labels: np.ndarray | list[int]) -> WeightedRandomSampler:
    """Build a :class:`WeightedRandomSampler` with inverse-frequency weights.

    Each class receives a weight equal to ``total_samples / class_count``
    so that under-represented classes are sampled more often.

    Parameters
    ----------
    labels : array-like of int
        1-D array (or list) of integer class labels.

    Returns
    -------
    WeightedRandomSampler
        Sampler with ``replacement=True`` and
        ``num_samples=len(labels)``.
    """
    labels_array = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels_array, return_counts=True)

    class_weights = len(labels_array) / counts.astype(np.float64)
    weight_map = dict(zip(classes, class_weights))

    sample_weights = np.array(
        [weight_map[label] for label in labels_array], dtype=np.float64
    )

    logger.info(
        "WeightedRandomSampler: classes=%s, inverse-freq weights=%s",
        classes.tolist(),
        [round(w, 4) for w in class_weights],
    )

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels_array),
        replacement=True,
    )
