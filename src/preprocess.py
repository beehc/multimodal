"""Preprocessing module for the Multi-Modal Emergency Triage System.

Provides cleaning, normalization, and feature-engineering utilities for
tabular vital signs, free-text chief complaints, and 12-lead ECG signals.
"""

from __future__ import annotations

import logging
import re
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tabular vital-sign cleaning
# ---------------------------------------------------------------------------

class TriageDataCleaner:
    """Clean and transform the 9-feature tabular vital-sign data."""

    @staticmethod
    def clean_temperature(series: pd.Series) -> pd.Series:
        """Normalise temperature values to Fahrenheit.

        * Values in [25.0, 45.0] are assumed Celsius and converted to °F.
        * Values below 25.0 or above 115.0 are treated as outliers (→ NaN).
        * All other values are kept as-is (assumed already in °F).

        Parameters
        ----------
        series : pd.Series
            Raw temperature column (mixed °C / °F / outliers).

        Returns
        -------
        pd.Series
            Cleaned temperature column in Fahrenheit.
        """
        series = series.copy().astype(float)
        celsius_mask = (series >= 25.0) & (series <= 45.0)
        outlier_mask = (series < 25.0) | (series > 115.0)

        series[celsius_mask] = series[celsius_mask] * 1.8 + 32
        series[outlier_mask] = np.nan

        n_converted = celsius_mask.sum()
        n_outliers = outlier_mask.sum()
        logger.info(
            "Temperature cleaning: %d converted C→F, %d outliers removed",
            n_converted,
            n_outliers,
        )
        return series

    @staticmethod
    def clean_pain(series: pd.Series) -> pd.Series:
        """Coerce non-numeric pain-score entries to NaN.

        Parameters
        ----------
        series : pd.Series
            Raw pain-score column (may contain strings such as
            ``"unable"``, ``"ok"``, ``"uta"``, ``"leg pain"``).

        Returns
        -------
        pd.Series
            Numeric pain scores with non-parseable values set to NaN.
        """
        cleaned = pd.to_numeric(series, errors="coerce")
        n_coerced = series.notna().sum() - cleaned.notna().sum()
        logger.info("Pain cleaning: %d non-numeric entries coerced to NaN", n_coerced)
        return cleaned

    @staticmethod
    def impute_and_scale(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
        """Median-impute missing values then Z-score standardise.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the 9 numeric vital-sign features.  Missing
            values are filled with column medians before standardisation.

        Returns
        -------
        tuple[pd.DataFrame, StandardScaler]
            A tuple of (scaled DataFrame, fitted ``StandardScaler``).
        """
        n_missing = df.isna().sum().sum()
        logger.info("Imputing %d missing values with column medians", n_missing)
        df_imputed = df.fillna(df.median())

        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df_imputed)
        df_scaled = pd.DataFrame(
            scaled_values, columns=df_imputed.columns, index=df_imputed.index
        )
        logger.info(
            "Z-score standardisation applied to %d features", df_scaled.shape[1]
        )
        return df_scaled, scaler

    @staticmethod
    def generate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create pairwise interaction features from the 9 original columns.

        For every unordered pair of distinct columns (C(9,2) = 36 pairs) the
        element-wise product is appended, yielding 9 + 36 = 45 features.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with the 9 original (scaled) features.

        Returns
        -------
        pd.DataFrame
            DataFrame with 45 columns (9 originals + 36 interactions).
        """
        interaction_frames: list[pd.Series] = []
        cols = list(df.columns)
        for col_a, col_b in combinations(cols, 2):
            name = f"{col_a}_x_{col_b}"
            interaction_frames.append(
                pd.Series(df[col_a].values * df[col_b].values, name=name, index=df.index)
            )

        df_interactions = pd.concat([df] + interaction_frames, axis=1)
        logger.info(
            "Generated %d interaction features → %d total columns",
            len(interaction_frames),
            df_interactions.shape[1],
        )
        return df_interactions


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

class TextCleaner:
    """Clean free-text chief-complaint strings."""

    ABBREVIATIONS: dict[str, str] = {
        "sob": "shortness of breath",
        "abd": "abdominal",
        "cp": "chest pain",
        "s/p": "status post",
        "n/v/d": "nausea vomiting diarrhea",
        "n/v": "nausea vomiting",
        "pna": "pneumonia",
        "loc": "loss of consciousness",
        "htn": "hypertension",
        "dm": "diabetes mellitus",
        "cva": "cerebrovascular accident",
        "mi": "myocardial infarction",
        "chf": "congestive heart failure",
        "etoh": "alcohol",
        "ams": "altered mental status",
        "fx": "fracture",
        "ha": "headache",
        "r/o": "rule out",
        "w/": "with",
        "pt": "patient",
    }

    # Pre-compile patterns sorted by length (longest first) so that
    # multi-character abbreviations like ``n/v/d`` match before ``n/v``.
    _sorted_abbrevs: list[tuple[re.Pattern[str], str]] = []

    def __init__(self) -> None:
        sorted_keys = sorted(self.ABBREVIATIONS, key=len, reverse=True)
        self._sorted_abbrevs = []
        for abbr in sorted_keys:
            escaped = re.escape(abbr)
            # Use word boundaries; for abbreviations ending with '/'
            # the trailing boundary is already a non-word char.
            if abbr.endswith("/"):
                pattern = re.compile(rf"\b{escaped}", re.IGNORECASE)
            else:
                pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            self._sorted_abbrevs.append((pattern, self.ABBREVIATIONS[abbr]))

    @staticmethod
    def remove_noise(text: str) -> str:
        """Strip noise from clinical free-text.

        Steps applied in order:
        1. Replace underscores with spaces.
        2. Collapse multiple whitespace characters into a single space.
        3. Remove special characters except alphanumeric and basic punctuation.
        4. Convert to lowercase.

        Parameters
        ----------
        text : str
            Raw chief-complaint string.

        Returns
        -------
        str
            Cleaned text.
        """
        text = text.replace("_", " ")
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,;:!?'\-/]", "", text)
        text = text.lower().strip()
        return text

    def expand_abbreviations(self, text: str) -> str:
        """Replace medical abbreviations with full terms.

        Matching is case-insensitive and uses word boundaries so that
        partial words are not affected.

        Parameters
        ----------
        text : str
            Input text (should already be lowercased for best results).

        Returns
        -------
        str
            Text with abbreviations expanded.
        """
        for pattern, expansion in self._sorted_abbrevs:
            text = pattern.sub(expansion, text)
        return text

    def clean(self, text: str) -> str:
        """Full cleaning pipeline: noise removal then abbreviation expansion.

        Parameters
        ----------
        text : str
            Raw chief-complaint string.

        Returns
        -------
        str
            Fully cleaned text.
        """
        text = self.remove_noise(text)
        text = self.expand_abbreviations(text)
        return text


# ---------------------------------------------------------------------------
# ECG signal processing
# ---------------------------------------------------------------------------

class ECGProcessor:
    """Bandpass filter and normalise 12-lead ECG signals.

    Designed for signals sampled at 500 Hz with shape
    ``(num_samples, 12, 5000)``.
    """

    SAMPLING_RATE: int = 500  # Hz

    @classmethod
    def apply_filters(cls, ecg_array: np.ndarray) -> np.ndarray:
        """Apply high-pass and low-pass Butterworth filters.

        * High-pass at 0.5 Hz (order 4) to remove baseline wander.
        * Low-pass at 50 Hz (order 4) to remove power-line / EMG noise.
        * Zero-phase filtering via ``scipy.signal.filtfilt``.

        Parameters
        ----------
        ecg_array : np.ndarray
            ECG data with shape ``(N, 12, 5000)`` or ``(12, 5000)``.

        Returns
        -------
        np.ndarray
            Filtered ECG array (same shape as input).
        """
        nyquist = cls.SAMPLING_RATE / 2.0

        b_high, a_high = butter(4, 0.5 / nyquist, btype="high")
        b_low, a_low = butter(4, 50.0 / nyquist, btype="low")

        filtered = ecg_array.copy().astype(np.float64)

        if filtered.ndim == 2:
            # Single sample: (12, 5000)
            for lead in range(filtered.shape[0]):
                filtered[lead] = filtfilt(b_high, a_high, filtered[lead])
                filtered[lead] = filtfilt(b_low, a_low, filtered[lead])
        elif filtered.ndim == 3:
            # Batch: (N, 12, 5000)
            for i in range(filtered.shape[0]):
                for lead in range(filtered.shape[1]):
                    filtered[i, lead] = filtfilt(b_high, a_high, filtered[i, lead])
                    filtered[i, lead] = filtfilt(b_low, a_low, filtered[i, lead])
        else:
            raise ValueError(
                f"Expected 2-D or 3-D ECG array, got shape {ecg_array.shape}"
            )

        logger.info("Applied Butterworth filters to ECG array of shape %s", ecg_array.shape)
        return filtered

    @staticmethod
    def normalize(ecg_array: np.ndarray) -> np.ndarray:
        """Per-lead Z-score normalisation.

        Each lead is independently standardised to zero mean and unit
        variance.

        Parameters
        ----------
        ecg_array : np.ndarray
            ECG data with shape ``(N, 12, 5000)`` or ``(12, 5000)``.

        Returns
        -------
        np.ndarray
            Normalised ECG array (same shape).
        """
        arr = ecg_array.copy().astype(np.float64)

        if arr.ndim == 2:
            mean = arr.mean(axis=1, keepdims=True)
            std = arr.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            arr = (arr - mean) / std
        elif arr.ndim == 3:
            mean = arr.mean(axis=2, keepdims=True)
            std = arr.std(axis=2, keepdims=True)
            std[std == 0] = 1.0
            arr = (arr - mean) / std
        else:
            raise ValueError(
                f"Expected 2-D or 3-D ECG array, got shape {ecg_array.shape}"
            )

        logger.info("Per-lead Z-score normalisation applied to shape %s", ecg_array.shape)
        return arr

    @classmethod
    def process(cls, ecg_array: np.ndarray) -> np.ndarray:
        """Full ECG processing pipeline: filter then normalise.

        Parameters
        ----------
        ecg_array : np.ndarray
            Raw ECG data.

        Returns
        -------
        np.ndarray
            Filtered and normalised ECG data.
        """
        filtered = cls.apply_filters(ecg_array)
        normalized = cls.normalize(filtered)
        return normalized

    @staticmethod
    def extract_handcrafted_features(ecg_array: np.ndarray) -> np.ndarray:
        """Extract statistical features per lead for the B7 XGBoost baseline.

        For each of the 12 leads the following statistics are computed:
        mean, std, max, min, RMS — giving 12 × 5 = 60 features per sample.

        Parameters
        ----------
        ecg_array : np.ndarray
            ECG data with shape ``(N, 12, 5000)`` or ``(12, 5000)``.

        Returns
        -------
        np.ndarray
            Feature matrix with shape ``(N, 60)`` (batch) or ``(60,)``
            (single sample).
        """
        single = ecg_array.ndim == 2
        if single:
            ecg_array = ecg_array[np.newaxis, ...]  # (1, 12, 5000)

        n_samples = ecg_array.shape[0]
        n_leads = ecg_array.shape[1]
        features = np.empty((n_samples, n_leads * 5), dtype=np.float64)

        for i in range(n_samples):
            for lead in range(n_leads):
                signal = ecg_array[i, lead].astype(np.float64)
                offset = lead * 5
                features[i, offset] = signal.mean()
                features[i, offset + 1] = signal.std()
                features[i, offset + 2] = signal.max()
                features[i, offset + 3] = signal.min()
                features[i, offset + 4] = np.sqrt(np.mean(signal ** 2))

        logger.info(
            "Extracted %d handcrafted ECG features for %d samples",
            features.shape[1],
            n_samples,
        )
        return features[0] if single else features


# ---------------------------------------------------------------------------
# Utility – dataset filtering & alignment
# ---------------------------------------------------------------------------

def filter_and_align(
    csv_path: str,
    ecg_path: str,
    target_classes: list[int],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Filter the CSV by acuity and align the ECG array accordingly.

    Steps:
    1. Load the CSV and retain only rows whose ``acuity`` is in
       *target_classes*.
    2. Record the original row indices of the retained samples.
    3. Load the ECG ``.npy`` file and slice it using those indices.
    4. Remap acuity labels (1 → 0, 2 → 1, 3 → 2).
    5. Assert that the number of CSV rows matches the number of ECG
       samples.

    Parameters
    ----------
    csv_path : str
        Path to the raw CSV file (must contain an ``acuity`` column).
    ecg_path : str
        Path to the ``.npy`` ECG array with shape ``(N, 12, 5000)``.
    target_classes : list[int]
        Acuity values to keep (e.g. ``[1, 2, 3]``).

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        ``(filtered_df, filtered_ecg)`` with aligned row/sample order
        and remapped labels.
    """
    logger.info("Loading CSV from %s", csv_path)
    df = pd.read_csv(csv_path)
    original_len = len(df)

    mask = df["acuity"].isin(target_classes)
    indices = df.index[mask].tolist()
    df_filtered = df.loc[mask].reset_index(drop=True)
    logger.info(
        "Filtered CSV: %d → %d rows (acuity in %s)",
        original_len,
        len(df_filtered),
        target_classes,
    )

    logger.info("Loading ECG array from %s", ecg_path)
    ecg_all = np.load(ecg_path)
    ecg_filtered = ecg_all[indices]

    assert len(df_filtered) == ecg_filtered.shape[0], (
        f"Row count mismatch after filtering: "
        f"CSV has {len(df_filtered)} rows but ECG has {ecg_filtered.shape[0]} samples"
    )
    logger.info("ECG aligned: %d samples selected", ecg_filtered.shape[0])

    label_map = {cls: idx for idx, cls in enumerate(sorted(target_classes))}
    df_filtered["acuity"] = df_filtered["acuity"].map(label_map)
    logger.info("Labels remapped: %s", label_map)

    return df_filtered, ecg_filtered
