"""B7: XGBoost baseline with handcrafted features.

A non-neural baseline that combines tabular features, TF-IDF text features,
and handcrafted ECG statistics for gradient-boosted tree classification.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier


class XGBoostBaseline:
    """XGBoost classifier over concatenated handcrafted features.

    Feature composition (default):
        * Tabular: 9 pre-processed clinical features
        * Text: TF-IDF (max 500 features, unigrams + bigrams)
        * ECG: per-lead statistics (mean, std, max, min, RMS) × 12 leads = 60

    Total: 569 features (9 + 500 + 60).

    Args:
        config: Dictionary of XGBoost hyperparameters.  Any key accepted by
            ``XGBClassifier`` may be included.  Sensible defaults are provided
            when ``config`` is ``None`` or when specific keys are absent.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.xgb_params: Dict[str, Any] = {
            "objective": "multi:softprob",
            "num_class": 3,
            "n_estimators": config.get("n_estimators", 500),
            "max_depth": config.get("max_depth", 6),
            "learning_rate": config.get("learning_rate", 0.1),
            "subsample": config.get("subsample", 0.8),
            "colsample_bytree": config.get("colsample_bytree", 0.8),
            "min_child_weight": config.get("min_child_weight", 1),
            "eval_metric": config.get("eval_metric", "mlogloss"),
            "random_state": config.get("random_state", 42),
            "n_jobs": config.get("n_jobs", -1),
        }

        self.model: Optional[XGBClassifier] = None
        self.tfidf: TfidfVectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self._tfidf_fitted: bool = False

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _ecg_handcrafted(ecg_data: NDArray[np.floating]) -> NDArray[np.floating]:
        """Extract per-lead summary statistics from raw ECG signals.

        Args:
            ecg_data: Array of shape ``(n_samples, 12, seq_len)``.

        Returns:
            Array of shape ``(n_samples, 60)`` with mean, std, max, min, and
            RMS for each of the 12 leads.
        """
        features = []
        for lead_idx in range(ecg_data.shape[1]):
            lead = ecg_data[:, lead_idx, :]  # (n_samples, seq_len)
            features.append(np.mean(lead, axis=1, keepdims=True))
            features.append(np.std(lead, axis=1, keepdims=True))
            features.append(np.max(lead, axis=1, keepdims=True))
            features.append(np.min(lead, axis=1, keepdims=True))
            features.append(
                np.sqrt(np.mean(lead ** 2, axis=1, keepdims=True)),
            )  # RMS
        return np.concatenate(features, axis=1)  # (n_samples, 60)

    def prepare_features(
        self,
        tabular_features: NDArray[np.floating],
        text_data: list[str],
        ecg_data: NDArray[np.floating],
        *,
        fit_tfidf: bool = False,
    ) -> NDArray[np.floating]:
        """Build the concatenated feature matrix.

        Args:
            tabular_features: Pre-processed tabular array ``(n, 9)``.
            text_data: List of *n* clinical note strings.
            ecg_data: Raw ECG array ``(n, 12, seq_len)``.
            fit_tfidf: If ``True``, fit the TF-IDF vectoriser on
                ``text_data`` before transforming.  Use for training data.

        Returns:
            Feature matrix of shape ``(n, 569)``.
        """
        # Tabular (9 features)
        tab = np.asarray(tabular_features, dtype=np.float32)

        # Text TF-IDF (500 features)
        if fit_tfidf or not self._tfidf_fitted:
            text_feats = self.tfidf.fit_transform(text_data).toarray()
            self._tfidf_fitted = True
        else:
            text_feats = self.tfidf.transform(text_data).toarray()

        # ECG handcrafted (60 features)
        ecg_feats = self._ecg_handcrafted(np.asarray(ecg_data, dtype=np.float32))

        return np.concatenate(
            [tab, text_feats.astype(np.float32), ecg_feats],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Train / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.integer],
    ) -> "XGBoostBaseline":
        """Train the XGBClassifier.

        Args:
            X: Feature matrix of shape ``(n, 569)``.
            y: Integer class labels of shape ``(n,)``.

        Returns:
            ``self`` for method chaining.
        """
        self.model = XGBClassifier(**self.xgb_params)
        self.model.fit(X, y)
        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.integer]:
        """Return predicted class labels.

        Args:
            X: Feature matrix of shape ``(n, 569)``.

        Returns:
            Predicted labels of shape ``(n,)``.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return predicted class probabilities.

        Args:
            X: Feature matrix of shape ``(n, 569)``.

        Returns:
            Probability matrix of shape ``(n, 3)``.
        """
        if self.model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")
        return self.model.predict_proba(X)
