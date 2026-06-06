"""Sklearn ranker fallback for serverless (Vercel) when LightGBM is unavailable."""

from __future__ import annotations

import base64
import logging
import pickle

import numpy as np

logger = logging.getLogger(__name__)


def load_sklearn_ranker():
    try:
        from recommender.sk_model_bundle import MODEL_B64  # auto-generated

        return pickle.loads(base64.b64decode(MODEL_B64))
    except Exception as exc:
        logger.warning("Sklearn ranker unavailable (%s).", exc)
        return None


class SklearnRanker:
    def __init__(self):
        self.model = load_sklearn_ranker()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Sklearn ranker is not loaded")
        return self.model.predict(X)
