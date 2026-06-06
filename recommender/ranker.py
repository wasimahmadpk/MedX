"""LightGBM learning-to-rank model (train offline, load at runtime for Vercel)."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np

from recommender.features import FEATURE_NAMES

MODEL_DIR = Path(__file__).parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "lgb_ranker.txt"


def train_ranker(
    X: np.ndarray,
    y: np.ndarray,
    group: list[int],
    model_path: Path = DEFAULT_MODEL_PATH,
) -> lgb.Booster:
    """Train LambdaRank model and persist to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train_set = lgb.Dataset(X, label=y, group=group, feature_name=FEATURE_NAMES)
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_at": [5],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "seed": 42,
    }
    booster = lgb.train(params, train_set, num_boost_round=80)
    booster.save_model(str(model_path))
    return booster


def load_ranker(model_path: Path = DEFAULT_MODEL_PATH) -> lgb.Booster | None:
    if not model_path.is_file():
        return None
    return lgb.Booster(model_file=str(model_path))


class LightGBMRanker:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.model = load_ranker(model_path)

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LightGBM ranker model is not loaded")
        return self.model.predict(X)
