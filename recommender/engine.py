"""
Hybrid recommender engine combining:
  1. Content-based filtering  — TF-IDF on article tags + doctor specialty (scikit-learn)
  2. Collaborative filtering  — SVD matrix factorization (pure numpy, no extra deps)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from data.seed_data import ARTICLES, DOCTORS, EVENT_LOGS, INTERACTIONS
from recommender.context import context_boost, get_time_slot
from recommender.features import LogStats, build_feature_row
from recommender.ranker import LightGBMRanker
from recommender.sklearn_ranker import SklearnRanker

# Ensure Vercel bundles auto-generated ranker modules (dynamic imports are omitted).
try:
    from recommender import model_bundle as _model_bundle  # noqa: F401
    from recommender import sk_model_bundle as _sk_model_bundle  # noqa: F401
except ImportError:
    pass

# Re-export for API consumers
__all__ = ["MedXRecommender", "get_time_slot", "TIME_SLOTS"]

from recommender.context import TIME_SLOTS  # noqa: E402
# ---------------------------------------------------------------------------
# Minimal SVD collaborative filter (numpy only — no scikit-surprise needed)
# ---------------------------------------------------------------------------
class _SVDCollab:
    """
    Mean-centered SVD collaborative filter.
    Predicts a rating for any (user, item) pair by decomposing the
    user-item interaction matrix: R ≈ U · Σ · Vᵀ
    """

    def fit(self, user_ids: list, item_ids: list, ratings: list, n_factors: int = 15):
        self.users = sorted(set(user_ids))
        self.items = sorted(set(item_ids))
        self.user_idx = {u: i for i, u in enumerate(self.users)}
        self.item_idx = {it: i for i, it in enumerate(self.items)}

        R = np.zeros((len(self.users), len(self.items)))
        for u, it, r in zip(user_ids, item_ids, ratings):
            R[self.user_idx[u], self.item_idx[it]] = r

        self.global_mean = float(np.mean([r for r in ratings]))

        # Per-user bias and mean-centered matrix
        self.user_bias: dict[str, float] = {}
        R_centered = R.copy()
        for u in self.users:
            ui = self.user_idx[u]
            rated_vals = R[ui, R[ui] > 0]
            bias = float(rated_vals.mean()) - self.global_mean if len(rated_vals) > 0 else 0.0
            self.user_bias[u] = bias
            R_centered[ui, R[ui] > 0] -= (self.global_mean + bias)

        # Truncated SVD
        k = min(n_factors, min(R.shape) - 1)
        U, sigma, Vt = np.linalg.svd(R_centered, full_matrices=False)
        self.U = U[:, :k]
        self.sigma = sigma[:k]
        self.Vt = Vt[:k, :]

    def predict(self, user_id: str, item_id: str) -> float:
        if user_id not in self.user_idx or item_id not in self.item_idx:
            return self.global_mean
        ui = self.user_idx[user_id]
        ii = self.item_idx[item_id]
        pred = (
            self.global_mean
            + self.user_bias.get(user_id, 0.0)
            + float(self.U[ui] @ np.diag(self.sigma) @ self.Vt[:, ii])
        )
        return float(np.clip(pred, 1.0, 5.0))


# ---------------------------------------------------------------------------
# Main recommender
# ---------------------------------------------------------------------------
class MedXRecommender:
    def __init__(self):
        self.articles_df = pd.DataFrame(ARTICLES)
        self.doctors_df  = pd.DataFrame(DOCTORS)
        self.interactions_df = pd.DataFrame(
            INTERACTIONS, columns=["doctor_id", "article_id", "rating"]
        )
        self.event_logs_df = pd.DataFrame(EVENT_LOGS)
        self.log_stats = LogStats.from_frames(
            self.doctors_df,
            self.articles_df,
            self.interactions_df,
            self.event_logs_df,
        )
        self.ranker = LightGBMRanker()
        self.sk_ranker = SklearnRanker()
        self._build_content_model()
        self._build_collab_model()

    # ------------------------------------------------------------------
    # Content-based model
    # ------------------------------------------------------------------
    def _build_content_model(self):
        corpus = self.articles_df.apply(
            lambda r: " ".join(r["tags"]) + " " + r["specialty"] + " " + r["type"],
            axis=1,
        )
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)
        self.art_idx = {aid: i for i, aid in enumerate(self.articles_df["id"])}

    def _content_scores_for_doctor(self, doctor_id: str) -> pd.Series:
        doctor   = self.doctors_df[self.doctors_df["id"] == doctor_id].iloc[0]
        read_ids = self.interactions_df[
            self.interactions_df["doctor_id"] == doctor_id
        ]["article_id"].tolist()

        profile_vectors = [self.tfidf.transform([doctor["specialty"]]).toarray()]
        for aid in read_ids:
            if aid in self.art_idx:
                profile_vectors.append(self.tfidf_matrix[self.art_idx[aid]].toarray())

        profile = np.mean(profile_vectors, axis=0)
        sims = cosine_similarity(profile, self.tfidf_matrix)[0]
        return pd.Series(sims, index=self.articles_df["id"])

    # ------------------------------------------------------------------
    # Collaborative filtering model
    # ------------------------------------------------------------------
    def _build_collab_model(self):
        self.svd = _SVDCollab()
        self.svd.fit(
            self.interactions_df["doctor_id"].tolist(),
            self.interactions_df["article_id"].tolist(),
            self.interactions_df["rating"].tolist(),
            n_factors=10,
        )
        self.all_article_ids = self.articles_df["id"].tolist()

    def _collab_scores_for_doctor(self, doctor_id: str) -> pd.Series:
        scores = {aid: self.svd.predict(doctor_id, aid) for aid in self.all_article_ids}
        return pd.Series(scores)

    def _norm_scores(self, s: pd.Series) -> pd.Series:
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s

    def _hybrid_scores(
        self, doctor_id: str, alpha: float, hour: int | None
    ) -> tuple[pd.Series, str | None]:
        content_scores = self._content_scores_for_doctor(doctor_id)
        collab_scores = self._collab_scores_for_doctor(doctor_id)
        combined = alpha * self._norm_scores(content_scores) + (1 - alpha) * self._norm_scores(
            collab_scores
        )
        slot = get_time_slot(hour) if hour is not None else None
        if slot is not None:
            for aid in combined.index:
                art_row = self.articles_df[self.articles_df["id"] == aid]
                if not art_row.empty:
                    art = art_row.iloc[0].to_dict()
                    combined[aid] *= context_boost(art, slot)
        return combined, slot["label"] if slot else None

    def _lgb_rank(
        self,
        doctor_id: str,
        candidate_ids: list[str],
        alpha: float,
        hour: int,
    ) -> pd.Series:
        doctor = self.doctors_df[self.doctors_df["id"] == doctor_id].iloc[0]
        content = self._content_scores_for_doctor(doctor_id)
        collab = self._collab_scores_for_doctor(doctor_id)
        rows = []
        for aid in candidate_ids:
            article = self.articles_df[self.articles_df["id"] == aid].iloc[0]
            rows.append(
                build_feature_row(
                    doctor,
                    article,
                    hour,
                    float(content[aid]),
                    float(collab[aid]),
                    alpha,
                    self.log_stats,
                )
            )
        scores = self.ranker.predict_scores(np.array(rows, dtype=np.float32))
        return pd.Series(scores, index=candidate_ids)

    def _rank_scores(
        self,
        doctor_id: str,
        candidate_ids: list[str],
        alpha: float,
        hour: int,
    ) -> tuple[pd.Series, str]:
        if self.ranker.is_loaded:
            ranked = self._lgb_rank(doctor_id, candidate_ids, alpha, hour)
            return ranked, "lightgbm"
        if self.sk_ranker.is_loaded:
            doctor = self.doctors_df[self.doctors_df["id"] == doctor_id].iloc[0]
            content = self._content_scores_for_doctor(doctor_id)
            collab = self._collab_scores_for_doctor(doctor_id)
            rows = []
            for aid in candidate_ids:
                article = self.articles_df[self.articles_df["id"] == aid].iloc[0]
                rows.append(
                    build_feature_row(
                        doctor, article, hour,
                        float(content[aid]), float(collab[aid]), alpha, self.log_stats,
                    )
                )
            scores = self.sk_ranker.predict_scores(np.array(rows, dtype=np.float32))
            return pd.Series(scores, index=candidate_ids), "sklearn"
        combined, _ = self._hybrid_scores(doctor_id, alpha, hour)
        combined = combined.reindex(candidate_ids).dropna()
        return combined, "hybrid"

    @property
    def ranker_loaded(self) -> bool:
        return self.ranker.is_loaded or self.sk_ranker.is_loaded

    # ------------------------------------------------------------------
    # Hybrid recommendation
    # ------------------------------------------------------------------
    def recommend(
        self,
        doctor_id: str,
        n: int = 5,
        alpha: float = 0.5,
        exclude_read: bool = True,
        hour: int | None = None,
        use_ranker: bool = True,
    ) -> tuple[list[dict], str]:
        n = min(max(n, 1), 5)
        slot = get_time_slot(hour) if hour is not None else None
        ranker_mode = "hybrid"

        candidate_ids = self.all_article_ids.copy()
        if exclude_read:
            read_ids = set(
                self.interactions_df[self.interactions_df["doctor_id"] == doctor_id][
                    "article_id"
                ]
            )
            candidate_ids = [aid for aid in candidate_ids if aid not in read_ids]

        if use_ranker and hour is not None and candidate_ids:
            ranked, ranker_mode = self._rank_scores(doctor_id, candidate_ids, alpha, hour)
        else:
            combined, _ = self._hybrid_scores(doctor_id, alpha, hour)
            combined = combined.reindex(candidate_ids).dropna()
            ranked = combined
            ranker_mode = "hybrid"

        top_ids = ranked.nlargest(n).index.tolist()
        results = []
        for aid in top_ids:
            row = self.articles_df[self.articles_df["id"] == aid].iloc[0]
            entry = {
                "id":                  aid,
                "title":               row["title"],
                "specialty":           row["specialty"],
                "type":                row["type"],
                "tags":                row["tags"],
                "summary":             row["summary"],
                "score":               round(float(ranked[aid]), 4),
                "complexity_score":    row["complexity_score"],
                "reading_time_minutes": int(row["reading_time_minutes"]),
            }
            if slot:
                entry["context_label"] = slot["label"]
                entry["context_icon"]  = slot["icon"]
            results.append(entry)
        return results, ranker_mode

    def similar_articles(self, article_id: str, n: int = 4) -> list[dict]:
        if article_id not in self.art_idx:
            return []
        idx  = self.art_idx[article_id]
        sims = cosine_similarity(self.tfidf_matrix[idx], self.tfidf_matrix)[0]
        sims_series = pd.Series(sims, index=self.articles_df["id"]).drop(
            index=article_id, errors="ignore"
        )
        results = []
        for aid in sims_series.nlargest(n).index:
            row = self.articles_df[self.articles_df["id"] == aid].iloc[0]
            results.append({
                "id":                  aid,
                "title":               row["title"],
                "specialty":           row["specialty"],
                "type":                row["type"],
                "tags":                row["tags"],
                "summary":             row["summary"],
                "similarity":          round(float(sims_series[aid]), 4),
                "complexity_score":    row["complexity_score"],
                "reading_time_minutes": int(row["reading_time_minutes"]),
            })
        return results

    def get_doctor(self, doctor_id: str) -> dict | None:
        row = self.doctors_df[self.doctors_df["id"] == doctor_id]
        if row.empty:
            return None
        r = row.iloc[0]
        read_ids = self.interactions_df[
            self.interactions_df["doctor_id"] == doctor_id
        ]["article_id"].tolist()
        return {
            "id":           r["id"],
            "name":         r["name"],
            "specialty":    r["specialty"],
            "years_exp":    int(r["years_exp"]),
            "articles_read": read_ids,
        }

    def get_all_doctors(self)  -> list[dict]: return self.doctors_df.to_dict(orient="records")
    def get_all_articles(self) -> list[dict]: return self.articles_df.to_dict(orient="records")

    def get_article(self, article_id: str) -> dict | None:
        row = self.articles_df[self.articles_df["id"] == article_id]
        return row.iloc[0].to_dict() if not row.empty else None
