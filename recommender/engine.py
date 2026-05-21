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

from data.seed_data import ARTICLES, DOCTORS, INTERACTIONS

# ---------------------------------------------------------------------------
# Time-context definitions
# ---------------------------------------------------------------------------
# Each slot defines the ideal complexity and max reading time a doctor is
# likely to tolerate given their available time and mental energy.
TIME_SLOTS = [
    {"label": "Early Morning",  "icon": "🌅", "hours": (5, 9),   "ideal_complexity": 0.8, "max_reading_min": 20},
    {"label": "Morning Work",   "icon": "💼", "hours": (9, 12),  "ideal_complexity": 0.6, "max_reading_min": 10},
    {"label": "Lunch Break",    "icon": "🍽️", "hours": (12, 14), "ideal_complexity": 0.3, "max_reading_min": 5},
    {"label": "Afternoon Work", "icon": "📋", "hours": (14, 18), "ideal_complexity": 0.55, "max_reading_min": 9},
    {"label": "Evening",        "icon": "🌆", "hours": (18, 22), "ideal_complexity": 0.8, "max_reading_min": 20},
    {"label": "Late Night",     "icon": "🌙", "hours": (22, 24), "ideal_complexity": 0.4, "max_reading_min": 6},
    {"label": "Night",          "icon": "🌙", "hours": (0, 5),   "ideal_complexity": 0.4, "max_reading_min": 6},
]


def get_time_slot(hour: int) -> dict:
    for slot in TIME_SLOTS:
        lo, hi = slot["hours"]
        if lo <= hour < hi:
            return slot
    return TIME_SLOTS[3]


def context_boost(article: dict, slot: dict, lam: float = 0.3) -> float:
    """
    Returns a multiplier in [1-lam, 1+lam] based on how well the article's
    complexity and reading time fit the current time slot.
    """
    complexity_fit = 1.0 - abs(article["complexity_score"] - slot["ideal_complexity"])
    time_fit = 1.0 if article["reading_time_minutes"] <= slot["max_reading_min"] else \
               max(0.0, 1.0 - (article["reading_time_minutes"] - slot["max_reading_min"]) / 20.0)
    fit = (complexity_fit + time_fit) / 2.0           # 0–1
    return 1.0 + lam * (2.0 * fit - 1.0)             # [1-lam, 1+lam]


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

    # ------------------------------------------------------------------
    # Hybrid recommendation
    # ------------------------------------------------------------------
    def recommend(
        self,
        doctor_id: str,
        n: int = 6,
        alpha: float = 0.5,
        exclude_read: bool = True,
        hour: int | None = None,
    ) -> list[dict]:
        content_scores = self._content_scores_for_doctor(doctor_id)
        collab_scores  = self._collab_scores_for_doctor(doctor_id)

        def norm(s: pd.Series) -> pd.Series:
            rng = s.max() - s.min()
            return (s - s.min()) / rng if rng > 0 else s

        combined = alpha * norm(content_scores) + (1 - alpha) * norm(collab_scores)

        if exclude_read:
            read_ids = set(
                self.interactions_df[self.interactions_df["doctor_id"] == doctor_id][
                    "article_id"
                ]
            )
            combined = combined.drop(index=list(read_ids & set(combined.index)), errors="ignore")

        # Apply time-context re-ranking if hour is provided
        slot = get_time_slot(hour) if hour is not None else None
        if slot is not None:
            for aid in combined.index:
                art_row = self.articles_df[self.articles_df["id"] == aid]
                if not art_row.empty:
                    art = art_row.iloc[0].to_dict()
                    combined[aid] *= context_boost(art, slot)

        top_ids = combined.nlargest(n).index.tolist()
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
                "score":               round(float(combined[aid]), 4),
                "complexity_score":    row["complexity_score"],
                "reading_time_minutes": int(row["reading_time_minutes"]),
            }
            if slot:
                entry["context_label"] = slot["label"]
                entry["context_icon"]  = slot["icon"]
            results.append(entry)
        return results

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
