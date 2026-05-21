"""
Hybrid recommender engine combining:
  1. Content-based filtering  — TF-IDF on article tags + doctor specialty
  2. Collaborative filtering  — SVD matrix factorization via Surprise
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

from data.seed_data import ARTICLES, DOCTORS, INTERACTIONS


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
        """TF-IDF on combined article text (tags + specialty + type)."""
        corpus = self.articles_df.apply(
            lambda r: " ".join(r["tags"]) + " " + r["specialty"] + " " + r["type"],
            axis=1,
        )
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2))
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)

        # Map article id → matrix row index
        self.art_idx = {aid: i for i, aid in enumerate(self.articles_df["id"])}

    def _content_scores_for_doctor(self, doctor_id: str, n: int = 10) -> pd.Series:
        """Score all articles based on doctor's specialty and reading history."""
        doctor = self.doctors_df[self.doctors_df["id"] == doctor_id].iloc[0]
        specialty = doctor["specialty"]

        # Build doctor profile vector = mean of articles read + specialty vector
        read_ids = self.interactions_df[
            self.interactions_df["doctor_id"] == doctor_id
        ]["article_id"].tolist()

        profile_vectors = []

        # Include specialty as a pseudo-document
        specialty_vec = self.tfidf.transform([specialty])
        profile_vectors.append(specialty_vec.toarray())

        for aid in read_ids:
            if aid in self.art_idx:
                profile_vectors.append(
                    self.tfidf_matrix[self.art_idx[aid]].toarray()
                )

        profile = np.mean(profile_vectors, axis=0)
        sims = cosine_similarity(profile, self.tfidf_matrix)[0]
        return pd.Series(sims, index=self.articles_df["id"])

    # ------------------------------------------------------------------
    # Collaborative filtering model
    # ------------------------------------------------------------------
    def _build_collab_model(self):
        """Train SVD on the interactions matrix."""
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.interactions_df, reader)

        # Use full dataset for training (small data — no held-out split needed for prototype)
        trainset = data.build_full_trainset()
        self.svd = SVD(n_factors=20, n_epochs=30, lr_all=0.005, reg_all=0.02)
        self.svd.fit(trainset)

        self.all_article_ids = self.articles_df["id"].tolist()

    def _collab_scores_for_doctor(self, doctor_id: str) -> pd.Series:
        """Predict ratings for all articles not yet read by this doctor."""
        read_ids = set(
            self.interactions_df[self.interactions_df["doctor_id"] == doctor_id][
                "article_id"
            ]
        )
        scores = {}
        for aid in self.all_article_ids:
            pred = self.svd.predict(doctor_id, aid)
            scores[aid] = pred.est
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
    ) -> list[dict]:
        """
        Return top-n recommendations for a doctor.

        alpha: weight for content score (1-alpha goes to collab score)
        """
        content_scores = self._content_scores_for_doctor(doctor_id)
        collab_scores  = self._collab_scores_for_doctor(doctor_id)

        # Normalize both to [0, 1]
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

        top_ids = combined.nlargest(n).index.tolist()

        results = []
        for aid in top_ids:
            row = self.articles_df[self.articles_df["id"] == aid].iloc[0]
            results.append({
                "id":       aid,
                "title":    row["title"],
                "specialty": row["specialty"],
                "type":     row["type"],
                "tags":     row["tags"],
                "summary":  row["summary"],
                "score":    round(float(combined[aid]), 4),
            })
        return results

    def similar_articles(self, article_id: str, n: int = 4) -> list[dict]:
        """Content-based similar articles."""
        if article_id not in self.art_idx:
            return []

        idx = self.art_idx[article_id]
        sims = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        )[0]
        sims_series = pd.Series(sims, index=self.articles_df["id"])
        sims_series = sims_series.drop(index=article_id, errors="ignore")
        top_ids = sims_series.nlargest(n).index.tolist()

        results = []
        for aid in top_ids:
            row = self.articles_df[self.articles_df["id"] == aid].iloc[0]
            results.append({
                "id":       aid,
                "title":    row["title"],
                "specialty": row["specialty"],
                "type":     row["type"],
                "tags":     row["tags"],
                "summary":  row["summary"],
                "similarity": round(float(sims_series[aid]), 4),
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
            "id":        r["id"],
            "name":      r["name"],
            "specialty": r["specialty"],
            "years_exp": int(r["years_exp"]),
            "articles_read": read_ids,
        }

    def get_all_doctors(self) -> list[dict]:
        return self.doctors_df.to_dict(orient="records")

    def get_all_articles(self) -> list[dict]:
        return self.articles_df.to_dict(orient="records")

    def get_article(self, article_id: str) -> dict | None:
        row = self.articles_df[self.articles_df["id"] == article_id]
        return row.iloc[0].to_dict() if not row.empty else None
