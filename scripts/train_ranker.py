#!/usr/bin/env python3
"""Train LightGBM ranker from seed data + event logs. Run before deploy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.engine import MedXRecommender  # noqa: E402
from recommender.features import LogStats, build_feature_row  # noqa: E402
from recommender.ranker import DEFAULT_MODEL_PATH, train_ranker  # noqa: E402


def build_training_matrix(rec: MedXRecommender, alpha: float = 0.5):
    stats = LogStats.from_frames(
        rec.doctors_df, rec.articles_df, rec.interactions_df, rec.event_logs_df
    )
    rows: list[list[float]] = []
    labels: list[int] = []
    groups: list[int] = []

    for doctor_id in rec.doctors_df["id"]:
        doctor = rec.doctors_df[rec.doctors_df["id"] == doctor_id].iloc[0]
        doctor_logs = rec.event_logs_df[rec.event_logs_df["doctor_id"] == doctor_id]
        if not doctor_logs.empty:
            hour = int(doctor_logs["hour"].median())
        else:
            hour = 12

        content = rec._content_scores_for_doctor(doctor_id)
        collab = rec._collab_scores_for_doctor(doctor_id)
        group_rows: list[list[float]] = []
        group_labels: list[int] = []

        for _, article in rec.articles_df.iterrows():
            aid = article["id"]
            group_rows.append(
                build_feature_row(
                    doctor,
                    article,
                    hour,
                    float(content[aid]),
                    float(collab[aid]),
                    alpha,
                    stats,
                )
            )
            group_labels.append(stats.relevance_label(doctor_id, aid))

        rows.extend(group_rows)
        labels.extend(group_labels)
        groups.append(len(group_rows))

    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.int32), groups


def main():
    rec = MedXRecommender()
    X, y, groups = build_training_matrix(rec)
    print(f"Training LightGBM on {X.shape[0]} rows, {len(groups)} doctor groups …")
    train_ranker(X, y, groups, DEFAULT_MODEL_PATH)
    print(f"Saved model → {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
