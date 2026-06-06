"""Feature engineering for LightGBM learning-to-rank."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recommender.context import context_boost, get_time_slot

FEATURE_NAMES = [
    "content_score",
    "collab_score",
    "alpha_blend",
    "specialty_match",
    "complexity_score",
    "reading_time_min",
    "context_boost",
    "request_hour_norm",
    "article_impressions",
    "article_reads",
    "article_reads_at_hour",
    "peer_reads_at_hour",
    "doctor_avg_read_hour",
    "doctor_lunch_read_share",
    "log_dwell_seconds",
]


@dataclass
class LogStats:
    article_impressions: dict[str, int]
    article_reads: dict[str, int]
    article_reads_at_hour: dict[tuple[str, int], int]
    peer_reads_at_hour: dict[tuple[str, int], int]
    doctor_avg_read_hour: dict[str, float]
    doctor_lunch_read_share: dict[str, float]
    pair_max_dwell: dict[tuple[str, str], float]
    pair_label: dict[tuple[str, str], int]

    @classmethod
    def from_frames(
        cls,
        doctors_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        event_logs_df: pd.DataFrame,
    ) -> LogStats:
        article_impressions: dict[str, int] = {}
        article_reads: dict[str, int] = {}
        article_reads_at_hour: dict[tuple[str, int], int] = {}
        peer_reads_at_hour: dict[tuple[str, int], int] = {}
        doctor_hours: dict[str, list[int]] = {}
        doctor_lunch: dict[str, list[bool]] = {}
        pair_max_dwell: dict[tuple[str, str], float] = {}
        pair_label: dict[tuple[str, str], int] = {}

        specialty_by_doctor = doctors_df.set_index("id")["specialty"].to_dict()
        event_weight = {"impression": 1, "click": 2, "read_complete": 3}

        for row in event_logs_df.itertuples(index=False):
            aid = row.article_id
            did = row.doctor_id
            hour = int(row.hour)
            et = row.event_type

            article_impressions[aid] = article_impressions.get(aid, 0) + 1
            if et == "read_complete":
                article_reads[aid] = article_reads.get(aid, 0) + 1
                key = (aid, hour)
                article_reads_at_hour[key] = article_reads_at_hour.get(key, 0) + 1
                spec = specialty_by_doctor.get(did, "")
                peer_key = (spec, hour)
                peer_reads_at_hour[peer_key] = peer_reads_at_hour.get(peer_key, 0) + 1
                doctor_hours.setdefault(did, []).append(hour)
                doctor_lunch.setdefault(did, []).append(12 <= hour < 14)

            pair = (did, aid)
            pair_max_dwell[pair] = max(pair_max_dwell.get(pair, 0.0), float(row.dwell_seconds))
            pair_label[pair] = max(pair_label.get(pair, 0), event_weight.get(et, 0))

        for row in interactions_df.itertuples(index=False):
            pair = (row.doctor_id, row.article_id)
            rating_label = max(0, int(row.rating) - 1)
            pair_label[pair] = max(pair_label.get(pair, 0), rating_label)

        doctor_avg_read_hour = {
            did: float(np.mean(hours)) if hours else 12.0
            for did, hours in doctor_hours.items()
        }
        doctor_lunch_read_share = {
            did: float(np.mean(flags)) if flags else 0.25
            for did, flags in doctor_lunch.items()
        }

        return cls(
            article_impressions=article_impressions,
            article_reads=article_reads,
            article_reads_at_hour=article_reads_at_hour,
            peer_reads_at_hour=peer_reads_at_hour,
            doctor_avg_read_hour=doctor_avg_read_hour,
            doctor_lunch_read_share=doctor_lunch_read_share,
            pair_max_dwell=pair_max_dwell,
            pair_label=pair_label,
        )

    def relevance_label(self, doctor_id: str, article_id: str) -> int:
        return self.pair_label.get((doctor_id, article_id), 0)


def build_feature_row(
    doctor_row: pd.Series,
    article_row: pd.Series,
    hour: int,
    content_score: float,
    collab_score: float,
    alpha: float,
    stats: LogStats,
) -> list[float]:
    slot = get_time_slot(hour)
    art = article_row.to_dict()
    boost = context_boost(art, slot)
    spec = doctor_row["specialty"]
    aid = article_row["id"]
    did = doctor_row["id"]

    return [
        float(content_score),
        float(collab_score),
        float(alpha),
        1.0 if art["specialty"] == spec else 0.0,
        float(art["complexity_score"]),
        float(art["reading_time_minutes"]) / 20.0,
        float(boost),
        float(hour) / 23.0,
        float(stats.article_impressions.get(aid, 0)),
        float(stats.article_reads.get(aid, 0)),
        float(stats.article_reads_at_hour.get((aid, hour), 0)),
        float(stats.peer_reads_at_hour.get((spec, hour), 0)),
        float(stats.doctor_avg_read_hour.get(did, 12.0)) / 23.0,
        float(stats.doctor_lunch_read_share.get(did, 0.25)),
        float(stats.pair_max_dwell.get((did, aid), 0.0)) / 600.0,
    ]
