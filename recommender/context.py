"""Time-of-day context for re-ranking."""

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
    complexity_fit = 1.0 - abs(article["complexity_score"] - slot["ideal_complexity"])
    time_fit = 1.0 if article["reading_time_minutes"] <= slot["max_reading_min"] else \
               max(0.0, 1.0 - (article["reading_time_minutes"] - slot["max_reading_min"]) / 20.0)
    fit = (complexity_fit + time_fit) / 2.0
    return 1.0 + lam * (2.0 * fit - 1.0)
