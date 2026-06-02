"""
Interpretable recommendation engine.

Each recommendation compares a user's metrics against evidence-informed healthy
benchmarks (and their segment context), so every suggestion carries a visible "why".
Output is ordered by estimated impact on retention.
"""

from __future__ import annotations

import pandas as pd

# Healthy targets for general adult wellness (rounded, defensible benchmarks).
BENCHMARKS = {
    "daily_steps": 8000,
    "sleep_hours": 7.0,
    "active_minutes": 30,          # ~150 min/week
    "workout_freq_per_week": 3,
    "app_logins_per_week": 5,
    "device_wear_days_per_week": 6,
    "usage_trend": 1.0,            # not declining vs personal baseline
}


def recommend(user: pd.Series, max_recs: int = 4) -> list[dict]:
    """Return a ranked list of {title, detail, priority} recommendations for one user."""
    recs: list[dict] = []

    def add(priority, title, detail):
        recs.append({"priority": priority, "title": title, "detail": detail})

    # Declining usage trend is the strongest churn signal -> highest priority.
    if user["usage_trend"] < 0.85:
        add(
            3,
            "Re-engage before drop-off",
            f"Recent usage is {(1 - user['usage_trend']) * 100:.0f}% below this user's own "
            "baseline. Trigger a check-in nudge, a streak challenge, or a fresh goal.",
        )

    if user["daily_steps"] < BENCHMARKS["daily_steps"]:
        gap = BENCHMARKS["daily_steps"] - user["daily_steps"]
        add(
            2,
            "Raise the daily step goal",
            f"Averaging {int(user['daily_steps']):,} steps/day — about {int(gap):,} short of "
            f"{BENCHMARKS['daily_steps']:,}. Suggest a gradual +500/week ramp.",
        )

    if user["sleep_hours"] < 6.5:
        add(
            2,
            "Improve sleep consistency",
            f"Sleeping {user['sleep_hours']:.1f} h/night. Enable a wind-down reminder and a "
            "consistent bedtime schedule (target 7–8 h).",
        )

    if user["workout_freq_per_week"] < BENCHMARKS["workout_freq_per_week"]:
        add(
            1,
            "Add a weekly workout",
            f"Working out {int(user['workout_freq_per_week'])}×/week. Nudge toward "
            f"{BENCHMARKS['workout_freq_per_week']}× with short guided sessions.",
        )

    if user["active_minutes"] < BENCHMARKS["active_minutes"]:
        add(
            1,
            "Increase active minutes",
            f"{int(user['active_minutes'])} active min/day vs a {BENCHMARKS['active_minutes']}-min "
            "target. Suggest two 10-minute movement breaks.",
        )

    if user["app_logins_per_week"] < BENCHMARKS["app_logins_per_week"]:
        add(
            2,
            "Build an app-check-in habit",
            f"Opening the app {int(user['app_logins_per_week'])}×/week. Surface a morning "
            "summary notification to build a daily habit.",
        )

    if user["device_wear_days_per_week"] < BENCHMARKS["device_wear_days_per_week"]:
        add(
            2,
            "Encourage consistent wear",
            f"Wearing the device {int(user['device_wear_days_per_week'])} days/week. Highlight "
            "sleep tracking + charging tips so it stays on more days.",
        )

    if not recs:
        add(
            0,
            "Healthy & engaged — reinforce it",
            "Metrics meet wellness targets. Celebrate streaks and offer stretch goals or "
            "community challenges to sustain momentum.",
        )

    recs.sort(key=lambda r: r["priority"], reverse=True)
    return recs[:max_recs]


def find_similar(user: pd.Series, pool: pd.DataFrame, cols: list[str], k: int = 5) -> pd.DataFrame:
    """Nearest users by normalized behavior distance — powers 'users like this one'."""
    p = pool[pool["user_id"] != user["user_id"]].copy()
    stds = pool[cols].std().replace(0, 1.0)
    dist = (((p[cols] - user[cols]) / stds) ** 2).sum(axis=1) ** 0.5
    p = p.assign(distance=dist).sort_values("distance")
    return p.head(k)
