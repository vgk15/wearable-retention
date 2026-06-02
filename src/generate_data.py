"""
Synthetic wearable-user dataset generator.

Design principle (the important part):
We do NOT label users by a hard rule on their visible metrics (that would make the
ML task circular and produce fake ~100% accuracy). Instead each user has a hidden
`engagement_propensity` latent variable. That latent variable *drives* their observed
behavior (steps, sleep, logins, the usage trend, ...) through noisy channels, and the
retention label is drawn from the latent variable plus independent noise.

The result: the visible metrics are imperfect signals of an unobserved truth — exactly
the regime where a model has something real to learn, and where evaluation metrics
(AUC ~0.85, not 1.0) are honest.
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "age",
    "daily_steps",
    "sleep_hours",
    "active_minutes",
    "workout_freq_per_week",
    "resting_heart_rate",
    "app_logins_per_week",
    "device_wear_days_per_week",
    "tenure_months",
    "usage_trend",  # recent usage vs personal baseline; <1 means declining
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_users: int = 8000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ----- demographics -----
    age = np.clip(rng.normal(38, 12, n_users), 18, 80).round().astype(int)
    gender = rng.choice(["Female", "Male", "Other"], size=n_users, p=[0.49, 0.49, 0.02])
    tenure_months = np.clip(rng.exponential(10, n_users), 0.5, 60).round(1)

    # ----- hidden latent engagement propensity (mean 0, sd 1) -----
    # Slightly higher engagement for mid-age adults; newer users a touch more enthused.
    age_effect = -0.012 * (age - 35) ** 2 / 100.0
    tenure_effect = -0.15 * np.log1p(tenure_months) + 0.2  # honeymoon fades with tenure
    propensity = rng.normal(0, 1, n_users) + age_effect + tenure_effect
    propensity = (propensity - propensity.mean()) / propensity.std()

    def driven(base, scale, noise):
        """A metric driven by latent propensity plus its own noise."""
        return base + scale * propensity + rng.normal(0, noise, n_users)

    # ----- behavior driven by latent propensity (with realistic noise) -----
    daily_steps = np.clip(driven(7000, 2600, 1800), 500, 25000).round().astype(int)
    sleep_hours = np.clip(driven(7.0, 0.55, 0.9), 3.0, 11.0).round(1)
    active_minutes = np.clip(driven(38, 22, 18), 0, 200).round().astype(int)
    workout_freq = np.clip(driven(3.0, 1.7, 1.4), 0, 14).round().astype(int)
    # resting HR is inversely related to fitness/engagement
    resting_hr = np.clip(72 - 7 * propensity + rng.normal(0, 6, n_users), 40, 110).round().astype(int)
    app_logins = np.clip(driven(9, 6, 4), 0, 50).round().astype(int)
    wear_days = np.clip(driven(5.0, 1.6, 1.2), 0, 7).round().astype(int)
    # recent-vs-baseline usage ratio: disengaging users trend below 1.0
    usage_trend = np.clip(0.95 + 0.18 * propensity + rng.normal(0, 0.18, n_users), 0.2, 1.8).round(3)

    # ----- retention label from latent propensity + noise (NOT from a rule on features) -----
    # logit of "active" rises with propensity and with a healthy usage trend.
    logit = 1.1 * propensity + 1.4 * (usage_trend - 1.0) + rng.normal(0, 0.6, n_users)
    p_active = _sigmoid(logit)
    is_active = (rng.random(n_users) < p_active).astype(int)

    df = pd.DataFrame(
        {
            "user_id": np.arange(1, n_users + 1),
            "age": age,
            "gender": gender,
            "daily_steps": daily_steps,
            "sleep_hours": sleep_hours,
            "active_minutes": active_minutes,
            "workout_freq_per_week": workout_freq,
            "resting_heart_rate": resting_hr,
            "app_logins_per_week": app_logins,
            "device_wear_days_per_week": wear_days,
            "tenure_months": tenure_months,
            "usage_trend": usage_trend,
            # status: 1 = Active, 0 = At Risk of Disengagement
            "status": np.where(is_active == 1, "Active", "At Risk"),
        }
    )
    return df


if __name__ == "__main__":
    import os

    out = os.path.join(os.path.dirname(__file__), "..", "data", "wearable_users.csv")
    df = generate()
    df.to_csv(out, index=False)
    rate = (df["status"] == "At Risk").mean()
    print(f"Wrote {len(df):,} users -> {os.path.abspath(out)}")
    print(f"At-Risk rate: {rate:.1%}")
