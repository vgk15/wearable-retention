"""
Model-agnostic per-user risk explanation and live scoring.

Works for any champion model (Logistic Regression or Random Forest) without extra
dependencies. The explanation is a *marginal contribution*: for each feature we measure
how the user's risk score changes when that feature is reset to the population-typical
(median) value, holding everything else fixed. Positive contribution = the feature is
pushing this user's risk UP.

The same `risk_from_values` helper drives the dashboard's what-if simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_from_values(values: dict, pipe: dict) -> float:
    """Predict at-risk probability for one feature dict using the champion model."""
    cols = pipe["feature_cols"]
    row = np.array([[values[c] for c in cols]], dtype=float)
    model = pipe["models"][pipe["champion"]]
    if pipe["champion"] == "Logistic Regression":
        row = pipe["scaler"].transform(row)
    return float(model.predict_proba(row)[0, 1])


def explain_user(user: pd.Series, pipe: dict, baseline: pd.Series) -> pd.DataFrame:
    """Per-feature marginal contribution to this user's risk score (sorted by magnitude)."""
    cols = pipe["feature_cols"]
    base_vals = {c: float(user[c]) for c in cols}
    current = risk_from_values(base_vals, pipe)

    rows = []
    for c in cols:
        swapped = dict(base_vals)
        swapped[c] = float(baseline[c])  # reset this one feature to the typical value
        risk_if_typical = risk_from_values(swapped, pipe)
        rows.append({
            "feature": c,
            "user_value": round(base_vals[c], 2),
            "typical_value": round(float(baseline[c]), 2),
            "contribution": round(current - risk_if_typical, 4),  # + => raises risk
        })
    out = pd.DataFrame(rows)
    out["abs"] = out["contribution"].abs()
    return out.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)
