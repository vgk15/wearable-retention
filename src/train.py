"""
Train the retention classifiers and the K-Means segmentation, then persist everything
the dashboard needs (models, scaler, segment profiles, metrics, scored users).

Run:  python -m src.train       (from the project root)
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.generate_data import FEATURE_COLS, generate

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts")
DATA = os.path.join(ROOT, "data")

# Features the K-Means uses to describe *behavior* (demographics excluded on purpose).
SEGMENT_COLS = [
    "daily_steps",
    "sleep_hours",
    "active_minutes",
    "workout_freq_per_week",
    "app_logins_per_week",
    "device_wear_days_per_week",
    "usage_trend",
]


def _name_segments(profile: pd.DataFrame) -> dict:
    """Assign human-readable names to clusters by ranking their composite activity."""
    score = (
        profile["daily_steps"].rank()
        + profile["active_minutes"].rank()
        + profile["workout_freq_per_week"].rank()
        + profile["app_logins_per_week"].rank()
        + profile["usage_trend"].rank()
    )
    order = score.sort_values().index.tolist()  # lowest activity first
    labels = {}
    n = len(order)
    if n == 3:
        names = ["At-Risk Users", "Casual Users", "Highly Active"]
    elif n == 4:
        names = ["At-Risk Users", "Casual Users", "Steady Users", "Highly Active"]
    else:
        names = [f"Segment {i}" for i in range(n)]
    for cluster_id, name in zip(order, names):
        labels[int(cluster_id)] = name
    return labels


def main(n_users: int = 8000, n_segments: int = 4, seed: int = 42):
    os.makedirs(ART, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)

    df = generate(n_users=n_users, seed=seed)
    df.to_csv(os.path.join(DATA, "wearable_users.csv"), index=False)

    X = df[FEATURE_COLS].values
    y = (df["status"] == "At Risk").astype(int).values  # 1 = at-risk (positive class)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.25, random_state=seed, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        ),
    }

    metrics, fitted = {}, {}
    for name, model in models.items():
        # LogReg needs scaling; RF is scale-invariant but we keep one feature space.
        Xtr = X_train_s if name == "Logistic Regression" else X_train
        Xte = X_test_s if name == "Logistic Regression" else X_test
        model.fit(Xtr, y_train)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test, pred), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "report": classification_report(
                y_test, pred, target_names=["Active", "At Risk"], output_dict=True
            ),
        }
        fitted[name] = model

    # Champion = best ROC-AUC. Used to produce the risk score for every user.
    champion = max(metrics, key=lambda k: metrics[k]["roc_auc"])
    champ_model = fitted[champion]
    X_all = scaler.transform(X) if champion == "Logistic Regression" else X
    df["risk_score"] = champ_model.predict_proba(X_all)[:, 1].round(4)

    # Feature importance (RF impurity importance + LogReg |coef| on scaled features).
    rf = fitted["Random Forest"]
    fi = pd.DataFrame({"feature": FEATURE_COLS, "rf_importance": rf.feature_importances_})
    lr = fitted["Logistic Regression"]
    fi["logreg_abs_coef"] = np.abs(lr.coef_[0])
    fi = fi.sort_values("rf_importance", ascending=False).reset_index(drop=True)

    # ----- K-Means segmentation on behavior features -----
    seg_scaler = StandardScaler().fit(df[SEGMENT_COLS].values)
    km = KMeans(n_clusters=n_segments, random_state=seed, n_init=10)
    df["cluster"] = km.fit_predict(seg_scaler.transform(df[SEGMENT_COLS].values))

    profile = df.groupby("cluster")[SEGMENT_COLS].mean()
    seg_names = _name_segments(profile)
    df["segment"] = df["cluster"].map(seg_names)

    seg_summary = (
        df.groupby("segment")
        .agg(
            users=("user_id", "size"),
            at_risk_rate=("status", lambda s: (s == "At Risk").mean()),
            avg_risk_score=("risk_score", "mean"),
            avg_steps=("daily_steps", "mean"),
            avg_sleep=("sleep_hours", "mean"),
            avg_active_min=("active_minutes", "mean"),
            avg_workouts=("workout_freq_per_week", "mean"),
            avg_logins=("app_logins_per_week", "mean"),
            avg_usage_trend=("usage_trend", "mean"),
        )
        .round(3)
        .reset_index()
    )

    # ----- persist everything -----
    joblib.dump(
        {
            "models": fitted,
            "champion": champion,
            "scaler": scaler,
            "feature_cols": FEATURE_COLS,
            "seg_scaler": seg_scaler,
            "kmeans": km,
            "segment_names": seg_names,
            "segment_cols": SEGMENT_COLS,
        },
        os.path.join(ART, "pipeline.joblib"),
    )
    df.to_csv(os.path.join(ART, "scored_users.csv"), index=False)
    fi.to_csv(os.path.join(ART, "feature_importance.csv"), index=False)
    seg_summary.to_csv(os.path.join(ART, "segment_summary.csv"), index=False)
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump({"champion": champion, "models": metrics}, f, indent=2)

    # ----- console report -----
    print(f"Users: {len(df):,}  |  At-risk rate: {y.mean():.1%}\n")
    for name, m in metrics.items():
        star = "  <-- champion" if name == champion else ""
        print(f"{name:>22}:  accuracy={m['accuracy']:.3f}  ROC-AUC={m['roc_auc']:.3f}{star}")
    print("\nTop features (Random Forest):")
    for _, r in fi.head(5).iterrows():
        print(f"  {r['feature']:<26} {r['rf_importance']:.3f}")
    print("\nSegments:")
    print(seg_summary.to_string(index=False))
    print(f"\nArtifacts written to {ART}")


if __name__ == "__main__":
    main()
