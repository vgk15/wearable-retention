# Wearable User Retention & Health Engagement Dashboard

Predicts which wearable users are at risk of disengaging, segments users by behavior,
and serves personalized, interpretable recommendations — all on a synthetic dataset and
exposed through an interactive Streamlit dashboard.

## Why this is built the way it is

The common mistake in projects like this is **circular labeling**: generate random
metrics, then define "at risk" with a hard rule (e.g. `steps < 4000`). A model then just
re-learns your rule and reports a meaningless ~100% accuracy.

Here, each user has a **hidden engagement propensity** that *drives* their observed
behavior through noisy channels, and the retention label is drawn from that latent
variable **plus independent noise**. The visible metrics are imperfect signals — so the
models learn something real and the metrics are honest (ROC-AUC ≈ 0.78, not 1.0).

## Setup

```bash
cd wearable-retention
pip install -r requirements.txt
```

## Run

```bash
python -m src.train        # 1) generate data, train models + segments, write artifacts/
streamlit run app.py       # 2) launch the dashboard
```

`python -m src.train` prints model metrics and segment profiles, and writes everything
the dashboard needs into `artifacts/`.

## Project layout

| Path | Purpose |
|------|---------|
| `src/generate_data.py` | Synthetic data (latent-propensity design, no label leakage) |
| `src/train.py` | LogReg + Random Forest classifiers, K-Means segmentation, persistence |
| `src/recommend.py` | Interpretable benchmark-gap recommendations + nearest-user lookup |
| `app.py` | Streamlit dashboard (overview, predictions, segments, health, recommendations) |
| `data/` | Generated raw dataset |
| `artifacts/` | Trained pipeline, scored users, metrics, segment summaries |

## What the dashboard shows

- **Overview** — active vs at-risk split, risk-score distribution, adjustable risk threshold.
- **Retention Predictions** — model comparison (accuracy, ROC-AUC, precision/recall/F1),
  confusion matrix, feature importance, highest-risk users.
- **Segments** — K-Means behavioral groups (Highly Active → At-Risk), per-segment at-risk rates.
- **Health Metrics** — distributions of steps, sleep, active minutes, HR, etc. by status/segment.
- **Recommendations** — ranked personalized actions for any user, plus similar-user profiles.

## Tech

Python · pandas · NumPy · scikit-learn · Plotly · Streamlit · joblib
