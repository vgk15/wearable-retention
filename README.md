# Wearable User Retention & Health Engagement Dashboard

**The problem:** Wearable devices such as WHOOP bands, Garmin watches, and other fitness trackers often face a retention challenge. Many users are highly engaged when they first get their device, regularly tracking metrics like sleep, recovery, and activity, but their usage often declines after a few weeks or months. Eventually, some stop wearing the device, stop using the app, or cancel their subscription altogether.

For wearable companies, this disengagement leads to lost revenue, lower customer retention, and fewer opportunities to improve users' health outcomes. This project aims to identify early signs of disengagement so companies can proactively re-engage users before they stop using the product.

**What this project does:** This is an end-to-end analytics tool that helps a wearable company
spot at-risk users *early* and act on it. Using daily activity and health signals — steps, sleep,
workouts, heart rate, app logins, how consistently the device is worn, and whether usage is
trending up or down — it does four things:

1. **Predicts** each user's risk of disengaging, as a 0–100% risk score, using machine learning.
2. **Segments** the user base into behavioral groups (for example, *Highly Active*, *Steady*,
   *Casual*, and *At-Risk* users) so teams can see who they're serving.
3. **Recommends** a specific, plain-English next step for any user (e.g. "improve sleep
   consistency," "re-engage before drop-off") — and shows *why* the model flagged them.
4. **Visualizes** all of it in an interactive dashboard a product or marketing team could
   actually use day to day.

Everything runs on a realistic **synthetic dataset** (no real personal data), so the project is
fully reproducible by anyone who clones it. The result is a complete, honest demonstration of how
a data team turns raw wearable signals into retention decisions.

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
