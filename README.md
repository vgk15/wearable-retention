# Wearable User Retention & Health Engagement Dashboard

**The problem:** Wearable devices such as WHOOP bands, Garmin watches, and other fitness trackers often face a retention challenge. Many users are highly engaged when they first get their device, regularly tracking metrics like sleep, recovery, and activity, but their usage often declines after a few weeks or months. Eventually, some stop wearing the device, stop using the app, or cancel their subscription altogether.

For wearable companies, this disengagement leads to lost revenue, lower customer retention, and fewer opportunities to improve users' health outcomes. This project aims to identify early signs of disengagement so companies can proactively re-engage users before they stop using the product.

**Project Outcome:** This project helps wearable companies identify users who may stop using devices such as WHOOP bands, Garmin watches, and other fitness trackers before they disengage completely. The model uses physiological data to achieve the desired outcomes:

1. **Predict** churn risk by assigning each user a 0–100% likelihood of disengagement.
2. **Segment** users into behavioral groups such as Highly Active, Steady, Casual, and At-Risk.
3. **Recommend** actions and explain why a user was flagged.
4. **Visualize** insights in an interactive dashboard designed for product, marketing, and retention teams.

The project leverages a realistic synthetic dataset to provide a fully reproducible demonstration of how wearable technology companies can transform user engagement and health signals into actionable insights that drive retention, enhance customer lifetime value, and inform strategic business decisions.


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
