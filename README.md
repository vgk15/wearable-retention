# Wearable User Retention & Health Engagement Dashboard

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20App-FF4B4B?logo=streamlit&logoColor=white)](https://wearable-retention-fyvrqfyepeumhv4tehjvap.streamlit.app/)

**[▶ Launch the live dashboard](https://wearable-retention-fyvrqfyepeumhv4tehjvap.streamlit.app/)**

**The problem:** Wearable devices such as WHOOP bands, Garmin watches, and other fitness trackers often face a retention challenge. Many users are highly engaged when they first get their device, regularly tracking metrics like sleep, recovery, and activity, but their usage often declines after a few weeks or months. Eventually, some stop wearing the device, stop using the app, or cancel their subscription altogether.

For wearable companies, this disengagement leads to lost revenue, lower customer retention, and fewer opportunities to improve users' health outcomes. This project aims to identify early signs of disengagement so companies can proactively re-engage users before they stop using the product.

**Project Outcome:** This project helps wearable companies identify users who may stop using devices such as WHOOP bands, Garmin watches, and other fitness trackers before they disengage completely. The model uses physiological data to achieve the desired outcomes:

1. Predict churn risk by assigning each user a 0–100% likelihood of disengagement.
2. Segment users into behavioral groups such as Highly Active, Steady, Casual, and At-Risk.
3. Recommend actions and explains why a user was flagged.
4. Visualize insights in an interactive dashboard designed for product, marketing, and retention teams.

The project leverages a realistic synthetic dataset to provide a fully reproducible demonstration of how wearable technology companies can transform user engagement and health signals into actionable insights that drive retention, enhance customer lifetime value, and inform strategic business decisions.


## Setup

```bash
cd wearable-retention
pip install -r requirements.txt
```

## Dashboard View

1. Overview: Active vs at-risk split, risk-score distribution, adjustable risk threshold.
2. Retention Predictions: model comparison, confusion matrix, feature importance, highest-risk users.
3. Segments: K-Means behavioral groups (Highly Active → At-Risk), per-segment at-risk rates.
4. Health Metrics: Distributions of steps, sleep, active minutes, HR, etc. by status/segment.
5. Recommendations: Ranked personalized actions for any user, plus similar-user profiles.

## Tech Stack and Libraries

Python | pandas | NumPy | scikit-learn | Plotly | Streamlit | joblib
