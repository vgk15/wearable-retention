"""
AI-Powered Wearable User Retention & Health Engagement Dashboard.

Run:  streamlit run app.py
(Run `python -m src.train` first to generate the artifacts.)
"""

import json
import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.explain import explain_user, risk_from_values
from src.recommend import find_similar, recommend

ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")

st.set_page_config(page_title="Wearable Retention Dashboard", layout="wide")

# Color convention: green = positive / active, red = negative / at-risk.
ACTIVE_C, RISK_C = "#2E7D32", "#C62828"


@st.cache_data
def load():
    df = pd.read_csv(os.path.join(ART, "scored_users.csv"))
    fi = pd.read_csv(os.path.join(ART, "feature_importance.csv"))
    seg = pd.read_csv(os.path.join(ART, "segment_summary.csv"))
    with open(os.path.join(ART, "metrics.json")) as f:
        metrics = json.load(f)
    return df, fi, seg, metrics


@st.cache_resource
def load_pipeline():
    return joblib.load(os.path.join(ART, "pipeline.joblib"))


if not os.path.exists(os.path.join(ART, "scored_users.csv")):
    st.error("Artifacts not found. Run `python -m src.train` from the project root first.")
    st.stop()

df, fi, seg, metrics = load()
pipe = load_pipeline()
seg_cols = pipe["segment_cols"]

st.title("Wearable User Retention & Health Engagement")
st.caption(
    "Predicting disengagement risk, segmenting behavior, and recommending personalized "
    "interventions on synthetic wearable data."
)

# ---------------- sidebar filters ----------------
with st.sidebar:
    st.header("Filters")
    segs = sorted(df["segment"].unique())
    sel_segs = st.multiselect("Segments", segs, default=segs)
    sel_status = st.multiselect("Status", ["Active", "At Risk"], default=["Active", "At Risk"])
    age_lo, age_hi = int(df["age"].min()), int(df["age"].max())
    sel_age = st.slider("Age range", age_lo, age_hi, (age_lo, age_hi))
    risk_thr = st.slider("Flag as at-risk when risk score ≥", 0.0, 1.0, 0.5, 0.05)

f = df[
    df["segment"].isin(sel_segs)
    & df["status"].isin(sel_status)
    & df["age"].between(*sel_age)
].copy()

tab_over, tab_pred, tab_seg, tab_health, tab_rec = st.tabs(
    ["Overview", "Retention Predictions", "Segments", "Health Metrics", "Recommendations"]
)

# ================= OVERVIEW =================
with tab_over:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users (filtered)", f"{len(f):,}")
    c2.metric("At-risk rate", f"{(f['status'] == 'At Risk').mean():.1%}" if len(f) else "—")
    c3.metric("Flagged at threshold", f"{(f['risk_score'] >= risk_thr).sum():,}")
    c4.metric("Champion model AUC", f"{metrics['models'][metrics['champion']]['roc_auc']:.3f}")

    left, right = st.columns(2)
    with left:
        counts = f["status"].value_counts().reset_index()
        counts.columns = ["status", "count"]
        fig = px.pie(
            counts, names="status", values="count", hole=0.5, title="Active vs At-Risk",
            color="status", color_discrete_map={"Active": ACTIVE_C, "At Risk": RISK_C},
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.histogram(
            f, x="risk_score", color="status", nbins=30, title="Risk-score distribution",
            color_discrete_map={"Active": ACTIVE_C, "At Risk": RISK_C},
        )
        fig.add_vline(x=risk_thr, line_dash="dash", line_color="black")
        st.plotly_chart(fig, use_container_width=True)

# ================= PREDICTIONS =================
with tab_pred:
    st.subheader("Model performance (held-out test set)")
    rows = []
    for name, m in metrics["models"].items():
        rep = m["report"]["At Risk"]
        rows.append({
            "Model": name + (" (champion)" if name == metrics["champion"] else ""),
            "Accuracy": m["accuracy"], "ROC-AUC": m["roc_auc"],
            "Precision (At Risk)": round(rep["precision"], 3),
            "Recall (At Risk)": round(rep["recall"], 3),
            "F1 (At Risk)": round(rep["f1-score"], 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "The champion model (highest ROC-AUC) generates the per-user risk scores. AUC well below 1.0 "
        "is expected and healthy — labels come from a hidden propensity + noise, not a leaked rule."
    )

    cm = metrics["models"][metrics["champion"]]["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual Active", "Actual At Risk"],
                         columns=["Pred Active", "Pred At Risk"])
    cfig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues",
                     title=f"Confusion matrix — {metrics['champion']}")
    st.plotly_chart(cfig, use_container_width=True)

    st.subheader("Feature importance")
    fig = px.bar(fi.sort_values("rf_importance"), x="rf_importance", y="feature",
                 orientation="h", title="Random Forest importance")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-risk users (filtered)")
    flagged = f[f["risk_score"] >= risk_thr].sort_values("risk_score", ascending=False)
    show_cols = ["user_id", "segment", "status", "risk_score", "usage_trend",
                 "daily_steps", "sleep_hours", "app_logins_per_week"]
    st.dataframe(flagged[show_cols].head(25), use_container_width=True, hide_index=True)
    st.download_button(
        f"Export all {len(flagged):,} flagged users (risk ≥ {risk_thr:.2f}) as CSV",
        data=flagged.to_csv(index=False).encode("utf-8"),
        file_name=f"flagged_users_thr{risk_thr:.2f}.csv",
        mime="text/csv",
        disabled=flagged.empty,
    )

# ================= SEGMENTS =================
with tab_seg:
    st.subheader("Behavioral segments (K-Means)")
    st.dataframe(seg, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        sc = f["segment"].value_counts().reset_index()
        sc.columns = ["segment", "users"]
        st.plotly_chart(px.bar(sc, x="segment", y="users", color="segment",
                               title="Users per segment"), use_container_width=True)
    with c2:
        st.plotly_chart(
            px.bar(seg.sort_values("at_risk_rate"), x="segment", y="at_risk_rate",
                   title="At-risk rate by segment", color="at_risk_rate",
                   color_continuous_scale="Reds"),
            use_container_width=True,
        )

    st.subheader("Segment map: activity vs engagement")
    fig = px.scatter(
        f.sample(min(len(f), 2500), random_state=0),
        x="daily_steps", y="app_logins_per_week", color="segment",
        size="active_minutes", hover_data=["user_id", "risk_score"],
        title="Steps vs app logins (bubble = active minutes)",
    )
    st.plotly_chart(fig, use_container_width=True)

# ================= HEALTH METRICS =================
with tab_health:
    st.subheader("Health & activity metrics by status")
    metric = st.selectbox(
        "Metric",
        ["daily_steps", "sleep_hours", "active_minutes", "workout_freq_per_week",
         "resting_heart_rate", "app_logins_per_week", "device_wear_days_per_week", "usage_trend"],
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.box(f, x="status", y=metric, color="status",
                   color_discrete_map={"Active": ACTIVE_C, "At Risk": RISK_C},
                   title=f"{metric} by status"),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            px.box(f, x="segment", y=metric, color="segment", title=f"{metric} by segment"),
            use_container_width=True,
        )
    st.plotly_chart(
        px.histogram(f, x=metric, color="status", barmode="overlay", nbins=30,
                     color_discrete_map={"Active": ACTIVE_C, "At Risk": RISK_C},
                     title=f"Distribution of {metric}"),
        use_container_width=True,
    )

# ================= RECOMMENDATIONS =================
with tab_rec:
    st.subheader("Personalized recommendations")
    default_user = int(f.sort_values("risk_score", ascending=False)["user_id"].iloc[0]) if len(f) else 1
    uid = st.number_input("User ID", min_value=int(df["user_id"].min()),
                          max_value=int(df["user_id"].max()), value=default_user, step=1)
    user = df[df["user_id"] == uid].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", user["status"])
    c2.metric("Risk score", f"{user['risk_score']:.2f}")
    c3.metric("Segment", user["segment"])

    st.markdown("**Recommended actions** (ranked by impact):")
    for r in recommend(user):
        tag = {
            3: ":red[**High priority**]",
            2: ":orange[**Medium priority**]",
            1: ":gray[**Low priority**]",
            0: ":green[**Maintain**]",
        }[r["priority"]]
        st.markdown(f"- {tag} — **{r['title']}.** {r['detail']}")

    # ----- per-user risk explanation (model-agnostic marginal contributions) -----
    st.markdown("---")
    st.subheader("Why this risk score?")
    baseline = df[pipe["feature_cols"]].median()
    exp = explain_user(user, pipe, baseline)
    exp["effect"] = exp["contribution"].apply(lambda v: "Raises risk" if v > 0 else "Lowers risk")
    efig = px.bar(
        exp.iloc[::-1], x="contribution", y="feature", orientation="h", color="effect",
        color_discrete_map={"Raises risk": RISK_C, "Lowers risk": ACTIVE_C},
        title="Each feature's contribution vs a typical user",
        hover_data=["user_value", "typical_value"],
    )
    efig.add_vline(x=0, line_color="black")
    st.plotly_chart(efig, use_container_width=True)
    st.caption(
        "Contribution = how much this user's risk score changes when that single metric is "
        "reset to the population median, holding everything else fixed."
    )

    # ----- what-if simulator -----
    st.markdown("---")
    st.subheader("What-if simulator")
    st.caption("Adjust this user's metrics and watch the predicted risk score respond live.")
    sim_specs = {
        "daily_steps": (0, 25000, 100), "sleep_hours": (3.0, 11.0, 0.1),
        "active_minutes": (0, 200, 1), "workout_freq_per_week": (0, 14, 1),
        "resting_heart_rate": (40, 110, 1), "app_logins_per_week": (0, 50, 1),
        "device_wear_days_per_week": (0, 7, 1), "usage_trend": (0.2, 1.8, 0.01),
    }
    vals = {c: float(user[c]) for c in pipe["feature_cols"]}  # age/tenure held fixed
    s1, s2 = st.columns(2)
    for i, (feat, (lo, hi, step)) in enumerate(sim_specs.items()):
        col = s1 if i % 2 == 0 else s2
        vals[feat] = col.slider(
            feat, float(lo), float(hi), float(user[feat]), float(step), key=f"sim_{feat}"
        )
    new_risk = risk_from_values(vals, pipe)
    delta = new_risk - user["risk_score"]
    st.metric("Simulated risk score", f"{new_risk:.2f}",
              delta=f"{delta:+.2f} vs current", delta_color="inverse")

    st.markdown("**Similar users** (nearest behavior profiles):")
    sim = find_similar(user, df, seg_cols, k=5)
    st.dataframe(
        sim[["user_id", "segment", "status", "risk_score"] + seg_cols],
        use_container_width=True, hide_index=True,
    )
