import streamlit as st
import pandas as pd
import joblib
import altair as alt
import numpy as np

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed_customer_data.csv")
    return df

@st.cache_resource
def load_model():
    model = joblib.load("models/customer_satisfaction_model.pkl")
    return model

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Customer Experience Dashboard", layout="wide")
st.title("📊 Customer Experience Dashboard")
st.markdown("Gain insights into customer satisfaction, feedback trends, and delivery performance.")

# ---------- LOAD ----------
df = load_data()
model = load_model()

# ---------- SIDEBAR FILTERS ----------
st.sidebar.header("🔍 Filters")
priority_filter = st.sidebar.multiselect("Order Priority", options=df["priority"].unique(), default=list(df["priority"].unique()) if "priority" in df.columns else [])
segment_filter = st.sidebar.multiselect("Customer Segment", options=df["customer_segment"].unique(), default=list(df["customer_segment"].unique()) if "customer_segment" in df.columns else [])

filtered_df = df.copy()
if "priority" in df.columns and len(priority_filter) > 0:
    filtered_df = filtered_df[filtered_df["priority"].isin(priority_filter)]
if "customer_segment" in df.columns and len(segment_filter) > 0:
    filtered_df = filtered_df[filtered_df["customer_segment"].isin(segment_filter)]

# ---------- KPIs ----------
avg_rating = filtered_df["rating"].mean() if "rating" in filtered_df.columns else np.nan
avg_sentiment = filtered_df["sent_compound"].mean() if "sent_compound" in filtered_df.columns else np.nan
low_satisfaction_pct = filtered_df["low_satisfaction"].mean() * 100 if "low_satisfaction" in filtered_df.columns else np.nan

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("⭐ Average Rating", f"{avg_rating:.2f}")
kpi2.metric("💬 Avg Sentiment", f"{avg_sentiment:.2f}")
kpi3.metric("⚠️ Low Satisfaction %", f"{low_satisfaction_pct:.1f}%")

st.divider()

# ---------- VISUALIZATIONS ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution of Ratings")
    if "rating" in filtered_df.columns:
        chart = alt.Chart(filtered_df).mark_bar(color="#4E79A7").encode(
            alt.X("rating:O", title="Customer Rating"),
            alt.Y("count()", title="Number of Orders")
        )
        st.altair_chart(chart, use_container_width=True)

with col2:
    st.subheader("Sentiment Score Distribution")
    if "sent_compound" in filtered_df.columns:
        chart2 = alt.Chart(filtered_df).mark_bar(color="#F28E2B").encode(
            alt.X("sent_compound:Q", bin=alt.Bin(maxbins=20), title="Sentiment Score"),
            alt.Y("count()", title="Number of Customers")
        )
        st.altair_chart(chart2, use_container_width=True)

st.divider()

# ---------- MODEL PREDICTION ----------
st.subheader("🧠 Predict Customer Satisfaction Risk")

numeric_features = filtered_df.select_dtypes(include=[np.number]).drop(columns=["low_satisfaction"], errors="ignore")
if len(numeric_features) > 0:
    preds = model.predict(numeric_features)
    filtered_df["predicted_low_satisfaction"] = preds
    risk_customers = filtered_df[filtered_df["predicted_low_satisfaction"] == 1]

    st.markdown(f"### ⚠️ At-Risk Customers: {len(risk_customers)} found")
    if len(risk_customers) > 0:
        st.dataframe(
            risk_customers[["order_id", "rating", "satisfaction_score", "sent_compound", "priority", "customer_segment"]].head(10),
            use_container_width=True
        )
    else:
        st.success("All customers currently appear satisfied. 🎉")

st.divider()
st.caption("Developed by Naman Srivastava — AI Internship Case Study")
