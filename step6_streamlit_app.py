import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------- Page config ----------
st.set_page_config(page_title="Forecast Dashboard", layout="wide")
st.title("📈 Predictive Analytics Dashboard")
st.caption("Prophet + External Regressors • Interactive Forecasting")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ Configuration")
uploaded = st.sidebar.file_uploader(
    "Upload CSV (ds, y, promo, weather, is_holiday)", type="csv"
)

horizon = st.sidebar.slider("Forecast horizon (months)", 3, 24, 12)
future_promo = st.sidebar.selectbox("Promo next months?", [0, 1])
future_weather = st.sidebar.slider("Avg weather index", -20, 20, 0)
future_holiday = st.sidebar.selectbox("Holiday season?", [0, 1])

# ---------- Load Data ----------
if uploaded:
    df = pd.read_csv(uploaded, parse_dates=["ds"])
    st.sidebar.success("✅ Using uploaded file")
else:
    df = pd.read_csv("historical_data_clean.csv", parse_dates=["ds"])
    st.sidebar.info("📂 Using default historical_data_clean.csv")

st.subheader("📊 Historical Data (last 10 rows)")
st.dataframe(df.tail(10), use_container_width=True)

# ---------- Train Prophet ----------
with st.spinner("Training Prophet model..."):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
    )
    model.add_country_holidays(country_name="US")
    for col in ["promo", "weather", "is_holiday"]:
        if col in df.columns:
            model.add_regressor(col)
    model.fit(df)

# ---------- Future Frame ----------
last_date = df["ds"].max()
future_dates = pd.date_range(
    last_date + pd.DateOffset(months=1), periods=horizon, freq="MS"
)
future = pd.DataFrame({
    "ds": future_dates,
    "promo": future_promo,
    "weather": future_weather,
    "is_holiday": future_holiday,
})

forecast = model.predict(future)

# ---------- Forecast Chart ----------
st.subheader("🔮 Forecast")
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=df["ds"], y=df["y"],
    name="Historical", mode="lines+markers",
    line=dict(color="steelblue"),
))

# Forecast
fig.add_trace(go.Scatter(
    x=forecast["ds"], y=forecast["yhat"],
    name="Forecast", mode="lines+markers",
    line=dict(color="orange"),
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=pd.concat([forecast["ds"], forecast["ds"][::-1]]),
    y=pd.concat([forecast["yhat_upper"], forecast["yhat_lower"][::-1]]),
    fill="toself",
    fillcolor="rgba(255,165,0,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Interval",
))

fig.update_layout(
    height=500,
    xaxis_title="Date",
    yaxis_title="Sales",
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ---------- Metrics (last 12 months backtest) ----------
if len(df) > 12:
    st.subheader("📊 Model Accuracy (last 12 months holdout)")
    train, test = df.iloc[:-12], df.iloc[-12:]

    m = Prophet(yearly_seasonality=True)
    for col in ["promo", "weather", "is_holiday"]:
        if col in df.columns:
            m.add_regressor(col)
    m.fit(train)

    cols = [c for c in ["promo", "weather", "is_holiday"] if c in df.columns]
    fc = m.predict(test[["ds"] + cols])

    y_true = test["y"].values
    y_pred = fc["yhat"].values

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.2f}")
    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    c3.metric("R²", f"{r2_score(y_true, y_pred):.3f}")

# ---------- Components ----------
st.subheader("🧩 Trend & Seasonality Components")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

# ---------- Forecast Table ----------
st.subheader("📋 Forecast Values")
st.dataframe(
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].round(2),
    use_container_width=True,
)

# ---------- Download ----------
st.download_button(
    "⬇️ Download Forecast CSV",
    forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv(index=False),
    file_name="forecast.csv",
    mime="text/csv",
)

st.caption("Built with Prophet + Streamlit")