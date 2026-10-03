import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load cleaned data
df = pd.read_csv("historical_data_clean.csv", parse_dates=["ds"])

# Train/test split — last 12 months for testing
train = df.iloc[:-12]
test = df.iloc[-12:]

print(f"Train: {len(train)} months | Test: {len(test)} months")

# --- Build Prophet model ---
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.15,
)
model.add_country_holidays(country_name="US")
model.add_regressor("promo")
model.add_regressor("weather")
model.add_regressor("is_holiday")

# --- Train ---
model.fit(train)

# --- Predict on test ---
future = test[["ds", "promo", "weather", "is_holiday"]].copy()
forecast = model.predict(future)

y_true = test["y"].values
y_pred = forecast["yhat"].values

# --- Metrics ---
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n📊 Prophet Performance:")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.3f}")

# --- Save forecast ---
forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_csv("prophet_forecast.csv", index=False)

# --- Plot 1: Forecast ---
fig1 = model.plot(forecast)
plt.title("Prophet Forecast (Last 12 Months)")
plt.tight_layout()
plt.savefig("prophet_forecast.png", dpi=120)
plt.show()

# --- Plot 2: Components ---
fig2 = model.plot_components(forecast)
plt.savefig("prophet_components.png", dpi=120)
plt.show()

print("✅ Saved: prophet_forecast.csv, prophet_forecast.png, prophet_components.png")