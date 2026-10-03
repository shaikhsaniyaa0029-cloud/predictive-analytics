import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (prevents plot window blocking)
import matplotlib.pyplot as plt

print("🚀 Starting cross-validation script...")

# ---------- Load ----------
print("📂 Loading data...")
df = pd.read_csv("historical_data_clean.csv", parse_dates=["ds"])
print(f"✅ Loaded {len(df)} rows")


def tscv_linear(df, n_splits=5):
    print("\n🔹 Running TimeSeriesSplit — Linear Regression...")
    df = df.copy()
    df["t"] = np.arange(len(df))
    X = df[["t", "promo", "weather", "is_holiday"]].values
    y = df["y"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(tscv.split(X), 1):
        model = LinearRegression().fit(X[tr], y[tr])
        pred = model.predict(X[te])
        mae = mean_absolute_error(y[te], pred)
        rmse = np.sqrt(mean_squared_error(y[te], pred))
        rows.append({"Fold": fold, "MAE": mae, "RMSE": rmse})
        print(f"  Fold {fold}: MAE={mae:.2f}  RMSE={rmse:.2f}")
    return pd.DataFrame(rows)


def tscv_prophet(df, n_splits=3):     # reduced from 4 → 3 (faster)
    print("\n🔹 Running TimeSeriesSplit — Prophet (this takes 1-2 min)...")
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(tscv.split(df), 1):
        print(f"  🔄 Training fold {fold}...", flush=True)
        train, test = df.iloc[tr], df.iloc[te]

        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.add_regressor("promo")
        m.add_regressor("weather")
        m.add_regressor("is_holiday")

        import logging
        logging.getLogger("prophet").setLevel(logging.WARNING)
        logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

        m.fit(train[["ds", "y", "promo", "weather", "is_holiday"]])
        fc = m.predict(test[["ds", "promo", "weather", "is_holiday"]])

        mae = mean_absolute_error(test["y"], fc["yhat"])
        rmse = np.sqrt(mean_squared_error(test["y"], fc["yhat"]))
        rows.append({"Fold": fold, "MAE": mae, "RMSE": rmse})
        print(f"  ✅ Fold {fold}: MAE={mae:.2f}  RMSE={rmse:.2f}", flush=True)
    return pd.DataFrame(rows)


# ---------- Linear ----------
lin_cv = tscv_linear(df)
print(f"\n📊 Linear CV — Mean MAE: {lin_cv['MAE'].mean():.2f}")

# ---------- Prophet ----------
proph_cv = tscv_prophet(df)
print(f"\n📊 Prophet CV — Mean MAE: {proph_cv['MAE'].mean():.2f}")

# ---------- Save results ----------
lin_cv.to_csv("cv_linear.csv", index=False)
proph_cv.to_csv("cv_prophet.csv", index=False)
print("\n✅ Saved cv_linear.csv and cv_prophet.csv")

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(lin_cv["Fold"] - 0.2, lin_cv["MAE"], width=0.4, label="Linear", color="steelblue")
ax.bar(proph_cv["Fold"] + 0.2, proph_cv["MAE"], width=0.4, label="Prophet", color="orange")
ax.set_xlabel("Fold")
ax.set_ylabel("MAE")
ax.set_title("TimeSeriesSplit Cross-Validation MAE")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("cross_validation.png", dpi=120)
print("✅ Saved cross_validation.png")

print("\n🎉 DONE!")