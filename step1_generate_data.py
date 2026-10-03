import numpy as np
import pandas as pd

np.random.seed(42)

# 4 years of monthly data
dates = pd.date_range(start="2015-01-01", end="2024-12-01", freq="MS")  # 120 months
n = len(dates)

# Building blocks of our signal
trend = np.linspace(100, 320, n)                              # upward growth
seasonality = 35 * np.sin(np.arange(n) * 2 * np.pi / 12)     # yearly cycle
noise = np.random.normal(0, 10, n)                            # random noise

# External features
holiday_boost = np.where(pd.Series(dates).dt.month.isin([11, 12]), 25, 0)
promo = np.random.choice([0, 1], size=n, p=[0.7, 0.3]) * 20
weather_idx = 15 * np.sin(np.arange(n) * 2 * np.pi / 6) + np.random.normal(0, 3, n)

# Combine everything
sales = trend + seasonality + holiday_boost + promo + weather_idx + noise

df = pd.DataFrame({
    "ds": dates,
    "y": sales,
    "promo": promo,
    "weather": weather_idx,
    "is_holiday": (holiday_boost > 0).astype(int),
})

df.to_csv("historical_data.csv", index=False)
print("✅ Saved historical_data.csv")
print(df.head())
print(f"\nTotal rows: {len(df)}")