import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load
df = pd.read_csv("historical_data.csv", parse_dates=["ds"])
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Fill missing (if any)
df["y"] = df["y"].interpolate(method="linear")

# Outlier clipping using IQR
Q1, Q3 = df["y"].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df["y"] = df["y"].clip(lower, upper)

# Save cleaned version
df.to_csv("historical_data_clean.csv", index=False)

# Visualize
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(df["ds"], df["y"], marker="o", color="steelblue")
axes[0].set_title("Historical Sales Trend")
axes[0].set_ylabel("Sales")
axes[0].grid(alpha=0.3)

axes[1].scatter(df["promo"], df["y"], alpha=0.6, label="Promo")
axes[1].scatter(df["weather"], df["y"], alpha=0.4, label="Weather", color="orange")
axes[1].set_title("Sales vs External Features")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("exploration.png", dpi=120)
plt.show()
print("✅ Saved exploration.png and historical_data_clean.csv")