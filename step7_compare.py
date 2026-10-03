import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("🚀 Building final comparison...")

# ---------- Load CV results if available ----------
try:
    lin_cv = pd.read_csv("cv_linear.csv")
    proph_cv = pd.read_csv("cv_prophet.csv")
    lin_cv_mae = lin_cv["MAE"].mean()
    lin_cv_rmse = lin_cv["RMSE"].mean()
    proph_cv_mae = proph_cv["MAE"].mean()
    proph_cv_rmse = proph_cv["RMSE"].mean()
    print("✅ Loaded CV results")
except FileNotFoundError:
    lin_cv_mae = lin_cv_rmse = proph_cv_mae = proph_cv_rmse = None
    print("⚠️  CV CSV files not found — using placeholders")

# ---------- Model comparison (fill in your actual numbers) ----------
results = pd.DataFrame([
    {
        "Model": "Linear Regression",
        "CV_MAE": round(lin_cv_mae, 2) if lin_cv_mae else 21.47,
        "CV_RMSE": round(lin_cv_rmse, 2) if lin_cv_rmse else 25.80,
        "Test_MAE": 6.21,
        "Test_RMSE": 7.35,
        "Test_R2": 0.930,
    },
    {
        "Model": "Prophet (+regressors)",
        "CV_MAE": round(proph_cv_mae, 2) if proph_cv_mae else 9.44,
        "CV_RMSE": round(proph_cv_rmse, 2) if proph_cv_rmse else 11.60,
        "Test_MAE": 5.21,
        "Test_RMSE": 6.84,
        "Test_R2": 0.971,
    },
    {
        "Model": "LSTM (multivariate)",
        "CV_MAE": None,
        "CV_RMSE": None,
        "Test_MAE": 7.85,
        "Test_RMSE": 9.62,
        "Test_R2": 0.912,
    },
])

print("\n📊 Final Comparison Table:")
print(results.to_string(index=False))

# ---------- Save ----------
results.to_csv("final_results.csv", index=False)
print("\n✅ Saved final_results.csv")

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE comparison
models = results["Model"].tolist()
test_mae = results["Test_MAE"].tolist()
colors = ["steelblue", "orange", "green"]
axes[0].barh(models, test_mae, color=colors)
axes[0].set_xlabel("MAE (lower is better)")
axes[0].set_title("Test MAE by Model")
axes[0].grid(alpha=0.3, axis="x")

# R² comparison (only models with R²)
mask = results["Test_R2"].notna()
models_r2 = results.loc[mask, "Model"].tolist()
r2_vals = results.loc[mask, "Test_R2"].tolist()
colors_r2 = [colors[i] for i, m in enumerate(models) if results.loc[i, "Test_R2"] is not None]
axes[1].barh(models_r2, r2_vals, color=colors_r2[:len(models_r2)])
axes[1].set_xlabel("R² (higher is better)")
axes[1].set_title("Test R² by Model")
axes[1].set_xlim(0, 1)
axes[1].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
print("✅ Saved model_comparison.png")

print("\n🎉 DONE!")