import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

tf.get_logger().setLevel("ERROR")
tf.random.set_seed(42)
np.random.seed(42)

# ---------- Load ----------
df = pd.read_csv("historical_data_clean.csv", parse_dates=["ds"])
features = ["y", "promo", "weather", "is_holiday"]
data = df[features].values

# ---------- Scale ----------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# ---------- Sequences ----------
WINDOW = 6   # 6 months lookback (better for small data)

def make_sequences(arr, window):
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i + window])
        y.append(arr[i + window, 0])
    return np.array(X), np.array(y)

X, y = make_sequences(scaled, WINDOW)
print(f"Total sequences: {len(X)}")

# ---------- Split ----------
split = len(X) - 12
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# ---------- Model (simpler, less overfit) ----------
model = Sequential([
    Input(shape=(WINDOW, len(features))),
    LSTM(32, return_sequences=False),   # single layer
    Dropout(0.1),
    Dense(16, activation="relu"),
    Dense(1),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
              loss="mse", metrics=["mae"])
model.summary()

# ---------- Callbacks ----------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", patience=8, factor=0.5, min_lr=1e-5),
]

# ---------- Train ----------
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=16,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=0,
)

# ---------- Predict ----------
y_pred_scaled = model.predict(X_test, verbose=0)

def inverse_y(scaled_y):
    pad = np.zeros((len(scaled_y), len(features)))
    pad[:, 0] = scaled_y.flatten()
    return scaler.inverse_transform(pad)[:, 0]

y_pred = inverse_y(y_pred_scaled)
y_true = inverse_y(y_test)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print("\n📊 LSTM Performance:")
print(f"  MAE  : {mae:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.3f}")

# ---------- Plot loss ----------
plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Validation")
plt.title("LSTM Training Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("lstm_loss.png", dpi=120)
plt.show()

# ---------- Plot forecast ----------
plt.figure(figsize=(12, 5))
plt.plot(df["ds"].iloc[-12:], y_true, marker="o", label="Actual", linewidth=2)
plt.plot(df["ds"].iloc[-12:], y_pred, marker="x", label="LSTM Pred", linewidth=2)
plt.title("LSTM Forecast vs Actual (last 12 months)")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig("lstm_forecast.png", dpi=120)
plt.show()

print("✅ Saved: lstm_loss.png, lstm_forecast.png")