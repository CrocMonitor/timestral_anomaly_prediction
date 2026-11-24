import time
import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster
from logger_data import Session, validate

# ==========================================
# 1. Configuration & Parameters
# ==========================================
XRK_FILE = "lambros_orange_aragon_generic testing_a_0300.xrk"
OUTPUT_CSV = "timestral_predictions_all_horizons_aragon_no_context.csv"

SAMPLE_RATE = 20  # Hz
TARGET_CHANNEL = "GPS Speed"
PAST_CONTEXT_CHANNELS = []

MAX_HORIZON_SECONDS = 120.0
MAX_HORIZON = int(MAX_HORIZON_SECONDS * SAMPLE_RATE)  # 2400 steps

MODEL_NAME = "mistralai/Timestral-5M"
MAX_CONTEXT_LENGTH = 32768
WARMUP_STEPS = 2048
STEP_STRIDE = 20  # Evaluate once every 1.0 second

# ==========================================
# 2. Load Session Data & Validate
# ==========================================
print(f"Loading session from {XRK_FILE} at {SAMPLE_RATE} Hz...")
session = Session(XRK_FILE, rate=SAMPLE_RATE)
validate(session)

all_channels = [TARGET_CHANNEL] + PAST_CONTEXT_CHANNELS
missing = [c for c in all_channels if c not in session.data.columns]
if missing:
    raise ValueError(f"Channels not found: {missing}\nAvailable: {list(session.data.columns)}")

df = session.data[all_channels].dropna()
data_matrix = df[all_channels].to_numpy().T
total_steps = data_matrix.shape[1]
target_idx = 0
target_unit = session.units.get(TARGET_CHANNEL, "")

if total_steps <= WARMUP_STEPS + MAX_HORIZON:
    raise ValueError(f"Session too short ({total_steps} steps) for WARMUP_STEPS and MAX_HORIZON.")

# ==========================================
# 3. Initialize Timestral Model
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading {MODEL_NAME} on {device}...")
forecaster = TimestralForecaster.from_pretrained(MODEL_NAME, device=device)

# ==========================================
# 4. Closed Inference & Collect Results for ALL Horizons
# ==========================================
start_idx = WARMUP_STEPS
end_idx = total_steps - MAX_HORIZON
records = []

print(f"Running closed inference across all horizons (1 to {MAX_HORIZON} steps / {MAX_HORIZON_SECONDS}s) with stride {STEP_STRIDE}...")
for anchor in range(start_idx, end_idx, STEP_STRIDE):
    start_history = max(0, anchor - MAX_CONTEXT_LENGTH)
    context = data_matrix[:, start_history:anchor]

    # Measure inference execution time for this anchor step
    t_start = time.perf_counter()
    result = forecaster.forecast(context, prediction_length=MAX_HORIZON)
    t_elapsed = time.perf_counter() - t_start

    pred_median = result.median
    if pred_median.ndim == 3:
        pred_median = pred_median[0]

    target_preds = pred_median[target_idx]
    anchor_time = df.index[anchor]
    future_timestamps = df.index[anchor : anchor + MAX_HORIZON]

    # Loop through every single forecast horizon step from 1 to MAX_HORIZON
    for h_step in range(1, MAX_HORIZON + 1):
        h_sec = h_step / SAMPLE_RATE
        records.append({
            "anchor_time_s": anchor_time,
            "target_time_s": future_timestamps[h_step - 1],
            "horizon_steps": h_step,
            "horizon_seconds": h_sec,
            "target_channel": TARGET_CHANNEL,
            "unit": target_unit,
            "actual": float(data_matrix[target_idx, anchor + h_step - 1]),
            "predicted": float(target_preds[h_step - 1]),
            "inference_time_s": float(t_elapsed),  # Track how long the forward pass took
        })

# ==========================================
# 5. Compute MAE, Merge into DataFrame, & Export
# ==========================================
results_df = pd.DataFrame(records)
results_df["abs_error"] = (results_df["actual"] - results_df["predicted"]).abs()

# Calculate MAE per horizon and merge it back
mae_summary = (
    results_df.groupby(["horizon_seconds", "horizon_steps"], as_index=False)["abs_error"]
    .mean()
    .rename(columns={"abs_error": "horizon_mae"})
)
results_df = results_df.merge(mae_summary, on=["horizon_seconds", "horizon_steps"], how="left")

# Save to CSV
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(results_df)} prediction records (including inference time and horizon MAE) to '{OUTPUT_CSV}'.")