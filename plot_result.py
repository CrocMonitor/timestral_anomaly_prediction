import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. Configuration & Load CSV
# ==========================================
INPUT_CSV = "timestral_predictions_all_horizons_serres_no_context.csv"

# Specify which forecast horizons (in seconds) to plot for Actual vs. Forecasted.
# Set to None to plot every horizon present in the CSV.
HORIZONS_TO_PLOT_SECONDS = [5.0 ,10.0,20.0, 30.0, 60.0, 90.0, 120.0]

print(f"Reading predictions from '{INPUT_CSV}'...")
df = pd.read_csv(INPUT_CSV, on_bad_lines="warn")

if df.empty:
    raise ValueError("The predictions CSV file is empty.")

target_channel = df["target_channel"].iloc[0]
target_unit = df["unit"].iloc[0] if pd.notna(df["unit"].iloc[0]) else ""
unit_label = f" ({target_unit})" if target_unit else ""

# Compute MAE summary across all horizons in the CSV
mae_summary = (
    df.groupby(["horizon_seconds", "horizon_steps"], as_index=False)["abs_error"]
    .mean()
    .rename(columns={"abs_error": "mae"})
    .sort_values("horizon_seconds")
)

# ==========================================
# 2. Plot: Actual vs Forecasted (Selected Horizons)
# ==========================================
available_horizons = np.sort(df["horizon_seconds"].unique())

if HORIZONS_TO_PLOT_SECONDS is None:
    selected_horizons = available_horizons
else:
    selected_horizons = []
    for target_h in HORIZONS_TO_PLOT_SECONDS:
        matches = available_horizons[np.isclose(available_horizons, target_h, atol=1e-6)]
        if len(matches) == 0:
            print(f"Warning: Horizon {target_h}s not found in '{INPUT_CSV}', skipping.")
        else:
            selected_horizons.append(float(matches[0]))

for h_sec in selected_horizons:
    group = df[np.isclose(df["horizon_seconds"], h_sec, atol=1e-6)].sort_values("target_time_s")
    if group.empty:
        continue

    plt.figure(figsize=(12, 5))
    plt.plot(
        group["target_time_s"],
        group["actual"],
        label=f"Actual {target_channel}",
        color="black",
        alpha=0.75,
        linewidth=1.5,
    )
    plt.plot(
        group["target_time_s"],
        group["predicted"],
        label=f"Predicted ({h_sec:g}s Horizon)",
        color="red",
        linestyle="--",
        alpha=0.85,
        linewidth=1.5,
    )

    plt.xlabel("Time (s)")
    plt.ylabel(f"{target_channel}{unit_label}")
    plt.title(f"{target_channel} vs Time (Horizon: {h_sec:g}s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"forecast_horizon_{h_sec:g}s.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved plot: {filename}")

# ==========================================
# 3. Plot: MAE vs Forecast Horizon
# ==========================================
plt.figure(figsize=(8, 4.5))
plt.plot(mae_summary["horizon_seconds"], mae_summary["mae"], marker="o", color="blue", linewidth=2)
plt.xlabel("Forecast Horizon (seconds)")
plt.ylabel(f"MAE{unit_label}")
plt.title(f"MAE vs Forecast Horizon ({target_channel})")
plt.grid(True)
plt.tight_layout()

mae_filename = "mae_vs_horizon.png"
plt.savefig(mae_filename, dpi=300)
plt.close()
print(f"Saved plot: {mae_filename}")