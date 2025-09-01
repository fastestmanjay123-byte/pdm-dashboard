# app.py — Streamlit PdM Dashboard (Mock-up) — fast start, CSV-safe, no scikit-learn

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- Page setup ----------------
st.set_page_config(page_title="Predictive Maintenance Dashboard (Mock-up)", layout="wide")
st.title("Predictive Maintenance Dashboard (Mock-up)")
st.caption("Demonstration using simulated/replayed data. Not a live deployment.")

# ---------------- Helpers ----------------
def find_unified_dataset():
    """
    Try to locate your repo CSV by common names and a fuzzy fallback.
    """
    candidates = [
        "unified_sensor_Dataset 2.csv",
        "unified_sensor_Dataset 2.CSV",
        "Unified_Sensor_Dataset 2.csv",
        "unified_sensor_Dataset_2.csv",
        # common alternates people try
        "unified_sensor_dataset_2.csv",
        "unified_sensor_dataset.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for f in os.listdir("."):
        if f.lower().endswith(".csv") and "unified" in f.lower() and "sensor" in f.lower():
            return f
    return None

def safe_read_csv(f, nrows=None):
    """
    Read CSV robustly and quickly.
    - nrows: pass small int (e.g., 5000) for fast first render
    """
    # Try fast path
    try:
        return pd.read_csv(f, nrows=nrows, low_memory=False)
    except UnicodeDecodeError:
        # Fallback encoding
        return pd.read_csv(f, nrows=nrows, encoding="latin-1", low_memory=False)

def load_demo():
    """Small synthetic dataset (3-min cadence) that behaves like your scenario."""
    n = 400
    ts = pd.date_range("2025-05-13 00:00:00", periods=n, freq="3min")
    vib = 0.25 + 0.02*np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.01, n)
    vib[300:] += np.linspace(0.0, 0.6, n-300)   # rise near failure
    temp = 40 + 0.5*np.sin(np.linspace(0, 8, n)) + (vib-0.25)*10
    rpm = np.full(n, 1750)
    label = np.zeros(n, dtype=int); label[330:380] = 1
    return pd.DataFrame({"timestamp": ts, "vib_rms": vib, "temp_c": temp, "rpm": rpm, "label": label})

def normalize_0_1(x: pd.Series) -> pd.Series:
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-9)

def demo_probability(vib_series: pd.Series) -> np.ndarray:
    """Simple, plausible probability proxy from vibration magnitude (replace with real model later)."""
    v = normalize_0_1(vib_series)
    p = np.clip(v**2.0, 0, 1)
    return pd.Series(p).rolling(5, min_periods=1).mean().to_numpy()

def shade_failure_windows(ax, df, label_col):
    if label_col not in df.columns or df[label_col].sum() == 0:
        return
    fail = df[label_col].values
    starts = np.where((fail[1:] == 1) & (fail[:-1] == 0))[0] + 1
    ends = np.where((fail[1:] == 0) & (fail[:-1] == 1))[0] + 1
    if fail[0] == 1: starts = np.r_[0, starts]
    if fail[-1] == 1: ends = np.r_[ends, len(fail)]
    for s, e in zip(starts, ends):
        ax.axvspan(df["timestamp"].iloc[s], df["timestamp"].iloc[e-1], alpha=0.12, label="Failure window")

def compute_confusion_matrix(y_true, y_pred, labels=(1, 0)):
    """Tiny NumPy confusion matrix (rows=actual, cols=pred)."""
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, a in enumerate(labels):
        for j, p in enumerate(labels):
            cm[i, j] = int(np.sum((y_true == a) & (y_pred == p)))
    return cm, [f"Failure ({labels[0]})", f"Normal ({labels[1]})"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Data Source")
    # Default OFF so the app renders instantly on demo data
    use_repo_file = st.checkbox(
        "Use repo file if present (e.g., 'unified_sensor_Dataset 2.csv')",
        value=False
    )
    uploaded = st.file_uploader("Or upload a CSV", type=["csv"])
    load_full = st.checkbox("Load full dataset (may be slow)", value=False)
    sample_rows = st.number_input("Sample N rows (when not loading full)", min_value=1000, max_value=200000, value=5000, step=1000)

    st.header("Visualization Settings")
    prob_threshold = st.slider("Alert threshold (P(failure))", 0.05, 0.95, 0.50, 0.01)
    smooth_win = st.slider("Smoothing window (points)", 1, 21, 5, 2)
    show_failure_window = st.checkbox("Shade failure windows (label==1)", value=True)
    show_confusion = st.checkbox("Show confusion matrix (requires label)", value=True)

# ---------------- Load data ----------------
df = None
repo_path = find_unified_dataset() if use_repo_file else None

if uploaded is not None:
    df = safe_read_csv(uploaded, nrows=None if load_full else int(sample_rows))
elif repo_path is not None:
    st.info(f"Attempting to load: **{repo_path}** "
            f"({'full' if load_full else f'first {int(sample_rows):,} rows'})")
    df = safe_read_csv(repo_path, nrows=None if load_full else int(sample_rows))

if df is None:
    st.warning("Using built-in demo data for a fast start.")
    df = load_demo()

# ---------------- Diagnostics (helpful if anything seems 'stuck') ----------------
st.caption(f"Python: {sys.version.split()[0]} • Rows loaded: {len(df):,} • Columns: {min(6, len(df.columns))} shown → {list(df.columns)[:6]}")

# ---------------- Column mapping ----------------
st.subheader("Column Mapping")

cols = list(df.columns)

def guess(name_options):
    for n in cols:
        for opt in name_options:
            if opt.lower() in str(n).lower():
                return n
    return cols[0]

# Include your real headers in the guesses to make first-time selection easier
default_time  = guess(["timestamp", "time", "datetime", "date"])
default_vib   = guess([
    "vibration rms - motor 1v (mm/sec)",
    "vibration rms - motor 1h (mm/sec)",
    "vibration rms - motor 1a (mm/sec)",
    "vib acc rms - motor 1v (g)",
    "vib acc rms - motor 1h (g)",
    "vib acc rms - motor 1a (g)",
    "vib", "rms", "acceleration", "vibration"
])
default_temp  = guess(["surface temperature - motor 1 (c)", "temperature", "temp"])
default_rpm   = guess(["motor speed - motor (hz)", "rpm", "speed", "frequency", "hz"])
# No default label unless present
default_label = None

c1, c2, c3 = st.columns(3)
with c1:
    time_col = st.selectbox("Timestamp column *", options=cols,
                            index=cols.index(default_time) if default_time in cols else 0)
    vib_col  = st.selectbox("Vibration RMS column *", options=cols,
                            index=cols.index(default_vib) if default_vib in cols else 0)
with c2:
    temp_col = st.selectbox("Temperature column (optional)", options=["(none)"] + cols,
                            index=(cols.index(default_temp)+1 if default_temp in cols else 0))
    rpm_col  = st.selectbox("RPM/Speed column (optional)", options=["(none)"] + cols,
                            index=(cols.index(default_rpm)+1 if default_rpm in cols else 0))
with c3:
    label_col = st.selectbox("Label column (0=normal,1=failure, optional)",
                             options=["(none)"] + cols,
                             index=(cols.index(default_label)+1 if (default_label and default_label in cols) else 0))

if time_col is None or vib_col is None or time_col == "(none)" or vib_col == "(none)":
    st.error("Please select the required columns: Timestamp and Vibration.")
    st.stop()

temp_col  = None if temp_col  == "(none)" else temp_col
rpm_col   = None if rpm_col   == "(none)" else rpm_col
label_col = None if label_col == "(none)" else label_col

# ---------------- Clean & prepare ----------------
try:
    df["timestamp"] = pd.to_datetime(df[time_col])
except Exception as e:
    st.error(f"Could not parse timestamps from '{time_col}': {e}")
    st.stop()

df["vib_rms"] = pd.to_numeric(df[vib_col], errors="coerce")
if temp_col:
    df["temp_c"] = pd.to_numeric(df[temp_col], errors="coerce")
if rpm_col:
    df["rpm"] = pd.to_numeric(df[rpm_col], errors="coerce")
if label_col:
    df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

df = df.dropna(subset=["timestamp", "vib_rms"]).sort_values("timestamp").reset_index(drop=True)
df["vib_rms_s"] = df["vib_rms"].rolling(int(smooth_win), min_periods=1, center=True).mean()

# ---------------- “Model” probability & alerts ----------------
df["p_failure"] = demo_probability(df["vib_rms_s"])
df["alert"] = (df["p_failure"] >= prob_threshold).astype(int)

# ---------------- Status card ----------------
latest_p = float(df["p_failure"].iloc[-1])
status_text = "FAILURE RISK" if latest_p >= prob_threshold else "NORMAL"
status_color = "#e74c3c" if status_text == "FAILURE RISK" else "#2ecc71"
st.markdown(
    f"""
    <div style="padding:14px;border-radius:10px;background:{status_color};color:white;
                font-weight:600;font-size:20px;text-align:center;">
        Equipment Health: {status_text} • P(failure)={latest_p:.2f} • Threshold={prob_threshold:.2f}
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Charts ----------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Vibration Over Time with Alerts")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["vib_rms_s"], label="Vibration RMS (smoothed)")
    idx = df["alert"] == 1
    if idx.any():
        ax.scatter(df.loc[idx, "timestamp"], df.loc[idx, "vib_rms_s"], s=15, label="Alerts", zorder=3)
    if show_failure_window and label_col and "label" in df.columns and df["label"].sum() > 0:
        shade_failure_windows(ax, df, "label")
    ax.set_xlabel("Time")
    ax.set_ylabel("Vibration RMS")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)

with col2:
    st.subheader("Recent Samples")
    view_cols = ["timestamp", "vib_rms_s", "p_failure", "alert"]
    if "temp_c" in df.columns: view_cols.insert(2, "temp_c")
    if "rpm"   in df.columns: view_cols.insert(3, "rpm")
    st.dataframe(df[view_cols].tail(12), use_container_width=True)

# ---------------- Confusion matrix (NumPy) ----------------
if show_confusion and label_col and "label" in df.columns:
    st.subheader("Confusion Matrix (if ground-truth labels provided)")
    y_true = df["label"].to_numpy(dtype=int)
    y_pred = df["alert"].to_numpy(dtype=int)
    cm, labels = compute_confusion_matrix(y_true, y_pred, labels=(1, 0))

    fig_cm, ax_cm = plt.subplots(figsize=(4.8, 4.2))
    im = ax_cm.imshow(cm, interpolation="nearest")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(labels, rotation=20, ha="right"); ax_cm.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, cm[i, j], ha="center", va="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black")
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
    fig_cm.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    st.pyplot(fig_cm, clear_figure=True)
else:
    st.caption("Upload or map a `label` column (0/1) to enable confusion matrix and shading.")

st.caption("Tip: Use the chart ⋮ menu to download PNGs for your manuscript.")
