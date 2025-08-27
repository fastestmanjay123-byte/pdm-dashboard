# app.py — Streamlit PdM Dashboard (Mock-up) for "unified_sensor_Dataset 2"
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------- Page setup ----------------
st.set_page_config(page_title="Predictive Maintenance Dashboard (Mock-up)", layout="wide")
st.title("Predictive Maintenance Dashboard (Mock-up)")
st.caption("This is a demonstration using simulated/replayed data. It is not a live deployment.")

# ---------------- Helpers ----------------
def find_unified_dataset():
    """
    Search the repo for the dataset named 'unified_sensor_Dataset 2.csv'.
    Returns a path or None if not found.
    """
    candidates = [
        "unified_sensor_Dataset 2.csv",
        "unified_sensor_Dataset 2.CSV",
        "Unified_Sensor_Dataset 2.csv",
        "unified_sensor_Dataset_2.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # fallback: scan current dir for similarly named file
    for f in os.listdir("."):
        if f.lower().endswith(".csv") and "unified" in f.lower() and "sensor" in f.lower():
            return f
    return None

def load_demo():
    """Small synthetic dataset if nothing is provided."""
    n = 400
    ts = pd.date_range("2025-05-13 00:00:00", periods=n, freq="3min")
    vib = 0.25 + 0.02*np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.01, n)
    vib[300:] += np.linspace(0.0, 0.6, n-300)  # rising vibration toward failure
    temp = 40 + 0.5*np.sin(np.linspace(0, 8, n)) + (vib-0.25)*10
    rpm = np.full(n, 1750)
    label = np.zeros(n, dtype=int); label[330:380] = 1
    return pd.DataFrame({"timestamp": ts, "vib_rms": vib, "temp_c": temp, "rpm": rpm, "label": label})

def normalize_0_1(x: pd.Series) -> pd.Series:
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-9)

def demo_probability(vib_series: pd.Series) -> np.ndarray:
    """A simple, plausible 'probability' based on vibration magnitude (replace with real model later)."""
    v = normalize_0_1(vib_series)
    p = np.clip(v**2.0, 0, 1)
    return pd.Series(p).rolling(5, min_periods=1).mean().to_numpy()

def shade_failure_windows(ax, df, label_col):
    """Shade contiguous regions where label==1."""
    if label_col not in df.columns:
        return
    if df[label_col].sum() == 0:
        return
    fail = df[label_col].values
    starts = np.where((fail[1:] == 1) & (fail[:-1] == 0))[0] + 1
    ends = np.where((fail[1:] == 0) & (fail[:-1] == 1))[0] + 1
    if fail[0] == 1: starts = np.r_[0, starts]
    if fail[-1] == 1: ends = np.r_[ends, len(fail)]
    for s, e in zip(starts, ends):
        ax.axvspan(df["timestamp"].iloc[s], df["timestamp"].iloc[e-1], color="red", alpha=0.12, label="Failure window")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Data Source")
    use_repo_file = st.checkbox("Use repo file: 'unified_sensor_Dataset 2.csv' (if present)", value=True)
    uploaded = st.file_uploader("Or upload a CSV", type=["csv"])

    st.header("Visualization Settings")
    prob_threshold = st.slider("Alert threshold (P(failure))", 0.05, 0.95, 0.50, 0.01)
    smooth_win = st.slider("Smoothing window (points)", 1, 21, 5, 2)
    show_failure_window = st.checkbox("Shade failure windows (label==1)", value=True)
    show_confusion = st.checkbox("Show confusion matrix (requires label)", value=True)

# ---------------- Load Data ----------------
df = None
repo_path = find_unified_dataset() if use_repo_file else None

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
elif repo_path is not None:
    try:
        df = pd.read_csv(repo_path)
        st.info(f"Loaded dataset from repository: **{repo_path}**")
    except Exception as e:
        st.error(f"Found '{repo_path}' but failed to read it: {e}")

if df is None:
    st.warning("Using built-in demo data (add or upload your 'unified_sensor_Dataset 2.csv' to replace this).")
    df = load_demo()

# ---------------- Column Mapping ----------------
st.subheader("Column Mapping")
cols = list(df.columns)

# Best-guess defaults
def guess(name_options):
    for n in cols:
        for opt in name_options:
            if opt.lower() in n.lower():
                return n
    return None

default_time   = guess(["timestamp", "time", "datetime", "date"])
default_vib    = guess(["vib", "rms", "acc", "vibration"])
default_temp   = guess(["temp", "temperature"])
default_rpm    = guess(["rpm", "speed", "hz", "frequency"])
default_label  = guess(["label", "failure", "anomaly", "target"])

c1, c2, c3 = st.columns(3)
with c1:
    time_col  = st.selectbox("Timestamp column *", options=cols, index=cols.index(default_time) if default_time in cols else 0)
    vib_col   = st.selectbox("Vibration RMS column *", options=cols, index=cols.index(default_vib) if default_vib in cols else 0)
with c2:
    temp_col  = st.selectbox("Temperature column (optional)", options=["(none)"] + cols, index=(cols.index(default_temp)+1 if default_temp in cols else 0))
    rpm_col   = st.selectbox("RPM/Speed column (optional)", options=["(none)"] + cols, index=(cols.index(default_rpm)+1 if default_rpm in cols else 0))
with c3:
    label_col = st.selectbox("Label column (0=normal,1=failure, optional)", options=["(none)"] + cols, index=(cols.index(default_label)+1 if default_label in cols else 0))

required_missing = []
for req, name in [("Timestamp", time_col), ("Vibration", vib_col)]:
    if name is None or name == "(none)":
        required_missing.append(req)
if required_missing:
    st.error(f"Please select required columns: {', '.join(required_missing)}.")
    st.stop()

temp_col = None if temp_col == "(none)" else temp_col
rpm_col  = None if rpm_col  == "(none)" else rpm_col
label_col = None if label_col == "(none)" else label_col

# ---------------- Cleaning & Sorting ----------------
try:
    df["timestamp"] = pd.to_datetime(df[time_col])
except Exception as e:
    st.error(f"Could not parse timestamps from '{time_col}': {e}")
    st.stop()

# Cast vibration to numeric
df["vib_rms"] = pd.to_numeric(df[vib_col], errors="coerce")
if temp_col:
    df["temp_c"] = pd.to_numeric(df[temp_col], errors="coerce")
if rpm_col:
    df["rpm"] = pd.to_numeric(df[rpm_col], errors="coerce")
if label_col:
    df["label"] = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int)

df = df.sort_values("timestamp").reset_index(drop=True)

# Smoothing for visualization
if smooth_win > 1:
    df["vib_rms_s"] = df["vib_rms"].rolling(smooth_win, min_periods=1, center=True).mean()
else:
    df["vib_rms_s"] = df["vib_rms"]

# ---------------- “Model” Probability & Alerts ----------------
df["p_failure"] = demo_probability(df["vib_rms_s"])
df["alert"] = (df["p_failure"] >= prob_threshold).astype(int)

# ---------------- Status Card ----------------
latest_p = float(df["p_failure"].iloc[-1])
status_text = "FAILURE RISK" if latest_p >= prob_threshold else "NORMAL"
status_color = "#e74c3c" if status_text == "FAILURE RISK" else "#2ecc71"
st.markdown(
    f"""
    <div style="padding:14px;border-radius:10px;background:{status_color};color:white;
                font-weight:600;font-size:20px;text-align:center;">
        Equipment Health: {status_text}  •  P(failure)={latest_p:.2f}
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- Main Layout ----------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.subheader("Vibration Over Time with Alerts")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["timestamp"], df["vib_rms_s"], label="Vibration RMS (smoothed)")
    # Alert markers
    idx = df["alert"] == 1
    if idx.any():
        ax.scatter(df.loc[idx, "timestamp"], df.loc[idx, "vib_rms_s"], s=15, label="Alerts", zorder=3)
    # Shade failure windows
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
    if "rpm" in df.columns: view_cols.insert(3, "rpm")
    st.dataframe(df[view_cols].tail(12), use_container_width=True)

# ---------------- Confusion Matrix ----------------
if show_confusion and label_col and "label" in df.columns:
    st.subheader("Confusion Matrix (if ground-truth labels provided)")
    try:
        y_true = df["label"].astype(int).values
        y_pred = df["alert"].astype(int).values
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Failure (1)", "Normal (0)"])
        fig2, ax2 = plt.subplots(figsize=(4.8, 4.2))
        disp.plot(ax=ax2, colorbar=False)
        ax2.grid(False)
        st.pyplot(fig2, clear_figure=True)
    except Exception as e:
        st.info(f"Confusion matrix unavailable: {e}")
else:
    st.caption("Upload or map a label column (0/1) to enable confusion matrix.")

st.caption("Tip: Click the chart ⋮ menu to download images for your manuscript.")
