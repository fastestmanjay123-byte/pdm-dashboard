# app.py — Streamlit PdM Dashboard (fast, cached, no Matplotlib)

import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------- Page setup ----------
st.set_page_config(page_title="Predictive Maintenance Dashboard (Mock-up)", layout="wide")
st.title("Predictive Maintenance Dashboard (Mock-up)")
st.caption("Demonstration using simulated/replayed data. Not a live deployment.")

# ---------- Caching ----------
@st.cache_data(show_spinner=False)
def find_unified_dataset_cached(cwd_files):
    candidates = [
        "unified_sensor_Dataset 2.csv",
        "unified_sensor_Dataset 2.CSV",
        "Unified_Sensor_Dataset 2.csv",
        "unified_sensor_Dataset_2.csv",
        "unified_sensor_dataset_2.csv",
        "unified_sensor_dataset.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for f in cwd_files:
        if f.lower().endswith(".csv") and "unified" in f.lower() and "sensor" in f.lower():
            return f
    return None

@st.cache_data(show_spinner=False)
def safe_read_csv(path_or_buf, nrows=None, encoding=None):
    try:
        return pd.read_csv(path_or_buf, nrows=nrows, low_memory=False, encoding=encoding)
    except UnicodeDecodeError:
        return pd.read_csv(path_or_buf, nrows=nrows, low_memory=False, encoding="latin-1")

@st.cache_data(show_spinner=False)
def load_demo():
    n = 400
    ts = pd.date_range("2025-05-13 00:00:00", periods=n, freq="3min")
    vib = 0.25 + 0.02*np.sin(np.linspace(0, 20, n)) + np.random.normal(0, 0.01, n)
    vib[300:] += np.linspace(0.0, 0.6, n-300)   # rising trend near failure
    temp = 40 + 0.5*np.sin(np.linspace(0, 8, n)) + (vib-0.25)*10
    rpm = np.full(n, 1750)
    label = np.zeros(n, dtype=int); label[330:380] = 1
    return pd.DataFrame({"timestamp": ts, "vib_rms": vib, "temp_c": temp, "rpm": rpm, "label": label})

# ---------- Small utils ----------
def normalize_0_1(x: pd.Series) -> pd.Series:
    rng = x.max() - x.min()
    return (x - x.min()) / (rng + 1e-9)

def demo_probability(vib_series: pd.Series) -> np.ndarray:
    v = normalize_0_1(vib_series)
    p = np.clip(v**2.0, 0, 1)
    return pd.Series(p).rolling(5, min_periods=1).mean().to_numpy()

def compute_confusion(y_true, y_pred, labels=(1, 0)):
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for i, a in enumerate(labels):
        for j, p in enumerate(labels):
            cm[i, j] = int(np.sum((y_true == a) & (y_pred == p)))
    return cm, [f"Failure ({labels[0]})", f"Normal ({labels[1]})"]

def decimate_for_chart(df, ts_col, max_points=5000):
    n = len(df)
    if n <= max_points:
        return df
    step = int(np.ceil(n / max_points))
    return df.iloc[::step, :]

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Data Source")
    use_repo_file = st.checkbox("Use repo file if present (e.g., 'unified_sensor_Dataset 2.csv')", value=False)
    uploaded = st.file_uploader("Or upload a CSV", type=["csv"])
    load_full = st.checkbox("Load full dataset (may be slow)", value=False)
    sample_rows = st.number_input("Sample N rows (when not loading full)", min_value=1000, max_value=200000, value=5000, step=1000)

    st.header("Visualization Settings")
    prob_threshold = st.slider("Alert Threshold (P(failure))", 0.05, 0.95, 0.50, 0.01)
    smooth_win = st.slider("Smoothing Window (points)", 1, 21, 5, 2)
    shade_fail = st.checkbox("Shade failure windows (label==1)", value=True)
    show_confusion = st.checkbox("Show confusion matrix (requires label)", value=True)

# ---------- Load data ----------
df = None
repo_path = None
cwd_files = tuple(os.listdir("."))  # hashable for cache
if use_repo_file:
    repo_path = find_unified_dataset_cached(cwd_files)

if uploaded is not None:
    df = safe_read_csv(uploaded, nrows=None if load_full else int(sample_rows))
elif repo_path is not None:
    st.info(f"Attempting to load: **{repo_path}** "
            f"({'full' if load_full else f'first {int(sample_rows):,} rows'})")
    df = safe_read_csv(repo_path, nrows=None if load_full else int(sample_rows))

if df is None:
    st.warning("Using built-in demo data for a fast start.")
    df = load_demo()

# ---------- Diagnostics ----------
st.caption(f"Python: {sys.version.split()[0]} • Rows loaded: {len(df):,} • Columns: {min(6, len(df.columns))} shown → {list(df.columns)[:6]}")

# ---------- Column Mapping ----------
st.subheader("Column Mapping")
cols = list(df.columns)

def guess(candidates):
    for c in cols:
        for key in candidates:
            if key.lower() in str(c).lower():
                return c
    return cols[0]

default_time = guess(["timestamp", "time", "datetime", "date"])
default_vib  = guess([
    "vibration rms - motor 1v (mm/sec)", "vibration rms - motor 1h", "vibration rms - motor 1a",
    "vib acc rms - motor 1v", "vibration", "vib", "rms"
])
default_temp = guess(["surface temperature - motor 1 (c)", "temperature", "temp"])
default_rpm  = guess(["motor speed - motor (hz)", "rpm", "speed", "hz", "frequency"])
default_label = None

c1, c2, c3 = st.columns(3)
with c1:
    time_col = st.selectbox("Timestamp column *", cols, index=cols.index(default_time) if default_time in cols else 0)
    vib_col  = st.selectbox("Vibration RMS column *", cols, index=cols.index(default_vib) if default_vib in cols else 0)
with c2:
    temp_col = st.selectbox("Temperature column (optional)", ["(none)"] + cols, index=(cols.index(default_temp)+1 if default_temp in cols else 0))
    rpm_col  = st.selectbox("RPM/Speed column (optional)", ["(none)"] + cols, index=(cols.index(default_rpm)+1 if default_rpm in cols else 0))
with c3:
    label_col = st.selectbox("Label column (0=normal, 1=failure, optional)", ["(none)"] + cols, index=0)

if time_col == "(none)" or vib_col == "(none)":
    st.error("Please select the required columns: Timestamp and Vibration.")
    st.stop()

temp_col  = None if temp_col  == "(none)" else temp_col
rpm_col   = None if rpm_col   == "(none)" else rpm_col
label_col = None if label_col == "(none)" else label_col

# ---------- Clean & features ----------
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
if smooth_win > 1:
    df["vib_s"] = df["vib_rms"].rolling(int(smooth_win), min_periods=1, center=True).mean()
else:
    df["vib_s"] = df["vib_rms"]

df["p_failure"] = demo_probability(df["vib_s"])
df["alert"] = (df["p_failure"] >= prob_threshold).astype(int)

# ---------- Status card ----------
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

# ---------- Layout ----------
left, right = st.columns([2,1], gap="large")

with left:
    st.subheader("Vibration Over Time with Alerts")
    plot_df = decimate_for_chart(df[["timestamp","vib_s","alert","p_failure"]].copy(), "timestamp", max_points=5000)
    base = alt.Chart(plot_df).mark_line().encode(
        x="timestamp:T",
        y=alt.Y("vib_s:Q", title="Vibration RMS"),
        tooltip=["timestamp:T", alt.Tooltip("vib_s:Q", format=".3f"), alt.Tooltip("p_failure:Q", format=".2f")]
    )
    points = alt.Chart(plot_df[plot_df["alert"]==1]).mark_point(size=18, opacity=0.9).encode(
        x="timestamp:T", y="vib_s:Q", tooltip=["timestamp:T", "vib_s:Q", alt.Tooltip("p_failure:Q", format=".2f")]
    )
    layer = base + points

    if label_col and "label" in df.columns and df["label"].sum() > 0:
        # compute contiguous label==1 spans and shade them
        spans, lab = [], df["label"].to_numpy()
        idx_start = None
        for i in range(len(lab)):
            if lab[i]==1 and idx_start is None:
                idx_start = i
            if lab[i]==0 and idx_start is not None:
                spans.append((df["timestamp"].iloc[idx_start], df["timestamp"].iloc[i-1])); idx_start=None
        if idx_start is not None:
            spans.append((df["timestamp"].iloc[idx_start], df["timestamp"].iloc[-1]))
        if spans:
            regions = pd.DataFrame({"start":[s for s,_ in spans], "end":[e for _,e in spans]})
            boxes = alt.Chart(regions).mark_rect(opacity=0.15).encode(x="start:T", x2="end:T")
            layer = boxes + layer

    st.altair_chart(layer.interactive(), use_container_width=True)

with right:
    st.subheader("Recent Samples")
    view_cols = ["timestamp", "vib_s", "p_failure", "alert"]
    if "temp_c" in df.columns: view_cols.insert(2, "temp_c")
    if "rpm"   in df.columns: view_cols.insert(3, "rpm")
    st.dataframe(df[view_cols].tail(12), use_container_width=True)

if show_confusion and label_col and "label" in df.columns:
    st.subheader("Confusion Matrix (if ground-truth labels provided)")
    y_true = df["label"].to_numpy(dtype=int)
    y_pred = df["alert"].to_numpy(dtype=int)
    cm, labels = compute_confusion(y_true, y_pred, labels=(1,0))
    cm_df = pd.DataFrame(cm, index=["Failure (1)","Normal (0)"], columns=["Failure (1)","Normal (0)"]).reset_index().melt("index")
    cm_df.columns = ["Actual","Predicted","Count"]
    heat = alt.Chart(cm_df).mark_rect().encode(x="Predicted:N", y="Actual:N", color="Count:Q")
    text = alt.Chart(cm_df).mark_text(fontWeight="bold").encode(x="Predicted:N", y="Actual:N", text="Count:Q")
    st.altair_chart((heat + text), use_container_width=True)
else:
    st.caption("Map a `label` column (0/1) to enable confusion matrix and shaded failure windows.")

st.caption("Tip: Use the chart menu to download images for your manuscript.")
