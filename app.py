"""
NeuroLens — EEG Insight & Seizure Pattern Explorer
All code is self-contained in this single file to avoid import issues.
Run with: streamlit run app.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroLens", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0d1117; color: #e6edf3; }
  .main { background-color: #0d1117; }
  .block-container { padding: 1.5rem 2rem; max-width: 1200px; }
  .hero { padding: 2rem 0 1rem; border-bottom: 1px solid #21262d; margin-bottom: 1.5rem; }
  .hero h1 { font-family: 'IBM Plex Mono', monospace; font-size: 2.4rem; font-weight: 600; color: #e6edf3; margin: 0; letter-spacing: -1px; }
  .hero span { color: #58a6ff; }
  .hero p { color: #8b949e; font-size: 1rem; margin: 0.4rem 0 0; }
  .card { background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 1.2rem 1.4rem; margin-bottom: 1rem; }
  .card h4 { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; letter-spacing: 1.5px; text-transform: uppercase; color: #58a6ff; margin: 0 0 0.6rem; }
  .badge-normal { display: inline-block; background: rgba(63,185,80,0.15); border: 1px solid #3fb950; color: #3fb950; font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; padding: 0.4rem 1.2rem; border-radius: 6px; }
  .badge-epileptic { display: inline-block; background: rgba(248,81,73,0.15); border: 1px solid #f85149; color: #f85149; font-family: 'IBM Plex Mono', monospace; font-size: 1.1rem; font-weight: 600; padding: 0.4rem 1.2rem; border-radius: 6px; }
  .insight-item { background: #161b22; border-left: 3px solid #58a6ff; padding: 0.6rem 1rem; margin: 0.4rem 0; border-radius: 0 6px 6px 0; font-size: 0.88rem; color: #c9d1d9; line-height: 1.5; }
  .flag-warn { display: inline-block; background: rgba(248,81,73,0.1); border: 1px solid rgba(248,81,73,0.4); color: #f85149; font-size: 0.78rem; padding: 2px 10px; border-radius: 20px; margin: 3px; font-family: 'IBM Plex Mono', monospace; }
  .flag-ok { display: inline-block; background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.4); color: #3fb950; font-size: 0.78rem; padding: 2px 10px; border-radius: 20px; margin: 3px; font-family: 'IBM Plex Mono', monospace; }
  section[data-testid="stSidebar"] { background: #161b22 !important; border-right: 1px solid #21262d; }
  section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
  [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 1.4rem !important; color: #58a6ff !important; }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.8rem !important; }
  hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
# 1. EEG DATA GENERATION
# ══════════════════════════════════════════════════════════════════════

def generate_normal_eeg(duration=10, fs=256, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.random.normal(0, 10, len(t))
    signal += 5  * np.sin(2 * np.pi * 2  * t + np.random.uniform(0, 2*np.pi))
    signal += 8  * np.sin(2 * np.pi * 6  * t + np.random.uniform(0, 2*np.pi))
    signal += 20 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
    signal += 15 * np.sin(2 * np.pi * 11 * t + np.random.uniform(0, 2*np.pi))
    signal += 10 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2*np.pi))
    return t, signal.astype(np.float32)

def generate_epileptic_eeg(duration=10, fs=256, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t = np.linspace(0, duration, int(duration * fs))
    signal = np.random.normal(0, 20, len(t))
    signal += 40 * np.sin(2 * np.pi * 3 * t + np.random.uniform(0, 2*np.pi))
    signal += 30 * np.sin(2 * np.pi * 4 * t + np.random.uniform(0, 2*np.pi))
    signal += 25 * np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2*np.pi))
    spikes = np.zeros(len(t))
    for st_ in np.arange(0, duration, 1.0 / 3.0):
        idx = int(st_ * fs)
        if idx < len(t) - 20:
            spikes[idx:idx+5] += np.array([0, 60, 120, 60, 0]) * np.random.uniform(0.7, 1.3)
            wave_len = int(0.3 * fs)
            if idx + 5 + wave_len < len(t):
                spikes[idx+5:idx+5+wave_len] += -50 * np.sin(np.pi * np.arange(wave_len) / wave_len)
    signal += spikes
    for bp in np.random.choice(len(t) - int(0.5*fs), size=5, replace=False):
        burst_len = int(0.5 * fs)
        signal[bp:bp+burst_len] += np.random.normal(0, 50, burst_len) * np.hanning(burst_len)
    return t, signal.astype(np.float32)

# ══════════════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════

BANDS = {"Delta":(0.5,4), "Theta":(4,8), "Alpha":(8,13), "Beta":(13,30), "Gamma":(30,50)}

def compute_band_powers(sig, fs):
    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    band_powers = {}
    for band, (lo, hi) in BANDS.items():
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        band_powers[band] = float(np.trapezoid(psd[idx], freqs[idx]))
    return band_powers, freqs, psd

def compute_spike_rate(sig, fs, threshold_std=3.0):
    std, mean = np.std(sig), np.mean(sig)
    spike_idx = np.where(np.abs(sig - mean) > threshold_std * std)[0]
    if len(spike_idx) == 0:
        return 0.0, np.array([])
    gap = int(0.05 * fs)
    clusters, cluster = [], [spike_idx[0]]
    for idx in spike_idx[1:]:
        if idx - cluster[-1] <= gap:
            cluster.append(idx)
        else:
            clusters.append(cluster[0])
            cluster = [idx]
    clusters.append(cluster[0])
    return len(clusters) / (len(sig) / fs), np.array(clusters)

def compute_spectral_entropy(sig, fs):
    _, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(scipy_entropy(psd_norm + 1e-12))

def compute_hjorth_params(sig):
    diff1, diff2 = np.diff(sig), np.diff(np.diff(sig))
    activity = float(np.var(sig))
    mobility = float(np.sqrt(np.var(diff1) / (np.var(sig) + 1e-12)))
    complexity = float(np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12)) / (mobility + 1e-12))
    return activity, mobility, complexity

def extract_all_features(sig, fs):
    band_powers, freqs, psd = compute_band_powers(sig, fs)
    total_power = sum(band_powers.values()) + 1e-12
    rel = {k: v/total_power for k, v in band_powers.items()}
    spike_rate, spike_idx = compute_spike_rate(sig, fs)
    spec_entropy = compute_spectral_entropy(sig, fs)
    activity, mobility, complexity = compute_hjorth_params(sig)
    features = {
        "delta_power": band_powers["Delta"], "theta_power": band_powers["Theta"],
        "alpha_power": band_powers["Alpha"], "beta_power":  band_powers["Beta"],
        "gamma_power": band_powers["Gamma"],
        "delta_rel": rel["Delta"], "theta_rel": rel["Theta"],
        "alpha_rel": rel["Alpha"], "beta_rel":  rel["Beta"],
        "delta_alpha_ratio": band_powers["Delta"] / (band_powers["Alpha"] + 1e-12),
        "spike_rate": spike_rate, "spectral_entropy": spec_entropy,
        "hjorth_activity": activity, "hjorth_mobility": mobility,
        "hjorth_complexity": complexity, "variance": float(np.var(sig)),
    }
    fvec = np.array([
        features["delta_power"], features["theta_power"], features["alpha_power"],
        features["beta_power"],  features["delta_rel"],   features["theta_rel"],
        features["alpha_rel"],   features["delta_alpha_ratio"], features["spike_rate"],
        features["spectral_entropy"], features["hjorth_activity"],
        features["hjorth_mobility"],  features["hjorth_complexity"], features["variance"],
    ], dtype=np.float32)
    return features, fvec, freqs, psd, spike_idx

def interpret_features(features, prediction_prob):
    reasons, flags = [], []
    if features["delta_rel"] > 0.35:
        reasons.append(f"High delta-band dominance ({features['delta_rel']*100:.1f}%) — slow waves often seen in ictal/seizure activity")
        flags.append("⚠️ Elevated Delta")
    else:
        flags.append("✅ Normal Delta")
    if features["alpha_rel"] < 0.15:
        reasons.append(f"Suppressed alpha rhythm ({features['alpha_rel']*100:.1f}%) — alpha is dominant in healthy awake EEG; its absence is notable")
        flags.append("⚠️ Alpha Suppression")
    else:
        flags.append("✅ Normal Alpha")
    if features["spike_rate"] > 2.0:
        reasons.append(f"High spike rate ({features['spike_rate']:.1f} spikes/sec) — rhythmic spike bursts are a hallmark of epileptiform discharges")
        flags.append("⚠️ Spike Bursts Detected")
    else:
        flags.append("✅ Low Spike Rate")
    if features["spectral_entropy"] < 2.5:
        reasons.append(f"Low spectral entropy ({features['spectral_entropy']:.2f}) — signal is unusually rhythmic/periodic, consistent with seizure patterns")
        flags.append("⚠️ Low Entropy (rhythmic)")
    else:
        flags.append("✅ Normal Entropy")
    if features["delta_alpha_ratio"] > 3.0:
        reasons.append(f"Delta/Alpha ratio = {features['delta_alpha_ratio']:.2f} — high ratio indicates slow-wave dominance over normal background rhythm")
        flags.append("⚠️ High δ/α Ratio")
    else:
        flags.append("✅ Normal δ/α Ratio")
    if features["variance"] > 2000:
        reasons.append(f"High signal variance ({features['variance']:.0f} μV²) — large amplitude swings consistent with seizure discharges")
        flags.append("⚠️ High Variance")
    else:
        flags.append("✅ Normal Variance")
    return reasons, flags

# ══════════════════════════════════════════════════════════════════════
# 3. ML MODEL
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Training model on synthetic EEG data...")
def get_model():
    X, y = [], []
    for i in range(150):
        _, s = generate_normal_eeg(duration=5, fs=256, seed=i)
        _, fv, *_ = extract_all_features(s, 256)
        X.append(fv); y.append(0)
    for i in range(150):
        _, s = generate_epileptic_eeg(duration=5, fs=256, seed=i+10000)
        _, fv, *_ = extract_all_features(s, 256)
        X.append(fv); y.append(1)
    X = np.nan_to_num(np.array(X), nan=0.0, posinf=1e6, neginf=-1e6)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])
    model.fit(X, np.array(y))
    return model

def predict(model, fvec):
    fvec = np.nan_to_num(fvec, nan=0.0, posinf=1e6, neginf=-1e6).reshape(1, -1)
    prob = model.predict_proba(fvec)[0]
    return ("Epileptic" if prob[1] > 0.5 else "Normal"), float(prob[1])

# ══════════════════════════════════════════════════════════════════════
# 4. PLOTTING
# ══════════════════════════════════════════════════════════════════════

DARK_BG="#0d1117"; CARD_BG="#161b22"; GRID_C="#21262d"
NORMAL_C="#3fb950"; EPILEPTIC_C="#f85149"; ACCENT="#58a6ff"; MUTED="#8b949e"; TEXT_C="#e6edf3"
BAND_COLORS={"Delta":"#f85149","Theta":"#e3b341","Alpha":"#3fb950","Beta":"#58a6ff","Gamma":"#bc8cff"}

def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(DARK_BG); ax.tick_params(colors=MUTED, labelsize=9); ax.spines[:].set_color(GRID_C)
    ax.set_title(title, color=TEXT_C, fontsize=12, pad=8, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9); ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.8)

def plot_signal(t, sig, spike_idx=None, label="EEG Signal", color=None):
    if color is None:
        color = NORMAL_C if "Normal" in label else EPILEPTIC_C
    fig, ax = plt.subplots(figsize=(12, 3)); fig.patch.set_facecolor(DARK_BG)
    ax.plot(t, sig, color=color, linewidth=0.8, alpha=0.9)
    if spike_idx is not None and len(spike_idx) > 0:
        ax.scatter(t[spike_idx], sig[spike_idx], color="#ffa657", s=15, zorder=5, alpha=0.8, label="Detected Spikes")
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)
    _style_ax(ax, f"EEG Segment — {label}", "Time (s)", "Amplitude (μV)"); plt.tight_layout(); return fig

def plot_psd(freqs, psd, label="Power Spectral Density"):
    fig, ax = plt.subplots(figsize=(12, 3.5)); fig.patch.set_facecolor(DARK_BG)
    mask = freqs <= 50; ax.semilogy(freqs[mask], psd[mask], color=ACCENT, linewidth=1.5)
    for band, (lo, hi) in BANDS.items():
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        ax.fill_between(freqs[idx], psd[idx], alpha=0.25, color=BAND_COLORS[band], label=band)
    ax.legend(facecolor=CARD_BG, edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9, ncol=5, loc="upper right")
    _style_ax(ax, label, "Frequency (Hz)", "Power (μV²/Hz)"); plt.tight_layout(); return fig

def plot_band_power_bars(band_powers, label="Band Power"):
    fig, ax = plt.subplots(figsize=(8, 3.5)); fig.patch.set_facecolor(DARK_BG)
    bands, powers = list(band_powers.keys()), list(band_powers.values())
    bars = ax.barh(bands, powers, color=[BAND_COLORS[b] for b in bands], alpha=0.85, height=0.6)
    for bar, val in zip(bars, powers):
        ax.text(val*1.02, bar.get_y()+bar.get_height()/2, f"{val:.1f}", va="center", color=TEXT_C, fontsize=9)
    ax.set_facecolor(DARK_BG); ax.tick_params(colors=TEXT_C, labelsize=10); ax.spines[:].set_color(GRID_C)
    ax.set_xlabel("Power (μV²/Hz)", color=MUTED, fontsize=9)
    ax.set_title(label, color=TEXT_C, fontsize=12, pad=8, fontweight="bold")
    ax.grid(True, axis="x", color=GRID_C, linewidth=0.5); plt.tight_layout(); return fig

def plot_comparison(t1, s1, t2, s2, f1, p1, f2, p2):
    fig = plt.figure(figsize=(14, 7)); fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1, ax2 = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
    ax3, ax4 = fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])
    ax1.plot(t1, s1, color=NORMAL_C, linewidth=0.8, alpha=0.9)
    ax2.plot(t2, s2, color=EPILEPTIC_C, linewidth=0.8, alpha=0.9)
    _style_ax(ax1, "Normal EEG", "Time (s)", "Amplitude (μV)")
    _style_ax(ax2, "Epileptic EEG", "Time (s)", "Amplitude (μV)")
    mask = f1 <= 50
    ax3.semilogy(f1[mask], p1[mask], color=NORMAL_C, linewidth=1.5)
    ax4.semilogy(f2[mask], p2[mask], color=EPILEPTIC_C, linewidth=1.5)
    _style_ax(ax3, "Normal — PSD", "Frequency (Hz)", "Power")
    _style_ax(ax4, "Epileptic — PSD", "Frequency (Hz)", "Power")
    return fig

def plot_feature_radar(feat_n, feat_e):
    keys = ["delta_rel","theta_rel","alpha_rel","spike_rate","spectral_entropy","hjorth_complexity"]
    labels = ["Delta\nRelPower","Theta\nRelPower","Alpha\nRelPower","Spike\nRate","Spectral\nEntropy","Hjorth\nComplexity"]
    vals_n = np.array([feat_n.get(k,0) for k in keys], dtype=float)
    vals_e = np.array([feat_e.get(k,0) for k in keys], dtype=float)
    max_v = np.maximum(np.maximum(vals_n, vals_e), 1e-9)
    vals_n, vals_e = vals_n/max_v, vals_e/max_v
    N = len(keys); angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist(); angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
    for vals, color, lbl in [(vals_n, NORMAL_C, "Normal"), (vals_e, EPILEPTIC_C, "Epileptic")]:
        v = vals.tolist() + [vals[0]]
        ax.plot(angles, v, color=color, linewidth=2, label=lbl)
        ax.fill(angles, v, color=color, alpha=0.15)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, color=TEXT_C, fontsize=9)
    ax.set_yticklabels([]); ax.spines["polar"].set_color(GRID_C); ax.grid(color=GRID_C)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3,1.1), facecolor=CARD_BG, edgecolor=GRID_C, labelcolor=TEXT_C)
    ax.set_title("Feature Comparison Radar", color=TEXT_C, pad=20, fontsize=12, fontweight="bold")
    return fig

# ══════════════════════════════════════════════════════════════════════
# 5. SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🧠 NeuroLens")
    st.markdown("---")
    st.markdown("### Signal Source")
    source = st.radio("Choose EEG type", ["Normal EEG", "Epileptic EEG", "Random (surprise me)"], index=1)
    st.markdown("### Parameters")
    duration = st.slider("Segment duration (s)", 3, 15, 5)
    fs = st.select_slider("Sampling rate (Hz)", [128, 256, 512], value=256)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    st.markdown("---")
    run_btn = st.button("▶  Analyze Segment", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Classifies EEG segments and explains *why* using band power, spike detection, spectral entropy, and Hjorth parameters.")

# ══════════════════════════════════════════════════════════════════════
# 6. MAIN UI
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>Neuro<span>Lens</span></h1>
  <p>EEG Insight &amp; Seizure Pattern Explorer — interpretable signal analysis</p>
</div>
""", unsafe_allow_html=True)

if "t" not in st.session_state or run_btn:
    import random
    if source == "Normal EEG":
        true_label = "Normal"; t, sig = generate_normal_eeg(duration=duration, fs=fs, seed=int(seed))
    elif source == "Epileptic EEG":
        true_label = "Epileptic"; t, sig = generate_epileptic_eeg(duration=duration, fs=fs, seed=int(seed))
    else:
        random.seed(int(seed))
        if random.random() > 0.5:
            true_label = "Normal"; t, sig = generate_normal_eeg(duration=duration, fs=fs, seed=int(seed))
        else:
            true_label = "Epileptic"; t, sig = generate_epileptic_eeg(duration=duration, fs=fs, seed=int(seed))
    st.session_state.update({"t": t, "sig": sig, "label": true_label, "fs": fs})

t = st.session_state.t; sig = st.session_state.sig
true_label = st.session_state.label; fs = st.session_state.fs

features, fvec, freqs, psd, spike_idx = extract_all_features(sig, fs)
model = get_model()
pred_label, seizure_prob = predict(model, fvec)
reasons, flags = interpret_features(features, seizure_prob)

col1, col2, col3, col4 = st.columns(4)
with col1:
    badge_class = "badge-epileptic" if pred_label == "Epileptic" else "badge-normal"
    st.markdown("**Prediction**")
    st.markdown(f'<span class="{badge_class}">{pred_label}</span>', unsafe_allow_html=True)
with col2:
    st.metric("Seizure Probability", f"{seizure_prob*100:.1f}%")
with col3:
    st.metric("Spike Rate", f"{features['spike_rate']:.1f} /s")
with col4:
    st.metric("Spectral Entropy", f"{features['spectral_entropy']:.2f}")

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Signal Viewer", "🌊 Frequency Analysis",
    "🔬 Feature Breakdown", "💡 Insight Panel", "⚖️ Compare Mode",
])

with tab1:
    st.markdown('<div class="card"><h4>Raw EEG Signal</h4>', unsafe_allow_html=True)
    color = NORMAL_C if true_label == "Normal" else EPILEPTIC_C
    st.pyplot(plot_signal(t, sig, spike_idx=spike_idx, label=true_label, color=color), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("True Label", true_label); c2.metric("Duration", f"{duration}s"); c3.metric("Samples", f"{len(sig):,}")
    if len(spike_idx) > 0:
        st.info(f"🔶 **{len(spike_idx)} spike events** detected. Spike rate: {features['spike_rate']:.1f}/sec")
    else:
        st.success("✅ No significant spike events detected.")

with tab2:
    st.markdown('<div class="card"><h4>Power Spectral Density</h4>', unsafe_allow_html=True)
    st.pyplot(plot_psd(freqs, psd, label=f"PSD — {true_label} EEG"), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="card"><h4>Band Power Distribution</h4>', unsafe_allow_html=True)
    band_powers = {k: features[f"{k.lower()}_power"] for k in ["Delta","Theta","Alpha","Beta","Gamma"]}
    st.pyplot(plot_band_power_bars(band_powers), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Normal EEG:** Alpha dominant, low delta, balanced rhythms")
    with c2:
        st.markdown("**Epileptic EEG:** High delta, suppressed alpha, rhythmic spike-waves")

with tab3:
    st.markdown('<div class="card"><h4>Extracted Features</h4>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Band Powers (relative %)**")
        for band in ["Delta","Theta","Alpha","Beta","Gamma"]:
            rel = features[f"{band.lower()}_rel"] * 100
            bar = "█" * int(rel/5) + "░" * (20 - int(rel/5))
            st.markdown(f"`{band:5s}` {bar} **{rel:.1f}%**")
    with c2:
        st.markdown("**Spike & Time-Domain**")
        st.metric("Spike Rate", f"{features['spike_rate']:.2f} /s")
        st.metric("Signal Variance", f"{features['variance']:.1f} μV²")
        st.metric("δ/α Ratio", f"{features['delta_alpha_ratio']:.2f}")
    with c3:
        st.markdown("**Complexity Measures**")
        st.metric("Spectral Entropy", f"{features['spectral_entropy']:.3f}")
        st.metric("Hjorth Mobility", f"{features['hjorth_mobility']:.3f}")
        st.metric("Hjorth Complexity", f"{features['hjorth_complexity']:.3f}")
    st.markdown("</div>", unsafe_allow_html=True)
    flag_html = "".join(
        f'<span class="flag-warn">{f}</span>' if "⚠️" in f else f'<span class="flag-ok">{f}</span>'
        for f in flags
    )
    st.markdown(flag_html, unsafe_allow_html=True)

with tab4:
    prob_pct = seizure_prob * 100
    bar_color = "#f85149" if seizure_prob > 0.5 else "#3fb950"
    st.markdown(f"""
    <div class="card">
      <h4>Model Prediction</h4>
      <p style="color:#8b949e;margin-bottom:0.8rem;">SVM classifier analyzed {len(fvec)} features from this EEG segment.</p>
      <div style="background:#21262d;border-radius:6px;height:18px;overflow:hidden;margin-bottom:0.5rem;">
        <div style="background:{bar_color};width:{prob_pct:.1f}%;height:100%;border-radius:6px;"></div>
      </div>
      <p style="font-family:'IBM Plex Mono',monospace;font-size:0.9rem;color:{bar_color};">
        Seizure probability: {prob_pct:.1f}%
      </p>
    </div>
    """, unsafe_allow_html=True)
    if reasons:
        st.markdown("##### Why this prediction was made")
        for r in reasons:
            st.markdown(f'<div class="insight-item">🔍 {r}</div>', unsafe_allow_html=True)
    else:
        st.success("No strong seizure indicators found. Signal appears within normal parameters.")
    with st.expander("📘 Clinical Context — EEG Interpretation Glossary"):
        st.markdown("""
        | Feature | What it measures | Why it matters |
        |---|---|---|
        | **Delta power** | 0.5–4 Hz energy | Elevated in deep sleep and seizures |
        | **Alpha rhythm** | 8–13 Hz energy | Dominant in normal awake EEG; suppressed in seizures |
        | **Spike rate** | Sharp deflections/sec | Epileptiform discharges appear as rhythmic spikes |
        | **Spectral entropy** | Signal complexity | Low = repetitive/seizure-like; high = normal |
        | **Hjorth mobility** | Mean frequency proxy | Increases with faster/more complex activity |
        | **δ/α ratio** | Slow vs fast power | High ratio = slow-wave dominance, seizure indicator |
        """)

with tab5:
    st.markdown("Compare a **Normal** and **Epileptic** segment side-by-side.")
    comp_seed = st.slider("Comparison seed", 0, 999, 7, key="comp_seed")
    t_n, sig_n = generate_normal_eeg(duration=5, fs=256, seed=comp_seed)
    t_e, sig_e = generate_epileptic_eeg(duration=5, fs=256, seed=comp_seed+1)
    feat_n, _, freqs_n, psd_n, _ = extract_all_features(sig_n, 256)
    feat_e, _, freqs_e, psd_e, _ = extract_all_features(sig_e, 256)
    st.pyplot(plot_comparison(t_n, sig_n, t_e, sig_e, freqs_n, psd_n, freqs_e, psd_e), use_container_width=True)
    st.pyplot(plot_feature_radar(feat_n, feat_e), use_container_width=False)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Normal:** Small amplitude, alpha/beta dominant PSD, low spike rate")
    with c2:
        st.markdown("**Epileptic:** Large bursts, delta-dominant PSD, rhythmic spike-wave complexes")

st.markdown("---")
st.markdown('<p style="color:#484f58;font-size:0.8rem;text-align:center;font-family:IBM Plex Mono,monospace;">NeuroLens · Interpretable EEG Analysis · Built with Python + Streamlit</p>', unsafe_allow_html=True)
