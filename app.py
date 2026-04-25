"""
NeuroLens — EEG Insight & Seizure Pattern Explorer
Run with: streamlit run app.py
"""

import sys
import os

# Must be first — fixes module resolution on Streamlit Cloud
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import numpy as np
import streamlit as st

from data.generate_eeg import generate_normal_eeg, generate_epileptic_eeg
from utils.features import extract_all_features, interpret_features
from utils.plots import (
    plot_signal, plot_psd, plot_band_power_bars,
    plot_comparison, plot_feature_radar,
)

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroLens",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d1117;
    color: #e6edf3;
  }
  .main { background-color: #0d1117; }
  .block-container { padding: 1.5rem 2rem; max-width: 1200px; }

  /* Hero */
  .hero {
    padding: 2rem 0 1rem;
    border-bottom: 1px solid #21262d;
    margin-bottom: 1.5rem;
  }
  .hero h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 600;
    color: #e6edf3;
    margin: 0;
    letter-spacing: -1px;
  }
  .hero span { color: #58a6ff; }
  .hero p { color: #8b949e; font-size: 1rem; margin: 0.4rem 0 0; }

  /* Cards */
  .card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
  }
  .card h4 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #58a6ff;
    margin: 0 0 0.6rem;
  }

  /* Prediction badge */
  .badge-normal {
    display: inline-block;
    background: rgba(63,185,80,0.15);
    border: 1px solid #3fb950;
    color: #3fb950;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
    border-radius: 6px;
  }
  .badge-epileptic {
    display: inline-block;
    background: rgba(248,81,73,0.15);
    border: 1px solid #f85149;
    color: #f85149;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.4rem 1.2rem;
    border-radius: 6px;
  }

  /* Insight bullets */
  .insight-item {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
    color: #c9d1d9;
    line-height: 1.5;
  }

  /* Feature tags */
  .flag-warn {
    display: inline-block;
    background: rgba(248,81,73,0.1);
    border: 1px solid rgba(248,81,73,0.4);
    color: #f85149;
    font-size: 0.78rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin: 3px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .flag-ok {
    display: inline-block;
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.4);
    color: #3fb950;
    font-size: 0.78rem;
    padding: 2px 10px;
    border-radius: 20px;
    margin: 3px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #21262d;
  }
  section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

  /* Metric overrides */
  [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.4rem !important;
    color: #58a6ff !important;
  }
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.8rem !important; }

  div[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.82rem;
    color: #8b949e;
  }

  hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NeuroLens")
    st.markdown("---")

    st.markdown("### Signal Source")
    source = st.radio(
        "Choose EEG type",
        ["Normal EEG", "Epileptic EEG", "Random (surprise me)"],
        index=1,
    )

    st.markdown("### Parameters")
    duration = st.slider("Segment duration (s)", 3, 15, 5)
    fs = st.select_slider("Sampling rate (Hz)", [128, 256, 512], value=256)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)

    st.markdown("---")
    run_btn = st.button("▶  Analyze Segment", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    NeuroLens classifies EEG segments as normal or epileptic and explains *why* using
    signal features:
    - Band power analysis
    - Spike detection
    - Spectral entropy
    - Hjorth parameters
    """)
    st.markdown("Built with `scipy`, `sklearn`, `streamlit`")


# ── Hero ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>Neuro<span>Lens</span></h1>
  <p>EEG Insight &amp; Seizure Pattern Explorer — interpretable signal analysis</p>
</div>
""", unsafe_allow_html=True)


# ── Generate data ──────────────────────────────────────────────────────
if "t" not in st.session_state or run_btn:
    if source == "Normal EEG":
        label = "Normal"
        t, sig = generate_normal_eeg(duration=duration, fs=fs, seed=int(seed))
    elif source == "Epileptic EEG":
        label = "Epileptic"
        t, sig = generate_epileptic_eeg(duration=duration, fs=fs, seed=int(seed))
    else:
        import random
        random.seed(int(seed))
        if random.random() > 0.5:
            label = "Normal"
            t, sig = generate_normal_eeg(duration=duration, fs=fs, seed=int(seed))
        else:
            label = "Epileptic"
            t, sig = generate_epileptic_eeg(duration=duration, fs=fs, seed=int(seed))

    st.session_state.t = t
    st.session_state.sig = sig
    st.session_state.label = label
    st.session_state.fs = fs

t = st.session_state.t
sig = st.session_state.sig
true_label = st.session_state.label
fs = st.session_state.fs

# Extract features
features, fvec, freqs, psd, spike_idx = extract_all_features(sig, fs)


# ── Run model ──────────────────────────────────────────────────────────
model = get_model()
from utils.model import predict
pred_label, seizure_prob = predict(model, fvec)
reasons, flags = interpret_features(features, seizure_prob)


# ── Top metrics row ────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    badge_class = "badge-epileptic" if pred_label == "Epileptic" else "badge-normal"
    st.markdown(f"**Prediction**")
    st.markdown(f'<span class="{badge_class}">{pred_label}</span>', unsafe_allow_html=True)
with col2:
    st.metric("Seizure Probability", f"{seizure_prob*100:.1f}%")
with col3:
    st.metric("Spike Rate", f"{features['spike_rate']:.1f} /s")
with col4:
    st.metric("Spectral Entropy", f"{features['spectral_entropy']:.2f}")

st.markdown("<br>", unsafe_allow_html=True)


# ── Main tabs ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Signal Viewer",
    "🌊 Frequency Analysis",
    "🔬 Feature Breakdown",
    "💡 Insight Panel",
    "⚖️ Compare Mode",
])


# ── TAB 1: Signal Viewer ───────────────────────────────────────────────
with tab1:
    st.markdown('<div class="card"><h4>Raw EEG Signal</h4>', unsafe_allow_html=True)
    color = "#3fb950" if true_label == "Normal" else "#f85149"
    fig = plot_signal(t, sig, spike_idx=spike_idx, label=true_label, color=color)
    st.pyplot(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("True Label (source)", true_label)
    c2.metric("Duration", f"{duration}s")
    c3.metric("Samples", f"{len(sig):,}")

    if len(spike_idx) > 0:
        st.info(f"🔶 **{len(spike_idx)} spike events** detected (orange dots). "
                f"Spike rate: {features['spike_rate']:.1f}/sec")
    else:
        st.success("✅ No significant spike events detected in this segment.")


# ── TAB 2: Frequency Analysis ──────────────────────────────────────────
with tab2:
    st.markdown('<div class="card"><h4>Power Spectral Density</h4>', unsafe_allow_html=True)
    fig_psd = plot_psd(freqs, psd, label=f"PSD — {true_label} EEG")
    st.pyplot(fig_psd, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card"><h4>Band Power Distribution</h4>', unsafe_allow_html=True)
    band_powers = {k: features[f"{k.lower()}_power"] for k in ["Delta","Theta","Alpha","Beta","Gamma"]}
    fig_bars = plot_band_power_bars(band_powers, label="Absolute Band Power")
    st.pyplot(fig_bars, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("##### What this means")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Normal EEG typically shows:**
        - Alpha (8–13 Hz) dominant when eyes closed
        - Low delta power in awake state
        - Balanced theta/beta rhythms
        """)
    with col2:
        st.markdown("""
        **Epileptic EEG typically shows:**
        - High delta power (slow waves)
        - Suppressed alpha rhythm
        - Rhythmic sharp-wave complexes in 1–4 Hz
        """)


# ── TAB 3: Feature Breakdown ───────────────────────────────────────────
with tab3:
    st.markdown('<div class="card"><h4>Extracted Features</h4>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Band Powers (relative %)**")
        for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
            rel = features[f"{band.lower()}_rel"] * 100
            bar = "█" * int(rel / 5) + "░" * (20 - int(rel / 5))
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

    st.markdown("##### Feature flags")
    flag_html = ""
    for flag in flags:
        if "⚠️" in flag:
            flag_html += f'<span class="flag-warn">{flag}</span>'
        else:
            flag_html += f'<span class="flag-ok">{flag}</span>'
    st.markdown(flag_html, unsafe_allow_html=True)


# ── TAB 4: Insight Panel ───────────────────────────────────────────────
with tab4:
    prob_pct = seizure_prob * 100
    bar_color = "#f85149" if seizure_prob > 0.5 else "#3fb950"

    st.markdown(f"""
    <div class="card">
      <h4>Model Prediction</h4>
      <p style="color:#8b949e; margin-bottom:0.8rem;">
        The SVM classifier analyzed {len(fvec)} features extracted from this EEG segment.
      </p>
      <div style="background:#21262d; border-radius:6px; height:18px; overflow:hidden; margin-bottom:0.5rem;">
        <div style="background:{bar_color}; width:{prob_pct:.1f}%; height:100%; border-radius:6px;
                    transition:width 0.5s ease;"></div>
      </div>
      <p style="font-family:'IBM Plex Mono',monospace; font-size:0.9rem; color:{bar_color};">
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

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📘 Clinical Context — EEG Interpretation Glossary"):
        st.markdown("""
        | Feature | What it measures | Why it matters |
        |---|---|---|
        | **Delta power** | 0.5–4 Hz energy | Elevated in deep sleep and seizures |
        | **Alpha rhythm** | 8–13 Hz energy | Dominant in normal awake EEG; suppressed in seizures |
        | **Spike rate** | Sharp deflections/sec | Epileptiform discharges appear as rhythmic spikes |
        | **Spectral entropy** | Signal complexity | Low = repetitive/seizure-like; high = normal complexity |
        | **Hjorth mobility** | Mean frequency proxy | Increases with faster/more complex activity |
        | **δ/α ratio** | Slow vs fast power | High ratio = slow-wave dominance, seizure indicator |
        """)


# ── TAB 5: Compare Mode ────────────────────────────────────────────────
with tab5:
    st.markdown("Compare a **Normal** and **Epileptic** segment side-by-side "
                "to see the visual and spectral differences.")

    comp_seed = st.slider("Comparison seed", 0, 999, 7, key="comp_seed")

    t_n, sig_n = generate_normal_eeg(duration=5, fs=256, seed=comp_seed)
    t_e, sig_e = generate_epileptic_eeg(duration=5, fs=256, seed=comp_seed + 1)

    _, fvec_n, freqs_n, psd_n, spike_n = extract_all_features(sig_n, 256)
    feat_e, fvec_e, freqs_e, psd_e, spike_e = extract_all_features(sig_e, 256)
    feat_n, *_ = extract_all_features(sig_n, 256)

    fig_comp = plot_comparison(t_n, sig_n, t_e, sig_e, freqs_n, psd_n, freqs_e, psd_e)
    st.pyplot(fig_comp, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card"><h4>Normal — Feature Radar</h4></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="card"><h4>Epileptic — Feature Radar</h4></div>',
                    unsafe_allow_html=True)

    fig_radar = plot_feature_radar(feat_n, feat_e)
    st.pyplot(fig_radar, use_container_width=False)

    st.markdown("##### Key visual differences")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("""
        **Normal EEG (left)**
        - Smaller amplitude fluctuations
        - PSD peaks in alpha/beta range
        - Low spike rate
        - Higher spectral entropy
        """)
    with d2:
        st.markdown("""
        **Epileptic EEG (right)**
        - Large amplitude bursts
        - PSD dominated by delta
        - Rhythmic spike-wave complexes
        - Lower spectral entropy (more rhythmic)
        """)


# ── Footer ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="color:#484f58; font-size:0.8rem; text-align:center; font-family:IBM Plex Mono, monospace;">'
    "NeuroLens · Interpretable EEG Analysis · Built with Python + Streamlit"
    "</p>",
    unsafe_allow_html=True,
)
