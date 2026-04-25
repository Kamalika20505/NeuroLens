"""
Plotting utilities for NeuroLens
All matplotlib figures used in the Streamlit app.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Color palette ──────────────────────────────────────────────────────
DARK_BG    = "#0d1117"
CARD_BG    = "#161b22"
GRID_COLOR = "#21262d"
NORMAL_C   = "#3fb950"    # green
EPILEPTIC_C= "#f85149"    # red
ACCENT     = "#58a6ff"    # blue
MUTED      = "#8b949e"
TEXT_C     = "#e6edf3"

BAND_COLORS = {
    "Delta": "#f85149",
    "Theta": "#e3b341",
    "Alpha": "#3fb950",
    "Beta":  "#58a6ff",
    "Gamma": "#bc8cff",
}


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.spines[:].set_color(GRID_COLOR)
    ax.set_title(title, color=TEXT_C, fontsize=12, pad=8, fontweight="bold")
    ax.set_xlabel(xlabel, color=MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.8)


def plot_signal(t, sig, spike_idx=None, label="EEG Signal", color=None):
    """Plot a single EEG segment with optional spike highlights."""
    if color is None:
        color = NORMAL_C if "Normal" in label else EPILEPTIC_C

    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor(DARK_BG)

    ax.plot(t, sig, color=color, linewidth=0.8, alpha=0.9)

    if spike_idx is not None and len(spike_idx) > 0:
        ax.scatter(
            t[spike_idx], sig[spike_idx],
            color="#ffa657", s=15, zorder=5, alpha=0.8, label="Detected Spikes"
        )
        ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_C, fontsize=9)

    _style_ax(ax, title=f"EEG Segment — {label}",
              xlabel="Time (s)", ylabel="Amplitude (μV)")
    plt.tight_layout()
    return fig


def plot_psd(freqs, psd, label="Power Spectral Density"):
    """Plot Power Spectral Density with band shading."""
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor(DARK_BG)

    mask = freqs <= 50
    ax.semilogy(freqs[mask], psd[mask], color=ACCENT, linewidth=1.5)

    band_ranges = {
        "Delta": (0.5, 4),
        "Theta": (4, 8),
        "Alpha": (8, 13),
        "Beta":  (13, 30),
        "Gamma": (30, 50),
    }

    for band, (lo, hi) in band_ranges.items():
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        ax.fill_between(
            freqs[idx], psd[idx],
            alpha=0.25, color=BAND_COLORS[band], label=band
        )

    ax.legend(facecolor=CARD_BG, edgecolor=GRID_COLOR,
              labelcolor=TEXT_C, fontsize=9, ncol=5,
              loc="upper right")
    _style_ax(ax, title=label, xlabel="Frequency (Hz)", ylabel="Power (μV²/Hz)")
    plt.tight_layout()
    return fig


def plot_band_power_bars(band_powers, label="Band Power"):
    """Horizontal bar chart of absolute band powers."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor(DARK_BG)

    bands = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = [BAND_COLORS[b] for b in bands]

    bars = ax.barh(bands, powers, color=colors, alpha=0.85, height=0.6)

    # Value labels
    for bar, val in zip(bars, powers):
        ax.text(
            val * 1.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", color=TEXT_C, fontsize=9
        )

    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_C, labelsize=10)
    ax.spines[:].set_color(GRID_COLOR)
    ax.set_xlabel("Power (μV²/Hz)", color=MUTED, fontsize=9)
    ax.set_title(label, color=TEXT_C, fontsize=12, pad=8, fontweight="bold")
    ax.grid(True, axis="x", color=GRID_COLOR, linewidth=0.5)
    plt.tight_layout()
    return fig


def plot_comparison(t1, sig1, t2, sig2, freqs1, psd1, freqs2, psd2):
    """Side-by-side comparison: Normal vs Epileptic."""
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Top row: raw signals
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1.plot(t1, sig1, color=NORMAL_C, linewidth=0.8, alpha=0.9)
    ax2.plot(t2, sig2, color=EPILEPTIC_C, linewidth=0.8, alpha=0.9)
    _style_ax(ax1, "Normal EEG", "Time (s)", "Amplitude (μV)")
    _style_ax(ax2, "Epileptic EEG", "Time (s)", "Amplitude (μV)")

    # Bottom row: PSD
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    mask = freqs1 <= 50
    ax3.semilogy(freqs1[mask], psd1[mask], color=NORMAL_C, linewidth=1.5)
    ax4.semilogy(freqs2[mask], psd2[mask], color=EPILEPTIC_C, linewidth=1.5)

    for band, (lo, hi) in {"Delta":(0.5,4),"Alpha":(8,13)}.items():
        idx1 = np.logical_and(freqs1 >= lo, freqs1 <= hi)
        idx2 = np.logical_and(freqs2 >= lo, freqs2 <= hi)
        ax3.fill_between(freqs1[idx1], psd1[idx1], alpha=0.2, color=BAND_COLORS[band])
        ax4.fill_between(freqs2[idx2], psd2[idx2], alpha=0.2, color=BAND_COLORS[band])

    _style_ax(ax3, "Normal — PSD", "Frequency (Hz)", "Power")
    _style_ax(ax4, "Epileptic — PSD", "Frequency (Hz)", "Power")

    return fig


def plot_feature_radar(features_normal, features_epileptic):
    """Radar chart comparing key features between normal and epileptic."""
    keys = ["delta_rel", "theta_rel", "alpha_rel", "spike_rate",
            "spectral_entropy", "hjorth_complexity"]
    labels = ["Delta\nRelPower", "Theta\nRelPower", "Alpha\nRelPower",
              "Spike\nRate", "Spectral\nEntropy", "Hjorth\nComplexity"]

    def normalize(vals):
        # Normalize 0–1 for radar
        arr = np.array(vals, dtype=float)
        mx = np.maximum(np.abs(arr), 1e-6)
        return np.clip(arr / (mx * 3), 0, 1)

    vals_n = [features_normal.get(k, 0) for k in keys]
    vals_e = [features_epileptic.get(k, 0) for k in keys]

    # Simple normalisation across both
    all_vals = np.array([vals_n, vals_e])
    max_vals = all_vals.max(axis=0) + 1e-9
    vals_n_norm = all_vals[0] / max_vals
    vals_e_norm = all_vals[1] / max_vals

    N = len(keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    def _plot_radar(vals, color, label):
        vals = vals.tolist() + [vals[0]]
        ax.plot(angles, vals, color=color, linewidth=2, label=label)
        ax.fill(angles, vals, color=color, alpha=0.15)

    _plot_radar(vals_n_norm, NORMAL_C, "Normal")
    _plot_radar(vals_e_norm, EPILEPTIC_C, "Epileptic")

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, color=TEXT_C, fontsize=9)
    ax.set_yticklabels([])
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor=CARD_BG, edgecolor=GRID_COLOR, labelcolor=TEXT_C)
    ax.set_title("Feature Comparison Radar", color=TEXT_C, pad=20,
                 fontsize=12, fontweight="bold")
    return fig
