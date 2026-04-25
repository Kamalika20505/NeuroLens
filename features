"""
Feature Extraction for NeuroLens
Computes interpretable features from EEG signals.
These features are both used by the ML model AND shown to the user.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import entropy as scipy_entropy


# ── Band definitions (Hz) ──────────────────────────────────────────────
BANDS = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta":  (13, 30),
    "Gamma": (30, 50),
}


def compute_band_powers(sig, fs):
    """
    Compute relative power in each EEG frequency band using Welch's method.
    Returns a dict: band_name -> absolute_power
    """
    freqs, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    band_powers = {}
    for band, (lo, hi) in BANDS.items():
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        band_powers[band] = float(np.trapezoid(psd[idx], freqs[idx]))
    return band_powers, freqs, psd


def compute_spike_rate(sig, fs, threshold_std=3.0):
    """
    Detect spikes as samples exceeding threshold_std standard deviations.
    Returns spikes_per_second and the spike indices.
    """
    std = np.std(sig)
    mean = np.mean(sig)
    spike_idx = np.where(np.abs(sig - mean) > threshold_std * std)[0]

    # Cluster nearby spike indices (within 50 ms)
    gap = int(0.05 * fs)
    if len(spike_idx) == 0:
        return 0.0, np.array([])

    clusters = []
    cluster = [spike_idx[0]]
    for idx in spike_idx[1:]:
        if idx - cluster[-1] <= gap:
            cluster.append(idx)
        else:
            clusters.append(cluster[0])
            cluster = [idx]
    clusters.append(cluster[0])

    duration_sec = len(sig) / fs
    rate = len(clusters) / duration_sec
    return rate, np.array(clusters)


def compute_spectral_entropy(sig, fs):
    """
    Spectral entropy: measures complexity/randomness of the signal.
    Low = rhythmic/seizure-like. High = complex/normal.
    """
    _, psd = scipy_signal.welch(sig, fs=fs, nperseg=min(256, len(sig)))
    psd_norm = psd / (psd.sum() + 1e-12)
    return float(scipy_entropy(psd_norm + 1e-12))


def compute_hjorth_params(sig):
    """
    Hjorth parameters: activity, mobility, complexity.
    Classic EEG features.
    """
    activity = float(np.var(sig))
    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    mobility = float(np.sqrt(np.var(diff1) / (np.var(sig) + 1e-12)))
    complexity = float(
        np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-12)) / (mobility + 1e-12)
    )
    return activity, mobility, complexity


def extract_all_features(sig, fs):
    """
    Master function: extract ALL features from one EEG segment.
    Returns a flat feature dict (for display) and a feature vector (for ML).
    """
    band_powers, freqs, psd = compute_band_powers(sig, fs)
    total_power = sum(band_powers.values()) + 1e-12
    relative_powers = {k: v / total_power for k, v in band_powers.items()}

    spike_rate, spike_idx = compute_spike_rate(sig, fs)
    spec_entropy = compute_spectral_entropy(sig, fs)
    activity, mobility, complexity = compute_hjorth_params(sig)
    variance = float(np.var(sig))

    # Delta/Alpha ratio — high in seizures
    delta_alpha_ratio = band_powers["Delta"] / (band_powers["Alpha"] + 1e-12)

    features = {
        # Band powers (absolute)
        "delta_power":       band_powers["Delta"],
        "theta_power":       band_powers["Theta"],
        "alpha_power":       band_powers["Alpha"],
        "beta_power":        band_powers["Beta"],
        "gamma_power":       band_powers["Gamma"],
        # Relative powers
        "delta_rel":         relative_powers["Delta"],
        "theta_rel":         relative_powers["Theta"],
        "alpha_rel":         relative_powers["Alpha"],
        "beta_rel":          relative_powers["Beta"],
        # Ratios
        "delta_alpha_ratio": delta_alpha_ratio,
        # Spike detection
        "spike_rate":        spike_rate,
        # Complexity measures
        "spectral_entropy":  spec_entropy,
        "hjorth_activity":   activity,
        "hjorth_mobility":   mobility,
        "hjorth_complexity": complexity,
        "variance":          variance,
    }

    # Feature vector for ML (ordered)
    feature_vector = [
        features["delta_power"],
        features["theta_power"],
        features["alpha_power"],
        features["beta_power"],
        features["delta_rel"],
        features["theta_rel"],
        features["alpha_rel"],
        features["delta_alpha_ratio"],
        features["spike_rate"],
        features["spectral_entropy"],
        features["hjorth_activity"],
        features["hjorth_mobility"],
        features["hjorth_complexity"],
        features["variance"],
    ]

    return features, np.array(feature_vector, dtype=np.float32), freqs, psd, spike_idx


def interpret_features(features, prediction_prob):
    """
    Convert raw feature values into human-readable clinical interpretation.
    This is the 'insight panel' — explaining WHY the model predicted what it did.
    """
    reasons = []
    flags = []

    # Delta power interpretation
    if features["delta_rel"] > 0.35:
        reasons.append(f"High delta-band dominance ({features['delta_rel']*100:.1f}%) — "
                        "slow waves often seen in ictal/seizure activity")
        flags.append("⚠️ Elevated Delta")
    else:
        flags.append("✅ Normal Delta")

    # Alpha interpretation
    if features["alpha_rel"] < 0.15:
        reasons.append(f"Suppressed alpha rhythm ({features['alpha_rel']*100:.1f}%) — "
                        "alpha is the dominant rhythm in healthy awake EEG; its absence is notable")
        flags.append("⚠️ Alpha Suppression")
    else:
        flags.append("✅ Normal Alpha")

    # Spike rate interpretation
    if features["spike_rate"] > 2.0:
        reasons.append(f"High spike rate ({features['spike_rate']:.1f} spikes/sec) — "
                        "rhythmic spike bursts are a hallmark of epileptiform discharges")
        flags.append("⚠️ Spike Bursts Detected")
    else:
        flags.append("✅ Low Spike Rate")

    # Spectral entropy
    if features["spectral_entropy"] < 2.5:
        reasons.append(f"Low spectral entropy ({features['spectral_entropy']:.2f}) — "
                        "signal is unusually rhythmic/periodic, consistent with seizure patterns")
        flags.append("⚠️ Low Entropy (rhythmic)")
    else:
        flags.append("✅ Normal Entropy")

    # Delta/Alpha ratio
    if features["delta_alpha_ratio"] > 3.0:
        reasons.append(f"Delta/Alpha ratio = {features['delta_alpha_ratio']:.2f} — "
                        "high ratio indicates slow-wave dominance over normal background rhythm")
        flags.append("⚠️ High δ/α Ratio")
    else:
        flags.append("✅ Normal δ/α Ratio")

    # Variance
    if features["variance"] > 2000:
        reasons.append(f"High signal variance ({features['variance']:.0f} μV²) — "
                        "large amplitude swings consistent with seizure discharges")
        flags.append("⚠️ High Variance")
    else:
        flags.append("✅ Normal Variance")

    return reasons, flags
