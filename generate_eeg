"""
Synthetic EEG Data Generator for NeuroLens
Generates realistic normal and epileptic EEG segments.
Replace this later with real PhysioNet data using the loader in utils/data_loader.py
"""

import numpy as np


def generate_normal_eeg(duration=10, fs=256, seed=None):
    """
    Generate a realistic normal EEG segment.
    Normal EEG: dominant alpha (8-13 Hz) and beta (13-30 Hz) rhythms,
    low variance, no spike bursts.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, duration, int(duration * fs))

    # Background noise (pink-ish)
    signal = np.random.normal(0, 10, len(t))

    # Delta (0.5-4 Hz) - low power in awake state
    signal += 5 * np.sin(2 * np.pi * 2 * t + np.random.uniform(0, 2 * np.pi))

    # Theta (4-8 Hz) - moderate
    signal += 8 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2 * np.pi))

    # Alpha (8-13 Hz) - dominant in normal EEG
    signal += 20 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2 * np.pi))
    signal += 15 * np.sin(2 * np.pi * 11 * t + np.random.uniform(0, 2 * np.pi))

    # Beta (13-30 Hz) - present but not dominant
    signal += 10 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2 * np.pi))

    return t, signal.astype(np.float32)


def generate_epileptic_eeg(duration=10, fs=256, seed=None):
    """
    Generate a realistic epileptic EEG segment.
    Ictal EEG: high delta/theta power, sharp spike-wave complexes,
    high variance, rhythmic bursting.
    """
    if seed is not None:
        np.random.seed(seed)

    t = np.linspace(0, duration, int(duration * fs))

    # Elevated background
    signal = np.random.normal(0, 20, len(t))

    # High delta (2-4 Hz) - characteristic of seizures
    signal += 40 * np.sin(2 * np.pi * 3 * t + np.random.uniform(0, 2 * np.pi))
    signal += 30 * np.sin(2 * np.pi * 4 * t + np.random.uniform(0, 2 * np.pi))

    # Elevated theta
    signal += 25 * np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2 * np.pi))

    # Spike-wave complexes (characteristic epileptic pattern)
    spike_rate = 3.0  # ~3 Hz spike-wave
    spikes = np.zeros(len(t))
    spike_times = np.arange(0, duration, 1.0 / spike_rate)

    for st in spike_times:
        idx = int(st * fs)
        if idx < len(t) - 20:
            # Sharp spike
            spikes[idx : idx + 5] += np.array([0, 60, 120, 60, 0]) * np.random.uniform(0.7, 1.3)
            # Followed by slow wave
            wave_len = int(0.3 * fs)
            if idx + 5 + wave_len < len(t):
                wave = -50 * np.sin(np.pi * np.arange(wave_len) / wave_len)
                spikes[idx + 5 : idx + 5 + wave_len] += wave

    signal += spikes

    # Add random high-amplitude bursts
    burst_positions = np.random.choice(len(t) - int(0.5 * fs), size=5, replace=False)
    for bp in burst_positions:
        burst_len = int(0.5 * fs)
        burst = np.random.normal(0, 50, burst_len) * np.hanning(burst_len)
        signal[bp : bp + burst_len] += burst

    return t, signal.astype(np.float32)


def get_sample_segments(n_segments=10, fs=256, duration=5):
    """
    Returns a list of (label, time, signal) tuples for demo use.
    """
    segments = []
    for i in range(n_segments // 2):
        t, sig = generate_normal_eeg(duration=duration, fs=fs, seed=i * 10)
        segments.append(("Normal", t, sig))
    for i in range(n_segments // 2):
        t, sig = generate_epileptic_eeg(duration=duration, fs=fs, seed=i * 10 + 1)
        segments.append(("Epileptic", t, sig))
    return segments
