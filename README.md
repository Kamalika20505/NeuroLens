# NeuroLens
# 🧠 NeuroLens — EEG Insight & Seizure Pattern Explorer

An interpretable EEG analysis tool that classifies epileptic patterns and **explains why** using signal features.

## What it does

| Tab | What you see |
|---|---|
| **Signal Viewer** | Raw EEG waveform with spike highlights |
| **Frequency Analysis** | Power Spectral Density + band power bars |
| **Feature Breakdown** | Delta/Alpha/Theta power, spike rate, entropy, Hjorth params |
| **Insight Panel** | Human-readable explanation of the prediction |
| **Compare Mode** | Normal vs Epileptic side-by-side |

## Run it

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

## Project structure

```
neurolens/
├── app.py                  # Main Streamlit UI
├── requirements.txt
├── data/
│   └── generate_eeg.py     # Synthetic EEG generator (replace with PhysioNet later)
└── utils/
    ├── features.py         # Feature extraction (band power, spikes, entropy, Hjorth)
    ├── model.py            # SVM classifier training + inference
    └── plots.py            # All matplotlib visualizations
```

## Features extracted

- **Band powers**: Delta, Theta, Alpha, Beta, Gamma (absolute + relative)
- **Delta/Alpha ratio**: Key seizure indicator
- **Spike rate**: Epileptiform discharge detection
- **Spectral entropy**: Signal complexity measure
- **Hjorth parameters**: Activity, Mobility, Complexity

## Using real data (PhysioNet)

Replace `data/generate_eeg.py` with a loader for the
[CHB-MIT EEG dataset](https://physionet.org/content/chbmit/1.0.0/)
or the [Bonn EEG dataset](https://www.upf.edu/web/ntsa/downloads).

The feature extraction and model code work with any 1D EEG array — just pass in `(signal, fs)`.

