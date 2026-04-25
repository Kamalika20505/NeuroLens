"""
ML Model for NeuroLens
Trains a simple, interpretable SVM classifier on extracted EEG features.
Intentionally lightweight — the insight comes from features, not black-box DL.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib

from data.generate_eeg import generate_normal_eeg, generate_epileptic_eeg
from utils.features import extract_all_features


MODEL_PATH = os.path.join(os.path.dirname(__file__), "eeg_model.pkl")


def build_training_data(n_samples=200, fs=256, duration=5):
    """
    Generate training data from synthetic EEG.
    Returns X (feature matrix) and y (labels).
    """
    X, y = [], []

    print(f"Generating {n_samples} training samples...")

    for i in range(n_samples // 2):
        t, sig = generate_normal_eeg(duration=duration, fs=fs, seed=i)
        features, fvec, *_ = extract_all_features(sig, fs)
        X.append(fvec)
        y.append(0)  # Normal

    for i in range(n_samples // 2):
        t, sig = generate_epileptic_eeg(duration=duration, fs=fs, seed=i + 10000)
        features, fvec, *_ = extract_all_features(sig, fs)
        X.append(fvec)
        y.append(1)  # Epileptic

    return np.array(X), np.array(y)


def train_model(n_samples=300):
    """Train and save the SVM model."""
    X, y = build_training_data(n_samples=n_samples)

    # Replace NaN/Inf just in case
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"Cross-validation accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model, scores.mean()


def load_or_train_model():
    """Load saved model or train a new one."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        model, _ = train_model()
        return model


def predict(model, feature_vector):
    """
    Predict seizure probability for a single EEG segment.
    Returns: (label, probability_of_seizure)
    """
    fvec = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)
    fvec = fvec.reshape(1, -1)
    prob = model.predict_proba(fvec)[0]
    label = "Epileptic" if prob[1] > 0.5 else "Normal"
    return label, float(prob[1])
