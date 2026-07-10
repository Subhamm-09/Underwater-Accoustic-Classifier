"""
predict.py

Loads the trained model and predicts the class of an audio file.
"""

from pathlib import Path
from collections import Counter

import joblib
import pandas as pd

from src.preprocess import preprocess_audio
from src.features import extract_features


# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "logistic_regression.pkl"
SCALER_PATH = MODEL_DIR / "standard_scaler.pkl"


# ==========================================================
# Load Model
# ==========================================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# ==========================================================
# Prediction Function
# ==========================================================

def predict_audio(audio_path):
    """
    Predict class of an audio file.

    Parameters
    ----------
    audio_path : str or Path

    Returns
    -------
    predicted_class : str

    confidence : float
    """

    # -----------------------------
    # Preprocess
    # -----------------------------

    chunks, sr = preprocess_audio(audio_path)

    feature_rows = []

    for chunk in chunks:

        feature_rows.append(
            extract_features(chunk, sr)
        )

    feature_df = pd.DataFrame(feature_rows)

    # -----------------------------
    # Match training feature order
    # -----------------------------

    feature_df = feature_df[scaler.feature_names_in_]

    # -----------------------------
    # Scale
    # -----------------------------

    X_scaled = scaler.transform(feature_df)

    # -----------------------------
    # Predict every chunk
    # -----------------------------

    predictions = model.predict(X_scaled)

    probabilities = model.predict_proba(X_scaled)

    # -----------------------------
    # Majority Voting
    # -----------------------------

    predicted_class = Counter(predictions).most_common(1)[0][0]

    # -----------------------------
    # Average confidence
    # -----------------------------

    avg_probability = probabilities.mean(axis=0)

    confidence_scores = dict(
        zip(
            model.classes_,
            avg_probability
        )
    )

    return predicted_class, confidence_scores
