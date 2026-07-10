"""
features.py

Extracts audio features from a single audio chunk.
"""

import librosa
import numpy as np

# ==========================================================
# Feature Extraction Parameters
# ==========================================================

N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128


# ==========================================================
# Feature Extraction
# ==========================================================

def extract_features(y, sr):
    """
    Extract features from one audio chunk.

    Returns
    -------
    dict
        Dictionary containing 155 features.
    """

    features = {}

    # ============================
    # MFCC
    # ============================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    for i, value in enumerate(mfcc_mean):
        features[f"mfcc_{i+1}"] = value


    # ============================
    # Mel Spectrogram
    # ============================

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(mel)

    mel_mean = np.mean(mel_db, axis=1)

    for i, value in enumerate(mel_mean):
        features[f"mel_{i+1}"] = value


    # ============================
    # Chroma
    # ============================

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    chroma_mean = np.mean(chroma, axis=1)

    for i, value in enumerate(chroma_mean):
        features[f"chroma_{i+1}"] = value


    # ============================
    # Spectral Centroid
    # ============================

    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr
    )

    features["spectral_centroid"] = np.mean(centroid)


    # ============================
    # Zero Crossing Rate
    # ============================

    zcr = librosa.feature.zero_crossing_rate(y)

    features["zero_crossing_rate"] = np.mean(zcr)

    return features