"""
preprocess.py

Contains all audio preprocessing functions used during inference.
"""

import librosa
import numpy as np

# ==========================================================
# Audio Constants
# ==========================================================

TARGET_SR = 16000

WINDOW_SEC = 3.0
HOP_SEC = 1.5


# ==========================================================
# Window Audio
# ==========================================================

def window_audio(y, sr, window_sec=WINDOW_SEC, hop_sec=HOP_SEC):
    """
    Split an audio signal into overlapping windows.

    Parameters
    ----------
    y : np.ndarray
        Audio waveform.

    sr : int
        Sampling rate.

    Returns
    -------
    list
        List of audio chunks.
    """

    window_len = int(window_sec * sr)
    hop_len = int(hop_sec * sr)

    # Audio shorter than one window
    if len(y) <= window_len:
        return [np.pad(y, (0, window_len - len(y)))]

    chunks = []

    for start in range(0, len(y) - window_len + 1, hop_len):
        chunks.append(y[start:start + window_len])

    # Add final chunk
    if (len(y) - window_len) % hop_len != 0:
        chunks.append(y[-window_len:])

    return chunks


# ==========================================================
# Preprocess Audio
# ==========================================================

def preprocess_audio(audio_path):
    """
    Complete preprocessing pipeline.

    Steps
    -----
    1. Load audio
    2. Resample to 16 kHz
    3. Convert to mono
    4. Remove silence
    5. Remove DC offset
    6. Peak normalization
    7. Windowing

    Returns
    -------
    chunks : list
        List of processed audio chunks.

    sr : int
        Sampling rate.
    """

    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SR,
        mono=True
    )

    # Skip tiny recordings
    if len(y) < int(0.2 * TARGET_SR):
        raise ValueError("Audio file is too short.")

    # Trim silence
    y, _ = librosa.effects.trim(
        y,
        top_db=25
    )

    # Remove DC offset
    y = y - np.mean(y)

    # Peak normalization
    peak = np.max(np.abs(y))

    if peak > 0:
        y = y / peak

    # Windowing
    chunks = window_audio(y, sr)

    return chunks, sr