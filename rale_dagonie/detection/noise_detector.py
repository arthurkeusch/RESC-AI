import numpy as np
import librosa

PEAK_THRESHOLD = 0.4
RMS_THRESHOLD = 0.03
DELTA_THRESHOLD = 0.2

def detect_loud_noise(audio_data):
    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    if len(y) < 10:
        return False
    rms = np.mean(librosa.feature.rms(y=y))
    peak = np.max(np.abs(y))
    delta = np.max(np.abs(np.diff(y)))
    if peak > PEAK_THRESHOLD:
        return True
    if rms > RMS_THRESHOLD and delta > DELTA_THRESHOLD:
        return True
    return False