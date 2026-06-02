import numpy as np
import librosa

LOW_FREQ_BAND = 20      # index bas
HIGH_FREQ_BAND = 80     # index haut

RMS_MIN = 0.01
RMS_MAX = 0.08

ZCR_MAX = 0.15
VARIATION_MIN = 0.005


def detect_rale(audio_data, sr=16000):

    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    if len(y) < 100:
        return False

    rms = np.mean(librosa.feature.rms(y=y))

    if not (RMS_MIN < rms < RMS_MAX):
        return False

    D = np.abs(librosa.stft(y))
    low_energy = np.mean(D[LOW_FREQ_BAND:HIGH_FREQ_BAND, :])
    total_energy = np.mean(D)

    low_ratio = low_energy / (total_energy + 1e-6)

    envelope = librosa.feature.rms(y=y)[0]
    variation = np.std(envelope)

    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    if (
        low_ratio > 0.6 and
        variation > VARIATION_MIN and
        zcr < ZCR_MAX
    ):
        return True

    return False