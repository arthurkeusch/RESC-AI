import numpy as np
import librosa
from config import RATE

def extract_features(audio_data):
    y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=RATE))

    return rms, zcr, mfcc