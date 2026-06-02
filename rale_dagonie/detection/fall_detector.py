import numpy as np
import librosa
from collections import deque

IMPACT_PEAK_THRESHOLD = 0.45
IMPACT_DELTA_THRESHOLD = 0.25

SILENCE_RMS_THRESHOLD = 0.015
SILENCE_FRAMES_REQUIRED = 2


class FallDetector:
    def __init__(self):
        self.state = "idle"
        self.silence_count = 0

    def reset(self):
        self.state = "idle"
        self.silence_count = 0

    def process(self, audio_data):
        y = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        if len(y) < 10:
            return False

        rms = np.mean(librosa.feature.rms(y=y))
        peak = np.max(np.abs(y))
        delta = np.max(np.abs(np.diff(y)))

        if self.state == "idle":
            if peak > IMPACT_PEAK_THRESHOLD or delta > IMPACT_DELTA_THRESHOLD:
                self.state = "impact"
                self.silence_count = 0
                return False

        elif self.state == "impact":
            if rms < SILENCE_RMS_THRESHOLD:
                self.silence_count += 1

                if self.silence_count >= SILENCE_FRAMES_REQUIRED:
                    self.reset()
                    return True

            else:
                self.reset()

        return False