import os
import json
from vosk import Model, KaldiRecognizer
from config import MODEL_PATH, RATE

class SpeechRecognizer:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Modèle Vosk manquant")

        self.model = Model(MODEL_PATH)
        self.recognizer = KaldiRecognizer(self.model, RATE)

    def transcribe(self, audio_data):
        if self.recognizer.AcceptWaveform(audio_data):
            result = json.loads(self.recognizer.Result())
            return result.get("text", "")
        return ""