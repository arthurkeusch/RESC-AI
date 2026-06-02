from collections import deque

from config import WINDOW_SECONDS, ALERT_THRESHOLD
from audio.recorder import AudioRecorder

from detection.rale_detector import detect_rale
from detection.help_detector import detect_help
from detection.noise_detector import detect_loud_noise
from detection.fall_detector import FallDetector

from speech.recognizer import SpeechRecognizer
from alert.alert_manager import trigger_alert


def main():
    recorder = AudioRecorder()
    recognizer = SpeechRecognizer()
    fall_detector = FallDetector()

    rale_history = deque(maxlen=5)
    help_history = deque(maxlen=5)
    noise_history = deque(maxlen=5)
    speech_history = deque(maxlen=5)

    print("Surveillance active...")

    try:
        while True:
            audio_data = recorder.record_chunk(WINDOW_SECONDS)
            rale = detect_rale(audio_data)
            rale_history.append(rale)

            noise = detect_loud_noise(audio_data)
            noise_history.append(noise)

            fall_detected = fall_detector.process(audio_data)

            text = recognizer.transcribe(audio_data)
            speech_history.append(text)

            if text:
                print("🗣️ :", text)

            help_call = detect_help(text)
            help_history.append(help_call)
            recent_speech = any(len(t.strip()) > 2 for t in speech_history)

            if fall_detected:
                trigger_alert("Chute probable détectée")

            if sum(rale_history) >= ALERT_THRESHOLD:
                trigger_alert("Râle suspect détecté")
                rale_history.clear()

            if sum(help_history) >= 1:
                trigger_alert("Appel à l'aide détecté")
                help_history.clear()

            if sum(noise_history) >= 2 and not recent_speech:
                trigger_alert("Bruit suspect sans parole")
                noise_history.clear()

    except KeyboardInterrupt:
        print("Arrêt")
        recorder.close()


if __name__ == "__main__":
    main()