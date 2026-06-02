import pyaudio
from config import RATE, CHUNK

class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

    def record_chunk(self, duration_seconds):
        frames = []
        for _ in range(int(RATE / CHUNK * duration_seconds)):
            data = self.stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        return b''.join(frames)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()