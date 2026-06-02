import os
import urllib.request
import zipfile

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-fr-0.6-linto-2.2.0.zip"
ZIP_PATH = "model/vosk-model-fr-0.6-linto-2.2.0.zip"
MODEL_DIR = "model/vosk-model-fr-0.6-linto-2.2.0"

os.makedirs("model", exist_ok=True)

if not os.path.exists(MODEL_DIR):
    print("Téléchargement du modèle...")

    urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)

    print("Décompression...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("model")

    os.remove(ZIP_PATH)

    print("Modèle installé.")
else:
    print("Modèle déjà présent.")