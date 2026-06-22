from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import cv2
import numpy as np
import base64
import time

from ultralytics import YOLO

import torch
from torch.serialization import safe_globals
import torch.nn.modules.container
import ultralytics.nn.tasks

safe_globals([
    torch.nn.modules.container.Sequential,
    ultralytics.nn.tasks.PoseModel,
])

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

model = None

CURRENT_MODEL = "yolov8n-pose.pt"

DETECTION_ACTIVE = False

AVAILABLE_MODELS = {
    "YOLOv8 Nano": "yolov8n-pose.pt",
    "YOLOv8 Small": "yolov8s-pose.pt",
    "YOLOv8 Medium": "yolov8m-pose.pt"
}


def detect_posture(keypoints):

    xs = keypoints[:, 0]
    ys = keypoints[:, 1]

    width = xs.max() - xs.min()
    height = ys.max() - ys.min()

    if width <= 0:
        return "UNKNOWN", 0.0

    ratio = float(height / width)

    if ratio >= 1.2:
        return "STANDING", ratio

    return "LYING", ratio


def analyze_upright_stability(keypoints):

    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    shoulder_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2

    hip_x = (left_hip[0] + right_hip[0]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2

    vertical_dist = abs(hip_y - shoulder_y)
    horizontal_dist = abs(hip_x - shoulder_x)

    if vertical_dist == 0:
        return "UNSTABLE"

    tilt_ratio = horizontal_dist / vertical_dist

    if tilt_ratio > 0.6:
        return "LEANING"

    if vertical_dist < 50:
        return "LOW POSTURE"

    return "STABLE"


@app.on_event("startup")
def load_model():

    global model

    model = YOLO(CURRENT_MODEL)

    try:
        model.fuse()
    except Exception:
        pass


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/models")
async def get_models():

    return {
        "current": CURRENT_MODEL,
        "models": AVAILABLE_MODELS
    }


@app.get("/status")
async def status():

    return {
        "active": DETECTION_ACTIVE,
        "current_model": CURRENT_MODEL
    }


@app.get("/alert")
async def alert():

    global DETECTION_ACTIVE

    DETECTION_ACTIVE = True

    return {
        "success": True,
        "message": "Detection activated"
    }


@app.get("/reset")
async def reset():

    global DETECTION_ACTIVE

    DETECTION_ACTIVE = False

    return {
        "success": True,
        "message": "Detection stopped"
    }


@app.post("/change-model")
async def change_model(request: Request):

    global model
    global CURRENT_MODEL

    data = await request.json()

    model_name = data.get("model")

    if model_name not in AVAILABLE_MODELS.values():
        return JSONResponse(
            {"error": "Model not found"},
            status_code=400
        )

    CURRENT_MODEL = model_name

    model = YOLO(CURRENT_MODEL)

    try:
        model.fuse()
    except Exception:
        pass

    return {
        "success": True,
        "current": CURRENT_MODEL
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):

    if not DETECTION_ACTIVE:

        return JSONResponse({
            "active": False,
            "persons": [],
            "image": None,
            "inference_ms": 0
        })

    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    start = time.time()

    results = model.predict(
        frame,
        imgsz=320,
        conf=0.75,
        verbose=False
    )

    inference_time = round(
        (time.time() - start) * 1000,
        1
    )

    annotated_frame = frame

    persons = []

    if results and len(results) > 0:

        result = results[0]

        annotated_frame = result.plot()

        if (
            result.keypoints is not None
            and result.keypoints.xy is not None
            and len(result.keypoints.xy) > 0
        ):

            all_persons = (
                result.keypoints.xy
                .cpu()
                .numpy()
            )

            for person_id, person_kpts in enumerate(all_persons):

                posture, ratio = detect_posture(
                    person_kpts
                )

                stability = "N/A"

                if posture == "STANDING":

                    stability = analyze_upright_stability(
                        person_kpts
                    )

                persons.append({
                    "id": person_id + 1,
                    "posture": posture,
                    "ratio": float(ratio),
                    "stability": stability
                })

    _, buffer = cv2.imencode(
        ".jpg",
        annotated_frame
    )

    image_base64 = base64.b64encode(
        buffer
    ).decode("utf-8")

    return JSONResponse({
        "active": True,
        "persons": persons,
        "image": image_base64,
        "inference_ms": float(inference_time)
    })