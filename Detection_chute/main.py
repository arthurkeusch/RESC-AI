import cv2
import mediapipe as mp
import os
import warnings
import time
from collections import deque, Counter

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)

def get_frame_skip(fps):
    if fps <= 26:          # 25 FPS → 8–10 analysés
        target_fps = 9
    elif fps <= 35:        # 30 FPS → 8–12 analysés
        target_fps = 10
    elif fps <= 55:        # 50 FPS → 10–15 analysés
        target_fps = 12
    else:                  # 60+ FPS → 12–15 analysés
        target_fps = 14

    skip = max(1, int(fps // target_fps))
    print(f"🔧 Frame skip choisi : 1 frame analysée / {skip} frames")
    return skip

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

video_path = "videos/test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Impossible d'ouvrir la vidéo")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25
delay = int(1000 / fps)
FRAME_SKIP = get_frame_skip(fps)
frame_count = 0

HISTORY_LENGTH = 5
posture_history = deque(maxlen=HISTORY_LENGTH)

FALL_TIME_THRESHOLD = 1.0
fall_timestamps = deque(maxlen=2)
last_fall_alert = 0

TARGET_WIDTH = 640
TARGET_HEIGHT = 360
USE_RESIZE = False # Passer à True pour limiter la qualité de la vidéo

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    if USE_RESIZE:
        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    h_img, w_img, _ = frame.shape

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    posture = None
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        x_vals = [lm.x for lm in landmarks]
        y_vals = [lm.y for lm in landmarks]
        x_min = int(min(x_vals) * w_img)
        x_max = int(max(x_vals) * w_img)
        y_min = int(min(y_vals) * h_img)
        y_max = int(max(y_vals) * h_img)

        y_values = [
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
            landmarks[mp_pose.PoseLandmark.LEFT_HIP].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y,
            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y,
        ]
        vertical_span = max(y_values) - min(y_values)
        posture = "A TERRE" if vertical_span < 0.25 else "DEBOUT"

        posture_history.append(posture)
        most_common = Counter(posture_history).most_common(1)[0][0]

        color = (0, 0, 255) if most_common == "A TERRE" else (0, 255, 0)

        current_time = time.time()
        fall_timestamps.append((most_common, current_time))

        if len(fall_timestamps) >= 2:
            prev_posture, prev_time = fall_timestamps[-2]
            if prev_posture == "DEBOUT" and most_common == "A TERRE":
                if (current_time - prev_time) <= FALL_TIME_THRESHOLD:
                    if current_time - last_fall_alert > FALL_TIME_THRESHOLD:
                        print("CHUTE DETECTÉE !")
                        last_fall_alert = current_time

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    if posture:
        cv2.putText(frame, f"Posture: {most_common}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Pose Detection (Optimisé)", frame)
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()