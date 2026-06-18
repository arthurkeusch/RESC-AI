FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY .. .

RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s-pose.pt')"
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m-pose.pt')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]