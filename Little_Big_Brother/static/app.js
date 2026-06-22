const video = document.getElementById("video");
const resultImage = document.getElementById("result");

const modelSelect = document.getElementById("model-select");
const currentModelLabel = document.getElementById("current-model");

const personsContainer = document.getElementById("persons-container");

const fpsLabel = document.getElementById("fps");
const latencyLabel = document.getElementById("latency");
const systemStatus = document.getElementById("system-status");

let detectionActive = false;

let frameCount = 0;
let fpsTime = performance.now();

const FPS = 2;
const interval = 1000 / FPS;
let lastCapture = 0;

/* CAMERA */
navigator.mediaDevices.getUserMedia({ video: true })
.then(stream => {
    video.srcObject = stream;
});

/* STATUS CHECK */
function checkStatus() {

    fetch("/status")
        .then(res => res.json())
        .then(data => {

            detectionActive = data.active;

            systemStatus.innerText = detectionActive
                ? "Detection ACTIVE"
                : "Waiting for alert...";
        });
}

setInterval(checkStatus, 1000);
checkStatus();

/* LOAD MODELS */
fetch("/models")
.then(res => res.json())
.then(data => {

    modelSelect.innerHTML = "";

    Object.entries(data.models).forEach(([label, value]) => {

        const option = document.createElement("option");

        option.value = value;
        option.textContent = label;

        if (value === data.current)
            option.selected = true;

        modelSelect.appendChild(option);
    });

    currentModelLabel.innerText = "Current: " + data.current;
});

/* CHANGE MODEL */
modelSelect.addEventListener("change", () => {

    fetch("/change-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelSelect.value })
    })
    .then(res => res.json())
    .then(data => {
        currentModelLabel.innerText = "Current: " + data.current;
    });
});

/* CAPTURE LOOP */
function captureFrame(timestamp) {

    if (!detectionActive) {
        requestAnimationFrame(captureFrame);
        return;
    }

    if (timestamp - lastCapture < interval) {
        requestAnimationFrame(captureFrame);
        return;
    }

    lastCapture = timestamp;

    const canvas = document.createElement("canvas");
    canvas.width = 320;
    canvas.height = 240;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {

        const formData = new FormData();
        formData.append("file", blob, "frame.jpg");

        fetch("/detect", { method: "POST", body: formData })
            .then(res => res.json())
            .then(data => {

                personsContainer.innerHTML = "";

                data.persons.forEach(person => {

                    const card = document.createElement("div");
                    card.className = "card";

                    card.innerHTML = `
                        <h2>Person ${person.id}</h2>
                        <p>Posture: ${person.posture}</p>
                        <p>Ratio: ${person.ratio.toFixed(2)}</p>
                        <p>Stability: ${person.stability}</p>
                    `;

                    personsContainer.appendChild(card);
                });

                latencyLabel.innerText =
                    "Inference: " + data.inference_ms + " ms";

                resultImage.src =
                    "data:image/jpeg;base64," + data.image;

                frameCount++;

                const now = performance.now();

                if (now - fpsTime >= 1000) {
                    fpsLabel.innerText = "FPS: " + frameCount;
                    frameCount = 0;
                    fpsTime = now;
                }
            });

    }, "image/jpeg", 0.8);

    requestAnimationFrame(captureFrame);
}

requestAnimationFrame(captureFrame);