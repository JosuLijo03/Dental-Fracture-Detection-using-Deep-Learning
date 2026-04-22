from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import io, time, base64
from ultralytics import YOLO

app = Flask(__name__, template_folder="templates")
device = torch.device("cpu")

clf_model = torch.load("efficientnet_full.pth", map_location=device, weights_only=False)
clf_model.eval()

yolo_model = YOLO("best.pt")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()
    start = time.time()

    img = Image.open(io.BytesIO(img_bytes)).convert("L")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = clf_model(tensor)
        prob = torch.sigmoid(output).item()

    if prob < 0.5:
        label = "Fractured"
        confidence = round((1 - prob) * 100, 1)
    else:
        label = "Non-Fractured"
        confidence = round(prob * 100, 1)

    img_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = yolo_model(img_rgb, conf=0.15
                        
                        )
    result = results[0]

    draw = ImageDraw.Draw(img_rgb)
    boxes_found = len(result.boxes) if result.boxes is not None else 0

    if boxes_found > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = round(float(box.conf[0]) * 100, 1)
            color = (255, 77, 109) if label == "Fractured" else (0, 230, 180)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, max(0, y1 - 18)), f"Fracture {conf_score}%", fill=color)

    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=90)
    annotated_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    elapsed = round((time.time() - start) * 1000)

    return jsonify({
        "classification": label,
        "confidence": confidence,
        "elapsed": elapsed,
        "boxes_found": boxes_found,
        "annotated_image": annotated_b64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
