from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw
import io, time, base64, cv2
import numpy as np
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn

app = Flask(__name__, template_folder="templates")
device = torch.device("cpu")

# ============================================
# Classification model (EfficientNetV2-S)
# ============================================
classifier = models.efficientnet_v2_s(weights=None)
classifier.classifier[1] = nn.Linear(
    classifier.classifier[1].in_features,
    2
)
classifier.load_state_dict(
    torch.load("classifier_60.pth", map_location=device)
)
classifier = classifier.to(device)
classifier.eval()

# ============================================
# YOLOv8
# ============================================
yolo_model = YOLO("best.pt")

# ============================================
# EigenCAM — targets classifier.features[-1]
# (matches the Colab pipeline approach)
# ============================================
eigen_target_layers = [classifier.features[-1]]
eigen_cam = EigenCAM(
    model=classifier,
    target_layers=eigen_target_layers
)

# ============================================
# Transforms
# ============================================
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

CLASS_NAMES   = ["normal", "fracture"]
FRACTURE_CLASS = 0
NORMAL_CLASS   = 1


# ============================================
# EigenCAM helper  (mirrors Colab function)
# ============================================
def generate_eigencam_b64(img_pil):
    """
    Accepts a PIL image, returns a base64-encoded JPEG of the
    EigenCAM overlay — or an empty string on error.
    """
    try:
        # Resize to 224×224 (same as Colab)
        resized   = img_pil.convert("RGB").resize((224, 224))
        rgb_array = np.array(resized).astype(np.float32) / 255.0

        input_tensor = clf_transform(resized).unsqueeze(0).to(device)

        grayscale_cam = eigen_cam(input_tensor=input_tensor)[0]

        visualization = show_cam_on_image(
            rgb_array,
            grayscale_cam,
            use_rgb=True
        )

        buf = io.BytesIO()
        Image.fromarray(visualization).save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    except Exception as e:
        print("EigenCAM error:", e)
        return ""


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file      = request.files["file"]
    img_bytes = file.read()
    start     = time.time()

    img_pil = Image.open(io.BytesIO(img_bytes))
    img_rgb = img_pil.convert("RGB")

    # --------------------------------------------------
    # Stage 1 — EfficientNetV2-S Classifier
    # --------------------------------------------------
    with torch.no_grad():
        tensor = clf_transform(img_rgb).unsqueeze(0).to(device)
        output = classifier(tensor)
        probs  = torch.softmax(output, dim=1)
        pred_class    = torch.argmax(probs).item()
        fracture_prob = probs[0][FRACTURE_CLASS].item()

    if pred_class == FRACTURE_CLASS:
        label      = "Fractured"
        confidence = round(fracture_prob * 100, 1)
    else:
        label      = "Non-Fractured"
        confidence = round((1 - fracture_prob) * 100, 1)

    # --------------------------------------------------
    # Stages 2 & 3 only run when fracture is confirmed
    # --------------------------------------------------
    if label == "Fractured":

        # --- Stage 2: YOLOv8 Detection ---
        img_for_yolo = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        with torch.no_grad():
            results = yolo_model(
                img_for_yolo,
                conf=0.05,
                iou=0.20
            )
        result      = results[0]
        boxes_found = len(result.boxes) if result.boxes is not None else 0

        draw = ImageDraw.Draw(img_for_yolo)
        if boxes_found > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_score = round(float(box.conf[0]) * 100, 1)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                draw.text(
                    (x1 + 4, max(0, y1 - 16)),
                    f"Fracture {conf_score}%",
                    fill=(0, 255, 0)
                )

        buf_det = io.BytesIO()
        img_for_yolo.save(buf_det, format="JPEG", quality=92)
        annotated_b64 = base64.b64encode(buf_det.getvalue()).decode("utf-8")

        # --- Stage 3: EigenCAM on Classifier ---
        eigen_b64 = generate_eigencam_b64(img_pil)

    else:
        # Non-Fractured: skip all heavy inference
        boxes_found   = 0
        annotated_b64 = ""
        eigen_b64     = ""

    elapsed = round((time.time() - start) * 1000)

    return jsonify({
        "classification":  label,
        "confidence":      confidence,
        "elapsed":         elapsed,
        "boxes_found":     boxes_found,
        "annotated_image": annotated_b64,
        "eigencam_image":  eigen_b64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)