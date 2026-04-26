from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
import io, time, base64, cv2
import numpy as np
import copy
from ultralytics import YOLO

app = Flask(__name__, template_folder="templates")
device = torch.device("cpu")

# Classification model
clf_model = torch.load("efficientnet_full.pth", map_location=device, weights_only=False)
clf_model.eval()

# YOLOv8 — two separate instances
# One for detection, one for GradCAM++
yolo_detect = YOLO("best.pt")   # for detection only
yolo_cam    = YOLO("best.pt")   # for GradCAM++ only
yolo_cam.model.eval()

# Transforms
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

yolo_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

# ============================================
# GRAD-CAM++ on YOLOv8
# ============================================
class YOLOGradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = self.model.model[9].cv1.conv
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()

        inp = input_tensor.clone().detach().requires_grad_(True)
        out = self.model(inp)

        score = out[0][0, 4, :].max()
        score.backward()

        if self.gradients is None or self.activations is None:
            return None

        grad = self.gradients
        act  = self.activations

        grad_sq = grad ** 2
        grad_cu = grad ** 3
        sum_act = act.sum(dim=[2, 3], keepdim=True)
        alpha   = grad_sq / (2 * grad_sq + sum_act * grad_cu + 1e-8)
        weights = (alpha * F.relu(grad)).sum(dim=[2, 3], keepdim=True)

        cam = (weights * act).sum(dim=1).squeeze()
        cam = F.relu(cam).numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        self.model.zero_grad()
        return cam


def apply_gradcam_overlay(original_img, cam):
    img_array = np.array(original_img.convert("RGB").resize((640, 640)))
    cam_resized = cv2.resize(cam, (640, 640))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_array + 0.45 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


gradcam = YOLOGradCAMPlusPlus(yolo_cam.model)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()
    start = time.time()

    img_pil = Image.open(io.BytesIO(img_bytes))
    img_rgb  = img_pil.convert("RGB")

    # --- Step 1: Classification ---
    clf_model.eval()
    with torch.no_grad():
        tensor = clf_transform(img_pil.convert("L")).unsqueeze(0).to(device)
        output = clf_model(tensor)
        prob   = torch.sigmoid(output).item()

    if prob < 0.5:
        label      = "Fractured"
        confidence = round((1 - prob) * 100, 1)
    else:
        label      = "Non-Fractured"
        confidence = round(prob * 100, 1)

    # --- Step 2: YOLOv8 Detection (dedicated instance) ---
    results     = yolo_detect(img_rgb, conf=0.25)
    result      = results[0]
    boxes_found = len(result.boxes) if result.boxes is not None else 0

    img_annotated = img_rgb.copy()
    draw = ImageDraw.Draw(img_annotated)

    if boxes_found > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = round(float(box.conf[0]) * 100, 1)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=2)
            draw.text((x1 + 4, max(0, y1 - 16)),
                      f"Fracture {conf_score}%", fill=(255, 50, 50))

    buf_det = io.BytesIO()
    img_annotated.save(buf_det, format="JPEG", quality=92)
    annotated_b64 = base64.b64encode(buf_det.getvalue()).decode("utf-8")

    # --- Step 3: GradCAM++ (dedicated instance) ---
    try:
        yolo_tensor = yolo_transform(img_rgb).unsqueeze(0).to(device)
        cam = gradcam.generate(yolo_tensor)
        if cam is not None:
            gradcam_img = apply_gradcam_overlay(img_pil, cam)
            buf_cam = io.BytesIO()
            gradcam_img.save(buf_cam, format="JPEG", quality=92)
            gradcam_b64 = base64.b64encode(buf_cam.getvalue()).decode("utf-8")
        else:
            gradcam_b64 = ""
    except Exception as e:
        print("GradCAM++ error:", e)
        gradcam_b64 = ""

    elapsed = round((time.time() - start) * 1000)

    return jsonify({
        "classification": label,
        "confidence":     confidence,
        "elapsed":        elapsed,
        "boxes_found":    boxes_found,
        "annotated_image": annotated_b64,
        "gradcam_image":   gradcam_b64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
