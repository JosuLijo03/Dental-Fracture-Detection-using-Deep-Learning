from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
import io, time, base64, cv2
import numpy as np
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

# ============================================
# GRAD-CAM++ — fixed, won't affect inference
# ============================================
class GradCAMPlusPlus:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer = self.model.features[5][0].block[0][0]
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor):
        # Always reset before GradCAM
        self.model.eval()
        self.model.zero_grad()

        inp = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(inp)
        output[0][0].backward()

        grad = self.gradients
        act  = self.activations

        # GradCAM++ weighting
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

        # Reset after GradCAM
        self.model.zero_grad()
        self.model.eval()

        return cam


def apply_gradcam_overlay(original_img, cam):
    img_array = np.array(original_img.convert("RGB"))
    h, w = img_array.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * img_array + 0.45 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


gradcam = GradCAMPlusPlus(clf_model)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()
    start = time.time()

    img_pil = Image.open(io.BytesIO(img_bytes))
    img_gray = img_pil.convert("L")

    # --- Step 1: Classification (always first) ---
    with torch.no_grad():
        tensor = transform(img_gray).unsqueeze(0).to(device)
        output = clf_model(tensor)
        prob = torch.sigmoid(output).item()

    if prob < 0.5:
        label = "Fractured"
        confidence = round((1 - prob) * 100, 1)
    else:
        label = "Non-Fractured"
        confidence = round(prob * 100, 1)

    # --- Step 2: GradCAM++ (after classification) ---
    try:
        tensor_grad = transform(img_gray).unsqueeze(0).to(device)
        cam = gradcam.generate(tensor_grad)
        gradcam_img = apply_gradcam_overlay(img_pil, cam)
        buf_cam = io.BytesIO()
        gradcam_img.save(buf_cam, format="JPEG", quality=92)
        gradcam_b64 = base64.b64encode(buf_cam.getvalue()).decode("utf-8")
    except Exception as e:
        print("GradCAM++ error:", e)
        gradcam_b64 = ""

    # --- Step 3: YOLOv8 (always last) ---
    img_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = yolo_model(img_rgb, conf=0.25)
    result = results[0]

    draw = ImageDraw.Draw(img_rgb)
    boxes_found = len(result.boxes) if result.boxes is not None else 0

    if boxes_found > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = round(float(box.conf[0]) * 100, 1)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=2)
            draw.text((x1 + 4, max(0, y1 - 16)), f"Fracture {conf_score}%", fill=(255, 50, 50))

    buf_det = io.BytesIO()
    img_rgb.save(buf_det, format="JPEG", quality=92)
    annotated_b64 = base64.b64encode(buf_det.getvalue()).decode("utf-8")

    elapsed = round((time.time() - start) * 1000)

    return jsonify({
        "classification": label,
        "confidence": confidence,
        "elapsed": elapsed,
        "boxes_found": boxes_found,
        "annotated_image": annotated_b64,
        "gradcam_image": gradcam_b64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
