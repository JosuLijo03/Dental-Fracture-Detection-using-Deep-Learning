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

# Load models
clf_model = torch.load("efficientnet_full.pth", map_location=device, weights_only=False)
clf_model.eval()

yolo_model = YOLO("best.pt")

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# ============================================
# GRAD-CAM
# ============================================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        # Hook into last conv block of EfficientNet
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # EfficientNet-B0 last conv layer
        target_layer = self.model.features[8]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.zero_grad()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        score = output[0][0]
        score.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        cam = cam.squeeze().numpy()
        return cam

gradcam = GradCAM(clf_model)

def apply_gradcam_overlay(original_img, cam):
    """Apply Grad-CAM heatmap overlay on original image"""
    img_array = np.array(original_img.convert("RGB"))
    h, w = img_array.shape[:2]

    # Resize CAM to image size
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (0.5 * img_array + 0.5 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()
    start = time.time()

    # --- Classification ---
    img_pil = Image.open(io.BytesIO(img_bytes))
    img_gray = img_pil.convert("L")
    tensor = transform(img_gray).unsqueeze(0).to(device)

    with torch.no_grad():
        output = clf_model(tensor)
        prob = torch.sigmoid(output).item()

    if prob < 0.5:
        label = "Fractured"
        confidence = round((1 - prob) * 100, 1)
    else:
        label = "Non-Fractured"
        confidence = round(prob * 100, 1)

    # --- Grad-CAM ---
    tensor_grad = transform(img_gray).unsqueeze(0).to(device)
    cam = gradcam.generate(tensor_grad)
    gradcam_img = apply_gradcam_overlay(img_pil, cam)

    buf_cam = io.BytesIO()
    gradcam_img.save(buf_cam, format="JPEG", quality=92)
    gradcam_b64 = base64.b64encode(buf_cam.getvalue()).decode("utf-8")

    # --- YOLOv8 Detection ---
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
# requirements.txt update needed
