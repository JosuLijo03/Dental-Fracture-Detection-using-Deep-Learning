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

# Deep copy for GradCAM++ only
gradcam_clf = copy.deepcopy(clf_model)
gradcam_clf.eval()

# YOLO for detection only — never touched by GradCAM++
yolo_model = YOLO("best.pt")

# Transforms
clf_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# ============================================
# GRAD-CAM++ on EfficientNet copy
# ============================================
gradcam_grads = {}
gradcam_acts  = {}

target = gradcam_clf.features[5][-1].block[3][0]
target.register_forward_hook(
    lambda m,i,o: gradcam_acts.update({'f': o.detach()}))
target.register_full_backward_hook(
    lambda m,gi,go: gradcam_grads.update({'g': go[0].detach()}))

def generate_gradcam(img_gray):
    gradcam_grads.clear()
    gradcam_acts.clear()
    gradcam_clf.eval()
    gradcam_clf.zero_grad()

    tensor = clf_transform(img_gray).unsqueeze(0)
    inp    = tensor.clone().detach().requires_grad_(True)
    out    = gradcam_clf(inp)
    out[0][0].backward()

    if 'g' not in gradcam_grads or 'f' not in gradcam_acts:
        return None

    grad    = gradcam_grads['g']
    act     = gradcam_acts['f']
    grad_sq = grad**2
    grad_cu = grad**3
    sum_act = act.sum(dim=[2,3], keepdim=True)
    alpha   = grad_sq / (2*grad_sq + sum_act*grad_cu + 1e-8)
    weights = (alpha * F.relu(grad)).sum(dim=[2,3], keepdim=True)
    cam     = F.relu((weights*act).sum(dim=1)).squeeze().numpy()

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 0:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = np.zeros_like(cam)

    gradcam_clf.zero_grad()
    return cam


def apply_gradcam_overlay(original_img, cam):
    img_array   = np.array(original_img.convert("RGB"))
    h, w        = img_array.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay     = (0.55 * img_array + 0.45 * heatmap).astype(np.uint8)
    return Image.fromarray(overlay)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file      = request.files["file"]
    img_bytes = file.read()
    start     = time.time()

    img_pil  = Image.open(io.BytesIO(img_bytes))
    img_rgb  = img_pil.convert("RGB")
    img_gray = img_pil.convert("L")

    # --- Step 1: Classification ---
    clf_model.eval()
    with torch.no_grad():
        tensor = clf_transform(img_gray).unsqueeze(0)
        prob   = torch.sigmoid(clf_model(tensor)).item()

    if prob < 0.5:
        label      = "Fractured"
        confidence = round((1 - prob) * 100, 1)
    else:
        label      = "Non-Fractured"
        confidence = round(prob * 100, 1)

    # --- Step 2: YOLOv8 Detection ---
    img_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    results = yolo_model(img_rgb, conf=0.25)
    result  = results[0]

    draw        = ImageDraw.Draw(img_rgb)
    boxes_found = len(result.boxes) if result.boxes is not None else 0

    if boxes_found > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = round(float(box.conf[0]) * 100, 1)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 50, 50), width=2)
            draw.text((x1+4, max(0, y1-16)),
                      f"Fracture {conf_score}%", fill=(255, 50, 50))

    buf_det = io.BytesIO()
    img_rgb.save(buf_det, format="JPEG", quality=92)
    annotated_b64 = base64.b64encode(buf_det.getvalue()).decode("utf-8")

    # --- Step 3: GradCAM++ ---
    try:
        cam = generate_gradcam(img_gray)
        if cam is not None:
            gradcam_img = apply_gradcam_overlay(img_pil, cam)
            buf_cam     = io.BytesIO()
            gradcam_img.save(buf_cam, format="JPEG", quality=92)
            gradcam_b64 = base64.b64encode(buf_cam.getvalue()).decode("utf-8")
        else:
            gradcam_b64 = ""
    except Exception as e:
        print("GradCAM++ error:", e)
        gradcam_b64 = ""

    elapsed = round((time.time() - start) * 1000)

    return jsonify({
        "classification":  label,
        "confidence":      confidence,
        "elapsed":         elapsed,
        "boxes_found":     boxes_found,
        "annotated_image": annotated_b64,
        "gradcam_image":   gradcam_b64
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
