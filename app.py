from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
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

    # --- Classification ---
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

    # --- YOLOv8 Detection ---
    img_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    orig_w, orig_h = img_rgb.size

    results = yolo_model(img_rgb, conf=0.25)
    result = results[0]

    draw = ImageDraw.Draw(img_rgb)
    boxes_found = len(result.boxes) if result.boxes is not None else 0

    # Scale line width based on image size
    line_width = max(3, int(min(orig_w, orig_h) / 100))
    BOX_COLOR = (255, 50, 50)        # bright red
    LABEL_BG  = (255, 50, 50)        # red background for label
    LABEL_FG  = (255, 255, 255)      # white text

    if boxes_found > 0:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score = round(float(box.conf[0]) * 100, 1)

            # Draw thick outer glow (slightly larger box, semi-transparent red)
            glow_offset = line_width + 2
            draw.rectangle(
                [x1 - glow_offset, y1 - glow_offset, x2 + glow_offset, y2 + glow_offset],
                outline=(255, 80, 80), width=max(1, line_width - 1)
            )

            # Draw main box
            draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_width)

            # Draw corner accents (L-shaped corners for a medical scanner look)
            corner = max(12, int(min(x2-x1, y2-y1) * 0.15))
            cw = max(2, line_width + 1)
            # top-left
            draw.line([x1, y1, x1+corner, y1], fill=BOX_COLOR, width=cw)
            draw.line([x1, y1, x1, y1+corner], fill=BOX_COLOR, width=cw)
            # top-right
            draw.line([x2, y1, x2-corner, y1], fill=BOX_COLOR, width=cw)
            draw.line([x2, y1, x2, y1+corner], fill=BOX_COLOR, width=cw)
            # bottom-left
            draw.line([x1, y2, x1+corner, y2], fill=BOX_COLOR, width=cw)
            draw.line([x1, y2, x1, y2-corner], fill=BOX_COLOR, width=cw)
            # bottom-right
            draw.line([x2, y2, x2-corner, y2], fill=BOX_COLOR, width=cw)
            draw.line([x2, y2, x2, y2-corner], fill=BOX_COLOR, width=cw)

            # Label background + text
            label_text = f" FRACTURE {conf_score}% "
            font_size = max(12, int(min(orig_w, orig_h) / 30))
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except:
                font = ImageFont.load_default()

            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            label_y = max(0, y1 - text_h - 6)
            draw.rectangle([x1, label_y, x1 + text_w, label_y + text_h + 4], fill=LABEL_BG)
            draw.text((x1, label_y + 2), label_text, fill=LABEL_FG, font=font)

    # Convert to base64
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=92)
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
