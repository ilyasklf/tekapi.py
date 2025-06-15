from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import cv2
import base64
import io
import os
import tempfile
import logging
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# MODELLER
vehicle_model_path = "C:\\Users\\ilyas\\OneDrive\\Masaüstü\\apison\\best_vechile.pt"
lane_model_path = "C:\\Users\\ilyas\\OneDrive\\Masaüstü\\apison\\best_lane.pt"

vehicle_model = YOLO(vehicle_model_path)
lane_model = YOLO(lane_model_path)


# --- Yardımcı Fonksiyonlar ---
def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def npimg_to_base64(img_np):
    _, buffer = cv2.imencode(".jpg", img_np)
    return base64.b64encode(buffer).decode("utf-8")


# --- VEHICLE IMAGE ---
@app.route('/vehicle-image', methods=['POST'])
def vehicle_image():
    try:
        image_file = request.files['image']
        image = Image.open(image_file.stream)
        image = ImageOps.exif_transpose(image).convert("RGB")

        results = vehicle_model(image)[0]
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("arial.ttf", size=20)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = vehicle_model.names[cls_id]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1 + 5, y1 + 5), label, fill="red", font=font)

        return jsonify({"image": image_to_base64(image)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- VEHICLE VIDEO ---
@app.route('/vehicle-video', methods=['POST'])
def vehicle_video():
    try:
        video_bytes = base64.b64decode(request.json['video'])
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as in_file:
            in_file.write(video_bytes)
            in_path = in_file.name

        out_path = tempfile.mktemp(suffix='.mp4')
        cap = cv2.VideoCapture(in_path)

        if not cap.isOpened():
            return jsonify({"error": "Video açma hatası"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = vehicle_model(frame)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = vehicle_model.names[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        with open(out_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()

        os.unlink(in_path)
        os.unlink(out_path)
        return jsonify({"processed_video": encoded})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- LANE IMAGE ---
@app.route('/lane-image', methods=['POST'])
def lane_image():
    try:
        file = request.files["image"]
        image = Image.open(file.stream).convert("RGB")
        img_resized = image.resize((640, 640))
        img_array = np.array(img_resized)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = lane_model(img_cv, conf=0.05, imgsz=640)

        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (img_cv.shape[1], img_cv.shape[0]))
                    img_cv[mask > 128] = (0, 0, 255)

        return jsonify({"result": npimg_to_base64(img_cv)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- LANE VIDEO ---
@app.route('/lane-video', methods=['POST'])
def lane_video():
    try:
        video_bytes = base64.b64decode(request.json['video'])

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as in_file:
            in_file.write(video_bytes)
            in_path = in_file.name

        out_path = tempfile.mktemp(suffix='.mp4')
        cap = cv2.VideoCapture(in_path)

        if not cap.isOpened():
            return jsonify({"error": "Video açma hatası"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = lane_model(frame, conf=0.05, imgsz=640)[0]
            if results.masks is not None:
                masks = results.masks.data.cpu().numpy()
                for mask in masks:
                    binary_mask = (mask * 255).astype(np.uint8)
                    binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))
                    color_mask = cv2.merge([binary_mask * 0, binary_mask, binary_mask])
                    frame = cv2.addWeighted(frame, 1, color_mask, 0.5, 0)

            out.write(frame)

        cap.release()
        out.release()

        with open(out_path, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode()

        os.unlink(in_path)
        os.unlink(out_path)
        return jsonify({"processed_video": encoded})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Sağlık Kontrolü ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "vehicle_model": os.path.basename(vehicle_model_path),
        "lane_model": os.path.basename(lane_model_path),
        "vehicle_classes": vehicle_model.names,
        "lane_classes": lane_model.names
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
