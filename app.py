import base64
import io
import os
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, redirect, render_template, request, url_for, flash
from werkzeug.utils import secure_filename

from src.neural_network.detection import FruitVegDetector
from src.neural_network.inference import FruitVegPredictor, ModelNotReadyError

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MODEL_DIR = Path("models")
MIN_CONFIDENCE = float(os.environ.get("FRUITVEG_MIN_CONFIDENCE", "0.60"))
MIN_MARGIN = float(os.environ.get("FRUITVEG_MIN_MARGIN", "0.10"))
DETECT_CONFIDENCE = float(os.environ.get("FRUITVEG_DETECT_CONFIDENCE", "0.35"))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("FRUITVEG_SECRET", "dev-secret-change-me")
_predictor: Optional[FruitVegPredictor] = None
_predictor_signature: Optional[tuple] = None
_detector: Optional[FruitVegDetector] = None
_detector_signature: Optional[tuple] = None


def allowed_file(filename: str) -> bool:
    # Acceptăm doar fișiere imagine suportate.
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_predictor() -> FruitVegPredictor:
    # Cachează predictorul pentru a evita reîncărcarea la fiecare request.
    global _predictor
    global _predictor_signature

    model_path = MODEL_DIR / "fruitveg_cnn.pt"
    label_map_path = MODEL_DIR / "label_map.json"

    signature = None
    if model_path.exists() and label_map_path.exists():
        signature = (
            model_path.stat().st_mtime_ns,
            label_map_path.stat().st_mtime_ns,
        )

    if _predictor is None or signature != _predictor_signature:
        _predictor = FruitVegPredictor(
            model_path=model_path,
            label_map_path=label_map_path,
        )
        _predictor_signature = signature
    return _predictor


def get_detector() -> FruitVegDetector:
    # Folosește modelul optimizat dacă există, altfel fallback pe detectorul vechi.
    global _detector
    global _detector_signature

    optimized_path = MODEL_DIR / "optimized_model.pt"
    model_path = optimized_path if optimized_path.exists() else MODEL_DIR / "fruitveg_detector.pt"

    signature = None
    if model_path.exists():
        signature = (model_path.stat().st_mtime_ns,)

    if _detector is None or signature != _detector_signature:
        _detector = FruitVegDetector(model_path=model_path)
        _detector_signature = signature
    return _detector


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Flux clasic: upload → detectare → afișare în UI.
    file = request.files.get("image")
    if file is None or file.filename == "":
        flash("Te rog alege o imagine pentru a continua.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("Format invalid. Folosește PNG, JPG sau JPEG.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    image_bytes = file.read()

    try:
        detector = get_detector()
        detections = detector.detect(image_bytes, conf=DETECT_CONFIDENCE)
    except ModelNotReadyError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    predictions = [
        {"label": det.get("label"), "probability": det.get("confidence", 0.0)}
        for det in detections
    ]
    upload_warning = None
    if not predictions:
        upload_warning = "Nu am detectat niciun fruct/legumă în imagine."

    preview_data = base64.b64encode(image_bytes).decode("utf-8")

    return render_template(
        "index.html",
        predictions=predictions,
        detections=detections,
        preview_data=preview_data,
        filename=filename,
        upload_warning=upload_warning,
    )


@app.post("/api/predict")
def api_predict():
    # Endpoint de clasificare cu filtru de încredere.
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    image_bytes = file.read()

    try:
        predictor = get_predictor()
        accepted, predictions = predictor.predict_with_rejection(
            image_bytes,
            top_k=5,
            min_confidence=MIN_CONFIDENCE,
            min_margin=MIN_MARGIN,
        )
    except ModelNotReadyError as exc:
        return jsonify({"error": str(exc)}), 503

    message = None
    if not accepted and predictions:
        top1 = predictions[0]
        message = (
            "Modelul nu are suficientă încredere în această imagine. "
            f"Top-1: '{top1.get('label')}' ({top1.get('probability', 0.0) * 100:.2f}%)."
        )

    return jsonify(
        {
            "accepted": accepted,
            "min_confidence": MIN_CONFIDENCE,
            "min_margin": MIN_MARGIN,
            "message": message,
            "predictions": predictions,
        }
    )


@app.post("/api/detect")
def api_detect():
    # Endpoint pentru detecție (YOLO) folosit de UI live.
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image uploaded"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    image_bytes = file.read()

    try:
        detector = get_detector()
        detections = detector.detect(image_bytes, conf=DETECT_CONFIDENCE)
    except ModelNotReadyError as exc:
        return jsonify({"error": str(exc)}), 503

    return jsonify(
        {
            "detections": detections,
            "detect_confidence": DETECT_CONFIDENCE,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
