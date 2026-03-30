from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request

from sanskrit_ocr.infer import OCRInferenceEngine


def create_app() -> Flask:
    app = Flask(__name__)

    checkpoint = Path(os.environ.get("OCR_CHECKPOINT", "checkpoints/sanskrit_crnn.pt"))
    device = os.environ.get("OCR_DEVICE", "cuda")
    decoder = os.environ.get("OCR_DECODER", "greedy")
    dictionary_path = Path(os.environ.get("OCR_DICTIONARY", "sanskrit_ocr/data/basic_dictionary.txt"))
    engine: OCRInferenceEngine | None = None

    def get_engine() -> OCRInferenceEngine:
        nonlocal engine
        if engine is None:
            engine = OCRInferenceEngine(
                checkpoint_path=checkpoint,
                device=device,
                dictionary_path=dictionary_path,
            )
        return engine

    @app.get("/health")
    def health() -> tuple:
        return jsonify({"status": "ok"}), 200

    @app.post("/ocr")
    def ocr() -> tuple:
        if "image" not in request.files:
            return jsonify({"error": "Missing image file"}), 400

        uploaded = request.files["image"]
        if uploaded.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        temp_dir = Path("outputs")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded.filename
        uploaded.save(temp_path)

        try:
            prediction = get_engine().predict(temp_path, decoder=decoder)
            return jsonify({"text": prediction.text, "confidence": prediction.confidence}), 200
        finally:
            if temp_path.exists():
                temp_path.unlink()

    return app


app = create_app()
