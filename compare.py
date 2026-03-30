from __future__ import annotations

import argparse
import sys
from pathlib import Path

from metrics import character_error_rate, word_error_rate
from sanskrit_ocr.infer import OCRInferenceEngine
from tesseract_baseline import run_tesseract


def load_ground_truth(labels_path: Path, image_name: str) -> str | None:
    # Labels are stored in the same tab-separated format used by the training dataset.
    if not labels_path.exists():
        return None
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name, text = line.split("\t", maxsplit=1)
        if name.strip() == image_name:
            return text.strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Tesseract baseline and CRNN OCR on one image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels-file", type=Path, default=Path("dataset/labels.txt"))
    parser.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--tesseract-lang", type=str, default="hin")
    parser.add_argument("--tesseract-psm", type=int, default=6)
    parser.add_argument("--tesseract-cmd", type=str, default=None)
    return parser.parse_args()


def format_metric(name: str, value: float | None) -> str:
    if value is None:
        return f"{name}: N/A"
    return f"{name}: {value:.4f}"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    args = parse_args()
    ground_truth = load_ground_truth(args.labels_file, args.image.name)
    tesseract_text = None
    tesseract_error = None
    try:
        # Tesseract is optional at runtime, so comparison degrades gracefully if the executable is unavailable.
        tesseract_text = run_tesseract(
            args.image,
            lang=args.tesseract_lang,
            psm=args.tesseract_psm,
            tesseract_cmd=args.tesseract_cmd,
        )
    except Exception as exc:
        tesseract_error = str(exc)

    engine = OCRInferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        beam_width=args.beam_width,
    )
    # Reuse the existing CRNN inference entry point instead of duplicating OCR logic here.
    crnn_prediction = engine.predict(args.image, decoder=args.decoder)

    tess_cer = tess_wer = crnn_cer = crnn_wer = None
    if ground_truth is not None:
        # CER/WER are only meaningful when a ground-truth line is available for the input image.
        if tesseract_text is not None:
            tess_cer = character_error_rate(ground_truth, tesseract_text)
            tess_wer = word_error_rate(ground_truth, tesseract_text)
        crnn_cer = character_error_rate(ground_truth, crnn_prediction.text)
        crnn_wer = word_error_rate(ground_truth, crnn_prediction.text)

    print(f"Image: {args.image}")
    print(f"Ground truth: {ground_truth if ground_truth is not None else 'Not found in labels.txt'}")
    if tesseract_text is not None:
        print(f"Tesseract: {tesseract_text}")
    else:
        print(f"Tesseract: unavailable ({tesseract_error})")
    print(f"CRNN: {crnn_prediction.text}")
    print(f"CRNN confidence: {crnn_prediction.confidence:.4f}")
    print(format_metric("Tesseract CER", tess_cer))
    print(format_metric("Tesseract WER", tess_wer))
    print(format_metric("CRNN CER", crnn_cer))
    print(format_metric("CRNN WER", crnn_wer))


if __name__ == "__main__":
    main()
