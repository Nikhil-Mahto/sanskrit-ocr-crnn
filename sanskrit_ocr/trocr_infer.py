from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from sanskrit_ocr.data.postprocessing import SanskritPostProcessor
from sanskrit_ocr.utils.training import resolve_device


def _enable_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class TrOCRPrediction:
    text: str
    confidence: float


class TrOCRInferenceEngine:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        dictionary_path: str | Path | None = Path("sanskrit_ocr/data/basic_dictionary.txt"),
    ) -> None:
        self.device = resolve_device(device)
        self.processor = TrOCRProcessor.from_pretrained(str(model_path))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(model_path)).to(self.device)
        self.model.eval()
        self.postprocessor = SanskritPostProcessor(Path(dictionary_path) if dictionary_path else None)

    def predict(
        self,
        image_path: str | Path,
        num_beams: int = 4,
        max_length: int = 128,
    ) -> TrOCRPrediction:
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                pixel_values,
                num_beams=num_beams,
                max_length=max_length,
                return_dict_in_generate=True,
                output_scores=True,
            )

        decoded = self.processor.batch_decode(generated.sequences, skip_special_tokens=True)[0].strip()
        text = self.postprocessor.clean(decoded)

        confidence = 0.0
        if generated.scores:
            token_confidences = []
            for step_scores in generated.scores:
                probabilities = torch.softmax(step_scores, dim=-1)
                token_confidences.append(probabilities.max(dim=-1).values.mean().item())
            confidence = float(sum(token_confidences) / len(token_confidences))

        return TrOCRPrediction(text=text, confidence=confidence)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OCR inference with a fine-tuned TrOCR Sanskrit model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--dictionary-path", type=Path, default=Path("sanskrit_ocr/data/basic_dictionary.txt"))
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    _enable_utf8_stdout()
    args = build_argparser().parse_args()
    engine = TrOCRInferenceEngine(
        model_path=args.model_path,
        device=args.device,
        dictionary_path=args.dictionary_path,
    )
    prediction = engine.predict(args.image, num_beams=args.num_beams, max_length=args.max_length)
    payload = {"text": prediction.text, "confidence": prediction.confidence}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Recognized text: {prediction.text}")
        print(f"Confidence: {prediction.confidence:.4f}")


if __name__ == "__main__":
    main()
