from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from config import OCRConfig
from sanskrit_ocr.data.charset import SanskritCharset
from sanskrit_ocr.data.postprocessing import SanskritPostProcessor
from sanskrit_ocr.data.preprocessing import OCRPreprocessor
from sanskrit_ocr.models.crnn import CRNN
from sanskrit_ocr.utils.checkpointing import load_checkpoint
from sanskrit_ocr.utils.decoding import ctc_beam_search_decode, ctc_greedy_decode
from sanskrit_ocr.utils.image_io import load_image
from sanskrit_ocr.utils.training import resolve_device


DecoderType = Literal["greedy", "beam"]


def _enable_utf8_stdout() -> None:
    # Ensure recognized Sanskrit text prints correctly on Windows terminals.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


@dataclass
class OCRPrediction:
    text: str
    confidence: float


class OCRInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cuda",
        dictionary_path: str | Path | None = Path("sanskrit_ocr/data/basic_dictionary.txt"),
        beam_width: int = 5,
    ) -> None:
        self.device = resolve_device(device)
        payload = load_checkpoint(Path(checkpoint_path), map_location=self.device)
        charset_chars = payload.get("charset")
        self.charset = SanskritCharset(characters=charset_chars or [])

        # Reuse preprocessing and model dimensions saved with the checkpoint so inference matches training.
        config_data = payload.get("extra", {}).get("config", {})
        default_config = OCRConfig()
        self.preprocessor = OCRPreprocessor(
            target_height=config_data.get("image_height", default_config.image_height),
            min_width=config_data.get("min_width", default_config.min_width),
            max_width=config_data.get("max_width", default_config.max_width),
        )
        self.postprocessor = SanskritPostProcessor(Path(dictionary_path) if dictionary_path else None)
        self.beam_width = beam_width
        self.model = CRNN(
            num_classes=self.charset.vocab_size,
            hidden_size=config_data.get("hidden_size", default_config.lstm_hidden_size),
            lstm_layers=config_data.get("lstm_layers", default_config.lstm_layers),
            dropout=config_data.get("dropout", default_config.dropout),
        ).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def predict(self, image_path: str | Path, decoder: DecoderType = "greedy") -> OCRPrediction:
        # Inference follows the same OCR flow as production: load image -> preprocess -> decode -> clean text.
        image = load_image(image_path)
        processed, width = self.preprocessor(image)
        tensor = torch.from_numpy(processed).unsqueeze(0).unsqueeze(0).float().to(self.device)
        widths = torch.tensor([width], dtype=torch.long, device=self.device)

        with torch.no_grad():
            log_probs, _ = self.model(tensor, widths)
            # Greedy decoding is faster; beam search is available when accuracy matters more than speed.
            if decoder == "beam":
                decoded = ctc_beam_search_decode(log_probs, self.charset, beam_width=self.beam_width)[0]
            else:
                decoded = ctc_greedy_decode(log_probs, self.charset)[0]

        # Final cleanup removes repeated spaces and optionally applies lightweight dictionary correction.
        text = self.postprocessor.clean(decoded[0])
        return OCRPrediction(text=text, confidence=decoded[1])


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with Sanskrit OCR model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--decoder", type=str, choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dictionary-path", type=Path, default=Path("sanskrit_ocr/data/basic_dictionary.txt"))
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    _enable_utf8_stdout()
    args = build_argparser().parse_args()
    engine = OCRInferenceEngine(
        checkpoint_path=args.checkpoint,
        device=args.device,
        dictionary_path=args.dictionary_path,
        beam_width=args.beam_width,
    )
    prediction = engine.predict(args.image, decoder=args.decoder)
    payload = {"text": prediction.text, "confidence": prediction.confidence}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"Recognized text: {prediction.text}")
        print(f"Confidence: {prediction.confidence:.4f}")


if __name__ == "__main__":
    main()
