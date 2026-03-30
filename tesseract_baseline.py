from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image

try:
    import pytesseract
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("pytesseract is required. Install it with `pip install pytesseract`.") from exc


def enable_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def configure_tesseract(tesseract_cmd: str | None = None) -> None:
    resolved = tesseract_cmd or os.environ.get("TESSERACT_CMD")
    if resolved:
        pytesseract.pytesseract.tesseract_cmd = resolved


def run_tesseract(
    image_path: str | Path,
    lang: str = "hin",
    psm: int = 6,
    tesseract_cmd: str | None = None,
) -> str:
    configure_tesseract(tesseract_cmd)
    image = Image.open(image_path)
    config = f"--oem 3 --psm {psm}"
    return pytesseract.image_to_string(image, lang=lang, config=config).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Tesseract OCR baseline on a Sanskrit image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--lang", type=str, default="hin")
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--tesseract-cmd", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    enable_utf8_stdout()
    args = parse_args()
    try:
        text = run_tesseract(args.image, lang=args.lang, psm=args.psm, tesseract_cmd=args.tesseract_cmd)
    except pytesseract.TesseractNotFoundError as exc:
        raise SystemExit(
            "Tesseract executable not found. Install Tesseract or pass --tesseract-cmd "
            "or set TESSERACT_CMD to the full path of tesseract.exe."
        ) from exc
    print(text)


if __name__ == "__main__":
    main()
