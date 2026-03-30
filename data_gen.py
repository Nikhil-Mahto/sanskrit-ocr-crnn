from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


SANSKRIT_WORDS: Sequence[str] = (
    "अथ",
    "धर्मः",
    "धर्मक्षेत्रे",
    "कुरुक्षेत्रे",
    "योगः",
    "कर्मसु",
    "कौशलम्",
    "संस्कृतम्",
    "शान्तिः",
    "ज्ञानम्",
    "प्रज्ञा",
    "सत्यं",
    "शिवः",
    "नमः",
    "आत्मा",
    "ब्रह्म",
    "अनन्तः",
    "श्लोकः",
    "वेदः",
    "गीता",
    "रामः",
    "सीता",
    "गुरुः",
    "विद्या",
    "भवति",
    "पठति",
    "वदति",
    "मनः",
    "भक्ति",
    "मोक्षः",
    "सुखम्",
    "दुःखम्",
    "महर्षिः",
    "यज्ञः",
    "अग्निः",
    "सूर्यः",
    "चन्द्रः",
    "लोकः",
    "भारतम्",
    "देवनागरी",
)

SANSKRIT_LINES: Sequence[str] = (
    "धर्मक्षेत्रे कुरुक्षेत्रे",
    "योगः कर्मसु कौशलम्",
    "सत्यं वद धर्मं चर",
    "अहिंसा परमो धर्मः",
    "विद्या ददाति विनयम्",
    "शान्तिः शान्तिः शान्तिः",
    "कर्मण्येवाधिकारस्ते",
    "वसुधैव कुटुम्बकम्",
    "आत्मानं विद्धि",
    "नमो भगवते वासुदेवाय",
    "सा विद्या या विमुक्तये",
    "कर्मण्येवाधिकारस्ते मा फलेषु कदाचन",
    "उत्तिष्ठत जाग्रत प्राप्य वरान्निबोधत",
    "श्रद्धावान् लभते ज्ञानम्",
    "न हि ज्ञानेन सदृशं पवित्रमिह विद्यते",
    "विद्या विवादाय धनं मदाय",
    "धर्मो रक्षति रक्षितः",
    "सत्यमेव जयते",
    "नमो नमः शिवाय",
    "रामो विग्रहवान् धर्मः",
    "जननी जन्मभूमिश्च स्वर्गादपि गरीयसी",
    "श्रीगुरुभ्यो नमः",
    "त्वमेव माता च पिता त्वमेव",
    "गुरुर्ब्रह्मा गुरुर्विष्णुः",
    "असतो मा सद्गमय",
    "तमसो मा ज्योतिर्गमय",
    "मृत्योर्मा अमृतं गमय",
    "सर्वे भवन्तु सुखिनः",
    "सर्वे सन्तु निरामयाः",
    "लोकाः समस्ताः सुखिनो भवन्तु",
    "ॐ शान्तिः शान्तिः शान्तिः",
    "विद्यां च अविद्यां च यस्तद्वेद उभयं सह",
    "आ नो भद्राः क्रतवो यन्तु विश्वतः",
    "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत",
    "श्रेयान् स्वधर्मो विगुणः",
    "नायमात्मा बलहीनेन लभ्यः",
    "भवतु सर्वमङ्गलम्",
    "वेदोऽखिलो धर्ममूलम्",
    "माता भूमि: पुत्रोऽहं पृथिव्याः",
    "वागर्थाविव सम्पृक्तौ",
    "गङ्गे च यमुने चैव",
    "कराग्रे वसते लक्ष्मीः",
    "शुभं करोति कल्याणम्",
    "वक्रतुण्ड महाकाय",
    "अग्निमीळे पुरोहितम्",
    "ईशावास्यमिदं सर्वम्",
    "शं नो मित्रः शं वरुणः",
    "आत्मा वा अरे द्रष्टव्यः",
    "तत्त्वमसि",
    "अयमात्मा ब्रह्म",
)

DEFAULT_FONT_CANDIDATES: Sequence[Path] = (
    Path("NotoSansDevanagari-Regular.ttf"),
    Path("fonts/NotoSansDevanagari-Regular.ttf"),
    Path(r"C:\Windows\Fonts\Nirmala.ttf"),
    Path(r"C:\Windows\Fonts\mangal.ttf"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic Sanskrit OCR dataset.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--font-path", type=Path, default=None, help="Path to NotoSansDevanagari font.")
    parser.add_argument("--count", type=int, default=5000, help="Number of samples to generate.")
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--min-font-size", type=int, default=28)
    parser.add_argument("--max-font-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_font_path(requested_path: Path | None) -> Path:
    candidates = [requested_path] if requested_path else []
    candidates.extend(DEFAULT_FONT_CANDIDATES)
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a Devanagari font. Pass --font-path pointing to NotoSansDevanagari-Regular.ttf."
    )


def build_text_corpus() -> list[str]:
    # Mix fixed Sanskrit lines with synthetic word combinations to create broader OCR coverage.
    corpus = list(SANSKRIT_LINES)
    for size in range(2, 8):
        for _ in range(220):
            corpus.append(" ".join(random.choices(SANSKRIT_WORDS, k=size)))
    for _ in range(320):
        left = random.choice(SANSKRIT_LINES)
        right = " ".join(random.choices(SANSKRIT_WORDS, k=random.randint(2, 5)))
        corpus.append(f"{left} {right}")
    return corpus


def add_noise(draw: ImageDraw.ImageDraw, width: int, height: int, density: int) -> None:
    for _ in range(density):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        shade = random.randint(120, 220)
        draw.point((x, y), fill=shade)


def render_text_image(text: str, font_path: Path, image_height: int, font_size: int) -> Image.Image:
    font = ImageFont.truetype(str(font_path), font_size)
    temp = Image.new("L", (10, 10), color=255)
    temp_draw = ImageDraw.Draw(temp)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_width = max(1, bbox[2] - bbox[0])
    text_height = max(1, bbox[3] - bbox[1])

    padding_x = random.randint(18, 36)
    padding_y = max(6, (image_height - text_height) // 2 + random.randint(-2, 2))
    width = text_width + padding_x * 2
    canvas = Image.new("L", (width, image_height), color=255)
    draw = ImageDraw.Draw(canvas)

    offset_x = padding_x + random.randint(-4, 4)
    offset_y = max(0, padding_y - bbox[1])
    draw.text((offset_x, offset_y), text, font=font, fill=random.randint(0, 40))
    add_noise(draw, width, image_height, density=max(10, width // 18))

    # Layer multiple lightweight perturbations so the synthetic set better approximates scanned documents.
    if random.random() < 0.75:
        canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.75, 1.25))
    if random.random() < 0.55:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.4)))
    if random.random() < 0.45:
        angle = random.uniform(-5.0, 5.0)
        canvas = canvas.rotate(angle, expand=1, fillcolor=255)
        canvas = canvas.resize((max(width, canvas.width), image_height))
    if random.random() < 0.40:
        canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.8, 1.3))
    return canvas


def generate_samples(
    output_dir: Path,
    labels_path: Path,
    font_path: Path,
    count: int,
    image_height: int,
    min_font_size: int,
    max_font_size: int,
) -> None:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    corpus = build_text_corpus()

    with labels_path.open("w", encoding="utf-8") as labels_file:
        for index in range(count):
            text = random.choice(corpus)
            font_size = random.randint(min_font_size, max_font_size)
            image = render_text_image(text, font_path, image_height=image_height, font_size=font_size)
            image_name = f"synthetic_{index:05d}.png"
            image.save(images_dir / image_name)
            labels_file.write(f"{image_name}\t{text}\n")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    dataset_dir = args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)
    font_path = resolve_font_path(args.font_path)
    labels_path = dataset_dir / "labels.txt"
    generate_samples(
        output_dir=dataset_dir,
        labels_path=labels_path,
        font_path=font_path,
        count=args.count,
        image_height=args.image_height,
        min_font_size=args.min_font_size,
        max_font_size=args.max_font_size,
    )
    print(f"Generated {args.count} Sanskrit samples in {dataset_dir}")
    print(f"Font used: {font_path}")


if __name__ == "__main__":
    main()
