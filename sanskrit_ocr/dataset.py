from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import Tensor
from torch.utils.data import Dataset

from sanskrit_ocr.data.charset import SanskritCharset
from sanskrit_ocr.data.preprocessing import OCRAugmenter, OCRPreprocessor
from sanskrit_ocr.utils.image_io import load_image


@dataclass
class OCRSample:
    image: Tensor
    image_width: int
    encoded_text: Tensor
    text_length: int
    text: str
    image_name: str


class OCRDataset(Dataset[OCRSample]):
    def __init__(
        self,
        dataset_dir: str | Path,
        charset: SanskritCharset,
        labels_file: str | Path | None = None,
        augment: bool = False,
        image_height: int = 64,
        min_width: int = 64,
        max_width: int = 1024,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.images_dir = self.dataset_dir / "images"
        self.labels_file = Path(labels_file) if labels_file else self.dataset_dir / "labels.txt"
        self.charset = charset
        self.preprocessor = OCRPreprocessor(
            target_height=image_height,
            min_width=min_width,
            max_width=max_width,
        )
        self.augmenter = OCRAugmenter() if augment else None
        self.samples = self._load_labels()

    def _load_labels(self) -> List[Dict[str, str]]:
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")

        records: List[Dict[str, str]] = []
        for line in self.labels_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_name, text = line.split("\t", maxsplit=1)
            records.append({"image_name": image_name.strip(), "text": text.strip()})
        return records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> OCRSample:
        sample = self.samples[index]
        image = load_image(self.images_dir / sample["image_name"])
        if self.augmenter is not None:
            image = self.augmenter(image)
        processed, width = self.preprocessor(image)
        encoded = torch.tensor(self.charset.encode(sample["text"]), dtype=torch.long)
        return OCRSample(
            image=torch.from_numpy(processed).unsqueeze(0).float(),
            image_width=width,
            encoded_text=encoded,
            text_length=len(encoded),
            text=sample["text"],
            image_name=sample["image_name"],
        )


def ocr_collate_fn(batch: List[OCRSample]) -> Dict[str, Tensor | List[str]]:
    batch_size = len(batch)
    max_width = max(item.image.shape[-1] for item in batch)
    image_height = batch[0].image.shape[-2]
    images = torch.full((batch_size, 1, image_height, max_width), fill_value=-1.0, dtype=torch.float32)
    widths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    texts: List[str] = []
    image_names: List[str] = []
    targets: List[Tensor] = []

    for idx, item in enumerate(batch):
        current_width = item.image.shape[-1]
        images[idx, :, :, :current_width] = item.image
        widths[idx] = item.image_width
        target_lengths[idx] = item.text_length
        texts.append(item.text)
        image_names.append(item.image_name)
        targets.append(item.encoded_text)

    return {
        "images": images,
        "widths": widths,
        "targets": torch.cat(targets),
        "target_lengths": target_lengths,
        "texts": texts,
        "image_names": image_names,
    }
