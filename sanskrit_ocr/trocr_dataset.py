from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from transformers import TrOCRProcessor


@dataclass(frozen=True)
class TrOCRRecord:
    image_path: Path
    text: str


def load_trocr_records(dataset_dir: str | Path, labels_file: str | Path | None = None) -> list[TrOCRRecord]:
    dataset_root = Path(dataset_dir)
    images_dir = dataset_root / "images"
    labels_path = Path(labels_file) if labels_file else dataset_root / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    records: list[TrOCRRecord] = []
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        image_name, text = line.split("\t", maxsplit=1)
        records.append(TrOCRRecord(image_path=images_dir / image_name.strip(), text=text.strip()))
    return records


def train_val_split(records: Sequence[TrOCRRecord], val_ratio: float = 0.1, seed: int = 42) -> tuple[list[TrOCRRecord], list[TrOCRRecord]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")
    if len(records) < 2:
        raise ValueError("At least two records are required to build train/validation splits.")

    import random

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    if not train_records:
        raise ValueError("Validation split consumed the full dataset. Lower val_ratio or add more samples.")
    return train_records, val_records


class TrOCRDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        records: Sequence[TrOCRRecord],
        processor: TrOCRProcessor,
        max_target_length: int = 128,
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        image = Image.open(record.image_path).convert("RGB")

        # TrOCR consumes fixed-size encoder pixel values and tokenized decoder labels.
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text=record.text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}
