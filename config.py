from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class OCRConfig:
    dataset_dir: Path = Path("dataset")
    labels_file: Optional[Path] = None
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs")
    dictionary_path: Optional[Path] = Path("sanskrit_ocr/data/basic_dictionary.txt")

    image_height: int = 64
    min_width: int = 64
    max_width: int = 1024
    batch_size: int = 16
    num_workers: int = 0

    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5

    lstm_hidden_size: int = 256
    lstm_layers: int = 2
    dropout: float = 0.1

    beam_width: int = 5
    device: str = "cuda"
    seed: int = 42

    save_every: int = 1
    log_interval: int = 10

    def resolve(self) -> "OCRConfig":
        if self.labels_file is None:
            self.labels_file = self.dataset_dir / "labels.txt"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self


DEFAULT_CONFIG = OCRConfig().resolve()
