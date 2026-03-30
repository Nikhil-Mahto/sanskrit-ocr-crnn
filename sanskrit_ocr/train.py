from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from config import OCRConfig
from sanskrit_ocr.data.charset import SanskritCharset
from sanskrit_ocr.dataset import OCRDataset, ocr_collate_fn
from sanskrit_ocr.models.crnn import CRNN
from sanskrit_ocr.utils.checkpointing import load_checkpoint, save_checkpoint
from sanskrit_ocr.utils.decoding import ctc_greedy_decode
from sanskrit_ocr.utils.training import resolve_device, seed_everything


def _enable_utf8_stdout() -> None:
    # Windows consoles often default to a code page that cannot print Devanagari text.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Sanskrit CRNN OCR model.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--save-name", type=str, default="sanskrit_crnn.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--scheduler-step-size", type=int, default=10)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=64)
    parser.add_argument("--min-width", type=int, default=64)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a saved checkpoint.")
    return parser


def train_model(config: OCRConfig, save_name: str = "sanskrit_crnn.pt", resume_path: Path | None = None) -> Path:
    # Keep training deterministic across runs so comparisons are easier to reason about.
    seed_everything(config.seed)
    device = resolve_device(config.device)

    charset = SanskritCharset()
    dataset = OCRDataset(
        dataset_dir=config.dataset_dir,
        labels_file=config.labels_file,
        charset=charset,
        augment=True,
        image_height=config.image_height,
        min_width=config.min_width,
        max_width=config.max_width,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=ocr_collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = CRNN(
        num_classes=charset.vocab_size,
        hidden_size=config.lstm_hidden_size,
        lstm_layers=config.lstm_layers,
        dropout=config.dropout,
    ).to(device)
    criterion = nn.CTCLoss(blank=charset.blank_index, zero_infinity=True)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)

    history = []
    start_epoch = 1
    if resume_path is not None:
        # Resume both model and optimizer state so the scheduler and gradients continue smoothly.
        payload = load_checkpoint(resume_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        optimizer_state = payload.get("optimizer_state_dict")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        history = payload.get("extra", {}).get("history", [])
        start_epoch = int(payload.get("epoch", 0)) + 1
        if start_epoch > config.epochs:
            print(f"Checkpoint already trained through epoch {start_epoch - 1}. Nothing to do.")
            return resume_path

    checkpoint_path = config.checkpoint_dir / save_name
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader, start=1):
            images = batch["images"].to(device)
            widths = batch["widths"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            # Standard CRNN + CTC training step on variable-width line images.
            optimizer.zero_grad(set_to_none=True)
            log_probs, output_lengths = model(images, widths)
            loss = criterion(log_probs, targets, output_lengths, target_lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item()
            if batch_idx % config.log_interval == 0 or batch_idx == len(dataloader):
                avg_loss = running_loss / batch_idx
                print(
                    f"Epoch {epoch:02d}/{config.epochs:02d} | "
                    f"Step {batch_idx:03d}/{len(dataloader):03d} | Loss {avg_loss:.4f}"
                )

        scheduler.step()
        epoch_loss = running_loss / max(1, len(dataloader))
        history.append({"epoch": epoch, "loss": epoch_loss})

        with torch.no_grad():
            # Greedy sample predictions give a quick qualitative signal while training is running.
            sample_batch = next(iter(dataloader))
            sample_images = sample_batch["images"].to(device)
            sample_widths = sample_batch["widths"].to(device)
            sample_log_probs, _ = model(sample_images, sample_widths)
            sample_predictions = ctc_greedy_decode(sample_log_probs, charset)[:2]
            print(f"Epoch {epoch:02d} summary | Loss {epoch_loss:.4f} | Sample predictions {sample_predictions}")

        save_checkpoint(
            checkpoint_path,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            charset=charset.characters,
            extra={
                "history": history,
                "config": {
                    "image_height": config.image_height,
                    "min_width": config.min_width,
                    "max_width": config.max_width,
                    "hidden_size": config.lstm_hidden_size,
                    "lstm_layers": config.lstm_layers,
                    "dropout": config.dropout,
                },
            },
        )

    history_path = config.output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Training complete. Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    _enable_utf8_stdout()
    args = build_argparser().parse_args()
    config = OCRConfig(
        dataset_dir=args.dataset_dir,
        labels_file=args.labels_file,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        num_workers=args.num_workers,
        image_height=args.image_height,
        min_width=args.min_width,
        max_width=args.max_width,
        lstm_hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
        device=args.device,
        seed=args.seed,
    ).resolve()
    train_model(config, save_name=args.save_name, resume_path=args.resume)


if __name__ == "__main__":
    main()
