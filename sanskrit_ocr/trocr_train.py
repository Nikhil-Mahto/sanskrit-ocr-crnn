from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from metrics import character_error_rate, word_error_rate
from sanskrit_ocr.trocr_dataset import TrOCRDataset, load_trocr_records, train_val_split


def _enable_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a TrOCR model on Sanskrit Devanagari OCR data.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="microsoft/trocr-small-printed")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/trocr_sanskrit"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--generation-max-length", type=int, default=128)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser


def _compute_metrics_factory(processor: TrOCRProcessor):
    def compute_metrics(eval_pred) -> dict[str, float]:
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        decoded_predictions = processor.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
        decoded_labels = processor.batch_decode(labels, skip_special_tokens=True)

        cleaned_predictions = [prediction.strip() for prediction in decoded_predictions]
        cleaned_labels = [label.strip() for label in decoded_labels]

        cer = sum(character_error_rate(reference, hypothesis) for reference, hypothesis in zip(cleaned_labels, cleaned_predictions))
        wer = sum(word_error_rate(reference, hypothesis) for reference, hypothesis in zip(cleaned_labels, cleaned_predictions))
        total = max(1, len(cleaned_labels))
        return {"cer": cer / total, "wer": wer / total}

    return compute_metrics


def train_trocr(args: argparse.Namespace) -> Path:
    records = load_trocr_records(args.dataset_dir, args.labels_file)
    train_records, val_records = train_val_split(records, val_ratio=args.val_ratio, seed=args.seed)
    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_eval_samples is not None:
        val_records = val_records[: args.max_eval_samples]

    processor = TrOCRProcessor.from_pretrained(args.model_name)
    model = VisionEncoderDecoderModel.from_pretrained(args.model_name)

    decoder_start_token_id = processor.tokenizer.cls_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = processor.tokenizer.bos_token_id
    eos_token_id = processor.tokenizer.sep_token_id
    if eos_token_id is None:
        eos_token_id = processor.tokenizer.eos_token_id

    model.config.decoder_start_token_id = decoder_start_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = eos_token_id
    model.generation_config.decoder_start_token_id = decoder_start_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id = eos_token_id
    model.generation_config.max_length = args.generation_max_length
    model.generation_config.num_beams = args.num_beams
    model.generation_config.early_stopping = True
    model.generation_config.no_repeat_ngram_size = 0
    model.generation_config.length_penalty = 1.0

    train_dataset = TrOCRDataset(train_records, processor, max_target_length=args.max_target_length)
    eval_dataset = TrOCRDataset(val_records, processor, max_target_length=args.max_target_length)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_kwargs = {
        "output_dir": str(args.output_dir),
        "predict_with_generate": True,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "cer",
        "greater_is_better": False,
        "generation_max_length": args.generation_max_length,
        "generation_num_beams": args.num_beams,
        "report_to": [],
        "seed": args.seed,
        "remove_unused_columns": False,
    }
    training_signature = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in training_signature:
        training_kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in training_signature:
        training_kwargs["eval_strategy"] = "epoch"
    if "use_cpu" in training_signature:
        training_kwargs["use_cpu"] = args.device == "cpu"
    elif "no_cuda" in training_signature:
        training_kwargs["no_cuda"] = args.device == "cpu"

    training_args = Seq2SeqTrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": default_data_collator,
        "compute_metrics": _compute_metrics_factory(processor),
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in trainer_signature:
        trainer_kwargs["tokenizer"] = processor

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    train_result = trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    metrics = trainer.evaluate(max_length=args.generation_max_length, num_beams=args.num_beams)
    metrics["train_loss"] = float(train_result.training_loss)
    metrics["train_samples"] = len(train_dataset)
    metrics["eval_samples"] = len(eval_dataset)

    metrics_path = args.output_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Saved fine-tuned TrOCR model to {args.output_dir}")
    return args.output_dir


def main() -> None:
    _enable_utf8_stdout()
    args = build_argparser().parse_args()
    train_trocr(args)


if __name__ == "__main__":
    main()
