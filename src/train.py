"""Fine-tune a T5 model for email reply generation."""

import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from config import (
    DATA_PROCESSED,
    DEFAULT_MODEL_NAME,
    MODEL_CHECKPOINT_DIR,
    MODEL_TRAINING_DIR,
    TRAIN_FILENAME,
    VAL_FILENAME,
)
from dataset import EmailDataset


def train(
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    final_dir: Path,
    model_name: str = DEFAULT_MODEL_NAME,
    epochs: int = 3,
    batch_size: int = 8,
    max_input: int = 512,
    max_target: int = 128,
    warmup_steps: int = 500,
):
    train_df = pd.read_csv(train_path).dropna()
    val_df = pd.read_csv(val_path).dropna()

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    train_dataset = EmailDataset(train_df, tokenizer, max_input=max_input, max_target=max_target)
    val_dataset = EmailDataset(val_df, tokenizer, max_input=max_input, max_target=max_target)

    model = T5ForConditionalGeneration.from_pretrained(model_name)

    use_fp16 = torch.cuda.is_available()
    effective_warmup = min(warmup_steps, max(1, len(train_dataset) // 2))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=effective_warmup,
        weight_decay=0.01,
        logging_dir=str(output_dir.parent / "logs"),
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        fp16=use_fp16,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
    trainer.train()

    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train email reply generation model.")
    parser.add_argument("--train-data", type=Path, default=DATA_PROCESSED / TRAIN_FILENAME)
    parser.add_argument("--val-data", type=Path, default=DATA_PROCESSED / VAL_FILENAME)
    parser.add_argument("--output", type=Path, default=MODEL_TRAINING_DIR)
    parser.add_argument("--final-output", type=Path, default=MODEL_CHECKPOINT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input", type=int, default=512)
    parser.add_argument("--max-target", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=500)
    args = parser.parse_args()

    if not args.train_data.exists() or not args.val_data.exists():
        raise FileNotFoundError("Train/val data not found. Run split_data.py after preprocessing.")

    train(
        train_path=args.train_data,
        val_path=args.val_data,
        output_dir=args.output,
        final_dir=args.final_output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_input=args.max_input,
        max_target=args.max_target,
        warmup_steps=args.warmup_steps,
    )


if __name__ == "__main__":
    main()
