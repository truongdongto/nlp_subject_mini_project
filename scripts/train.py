#!/usr/bin/env python3
"""Train LSTM Seq2Seq model on MedEV."""

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from medev.config import Config, get_device
from medev.data.load import load_medev_dataset
from medev.data.preprocess import apply_cleaning
from medev.dataset.translation import build_dataloaders
from medev.model.seq2seq import build_model
from medev.tokenizer.sentencepiece import load_tokenizer
from medev.training.trainer import run_training


def main():
    parser = argparse.ArgumentParser(description="Train MedEV Seq2Seq translator")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=None)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to .model file (default: artifacts/mbart_shared_tokenizer.model)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Checkpoint save path")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit train rows (debug)")
    args = parser.parse_args()

    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.artifacts_dir:
        config.artifacts_dir = args.artifacts_dir
    if args.epochs is not None:
        config.n_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.checkpoint:
        config.checkpoint_name = args.checkpoint.name
        config.artifacts_dir = args.checkpoint.parent

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    tokenizer_path = args.tokenizer or config.tokenizer_model_path
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    print(f"Loading dataset from {config.data_dir}...")
    dataset = load_medev_dataset(config.data_dir)
    dataset = apply_cleaning(dataset)

    if args.max_samples:
        from datasets import DatasetDict

        dataset = DatasetDict(
            {
                split: dataset[split].select(range(min(args.max_samples, len(dataset[split]))))
                for split in dataset.keys()
            }
        )

    print("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        dataset,
        tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        pin_memory=device.type == "cuda",
    )

    model, optimizer, criterion = build_model(tokenizer, device, config)
    print(model)

    run_training(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        tokenizer,
        config,
    )


if __name__ == "__main__":
    main()
