#!/usr/bin/env python3
"""Evaluate Seq2Seq model on MedEV test split (SacreBLEU, TER, METEOR)."""

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
from medev.evaluation.metrics import evaluate_test_set
from medev.model.seq2seq import load_checkpoint
from medev.tokenizer.sentencepiece import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate MedEV Seq2Seq on test set")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"])
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of test sentences (faster smoke test)",
    )
    args = parser.parse_args()

    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    device = get_device(args.device)

    checkpoint_path = args.checkpoint or config.checkpoint_path
    tokenizer_path = args.tokenizer or config.tokenizer_model_path

    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    print(
        f"Loaded epoch {checkpoint['epoch']}, "
        f"valid_loss={checkpoint['valid_loss']:.4f}"
    )

    print(f"Loading dataset from {config.data_dir}...")
    dataset = load_medev_dataset(config.data_dir)
    dataset = apply_cleaning(dataset)
    split_data = dataset[args.split]

    metrics, predictions, references = evaluate_test_set(
        model,
        split_data,
        tokenizer,
        device,
        max_samples=args.max_samples,
    )

    print("\n--- Test Set Metrics ---")
    print(f"SacreBLEU: {metrics['sacrebleu']:.2f}")
    print(f"TER:       {metrics['ter']:.2f}")
    print(f"METEOR:    {metrics['meteor']:.4f}")

    n_show = min(5, len(predictions))
    if n_show:
        print("\n--- Sample Translations ---")
        sources = split_data["en"][:n_show]
        for i in range(n_show):
            print(f"Source: {sources[i]}")
            print(f"Ref:    {references[i]}")
            print(f"Pred:   {predictions[i]}\n")


if __name__ == "__main__":
    main()
