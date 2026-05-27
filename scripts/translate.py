#!/usr/bin/env python3
"""Translate English sentences to Vietnamese using a trained checkpoint."""

import argparse
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

from medev.config import Config, get_device
from medev.inference.decode import translate_sentence
from medev.model.seq2seq import load_checkpoint
from medev.tokenizer.sentencepiece import load_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Translate EN → VI (MedEV Seq2Seq)")
    parser.add_argument(
        "--text",
        type=str,
        default="Blood pressure is elevated.",
        help="English source sentence",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()

    config = Config()
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

    translation = translate_sentence(
        model,
        tokenizer,
        args.text,
        beam_width=args.beam_width,
        device=device,
    )

    print(f"\nSource (EN): {args.text}")
    print(f"Translation (VI): {translation}")


if __name__ == "__main__":
    main()
