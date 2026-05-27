#!/usr/bin/env python3
"""Train shared SentencePiece tokenizer on MedEV train corpora."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from medev.config import Config
from medev.data.load import load_medev_dataset
from medev.data.preprocess import apply_cleaning
from medev.tokenizer.sentencepiece import prepare_corpus_file, train_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer on MedEV")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory with train.*.txt")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="Output directory for tokenizer files",
    )
    parser.add_argument("--vocab-size", type=int, default=40000)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit training sentences per language (smoke test)",
    )
    args = parser.parse_args()

    config = Config()
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.artifacts_dir:
        config.artifacts_dir = args.artifacts_dir
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {config.data_dir}...")
    dataset = load_medev_dataset(config.data_dir)
    dataset = apply_cleaning(dataset)

    en_corpus = list(dataset["train"]["en"])
    vi_corpus = list(dataset["train"]["vi"])
    if args.max_samples:
        en_corpus = en_corpus[: args.max_samples]
        vi_corpus = vi_corpus[: args.max_samples]

    corpus_path = prepare_corpus_file(
        en_corpus, vi_corpus, config.combined_corpus_path
    )
    print(f"Combined corpus: {corpus_path} ({len(en_corpus) + len(vi_corpus)} lines)")

    model_path = train_tokenizer(
        corpus_path=corpus_path,
        model_prefix=config.tokenizer_prefix_path,
        vocab_size=args.vocab_size,
    )
    print(f"Tokenizer saved: {model_path}")


if __name__ == "__main__":
    main()
