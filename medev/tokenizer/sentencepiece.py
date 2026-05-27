from pathlib import Path

import sentencepiece as spm


def prepare_corpus_file(
    en_corpus: list[str],
    vi_corpus: list[str],
    out_path: Path | str,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined = list(en_corpus) + list(vi_corpus)
    with open(out_path, "w", encoding="utf-8") as f:
        for sentence in combined:
            f.write(sentence + "\n")
    return out_path


def train_tokenizer(
    corpus_path: Path | str,
    model_prefix: Path | str,
    vocab_size: int = 40000,
    num_threads: int = 8,
    character_coverage: float = 0.9995,
) -> Path:
    model_prefix = Path(model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="unigram",
        num_threads=num_threads,
        character_coverage=character_coverage,
    )
    return Path(f"{model_prefix}.model")


def load_tokenizer(model_path: Path | str) -> spm.SentencePieceProcessor:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer model not found: {model_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(model_path))
    return tokenizer
