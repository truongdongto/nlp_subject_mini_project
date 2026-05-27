# MedEV English–Vietnamese Medical Translation

LSTM Seq2Seq model for translating English medical text to Vietnamese, trained on the [MedEV](https://huggingface.co/datasets/nhuvo/MedEV) dataset. Refactored from `MedEV_translation.ipynb` into a runnable Python package.

## Requirements

- Python 3.10+
- CUDA GPU recommended for training (CPU works for inference with a trained checkpoint)
- Internet on first run to download validation/test splits from Hugging Face (if not cached)

```bash
cd "Mini Project"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('punkt', quiet=True)"
```

## Project structure

```
Mini Project/
├── medev/                  # Core package
│   ├── config.py           # Paths and hyperparameters
│   ├── data/               # Dataset loading and cleaning
│   ├── tokenizer/          # SentencePiece training/loading
│   ├── dataset/            # PyTorch Dataset and DataLoader helpers
│   ├── model/              # Encoder, Decoder, Seq2Seq
│   ├── training/           # Training loop and early stopping
│   ├── inference/          # Beam search translation
│   └── evaluation/         # SacreBLEU, TER, METEOR
├── scripts/
│   ├── train_tokenizer.py
│   ├── train.py
│   ├── translate.py
│   └── evaluate.py
├── data/                   # Local train.en.txt, train.vi.txt
├── artifacts/              # Tokenizer + checkpoints (generated)
└── best_seq2seq_model.pt   # Optional: pre-trained weights at repo root
```

## Data setup

- **Train**: `data/train.en.txt` and `data/train.vi.txt` are included locally.
- **Validation / test**: On first run, `datasets` downloads `val.*.new.txt` and `test.*.new.txt` from the Hugging Face `nhuvo/MedEV` hub.

## Workflow

Run all commands from the project root (`Mini Project/`).

### 1. Train tokenizer

Trains a shared SentencePiece Unigram model (vocab 40k) on the combined EN+VI training corpora.

```bash
python scripts/train_tokenizer.py
```

Quick smoke test on a subset:

```bash
python scripts/train_tokenizer.py --max-samples 1000
```

Outputs:

- `artifacts/mbart_shared_tokenizer.model`
- `artifacts/mbart_shared_tokenizer.vocab`

### 2. Train model

```bash
python scripts/train.py --checkpoint artifacts/best_seq2seq_model.pt
```

Useful flags:

| Flag | Description |
|------|-------------|
| `--epochs 10` | Number of epochs (default: 10) |
| `--batch-size 32` | Batch size |
| `--device cuda` | Force GPU or CPU |
| `--max-samples 500` | Subset for debugging |

Training uses early stopping (patience 3) and saves the best checkpoint to `artifacts/best_seq2seq_model.pt`.

### 3. Translate a sentence

```bash
python scripts/translate.py --text "Blood pressure is elevated."
```

If the checkpoint is at the repo root instead of `artifacts/`:

```bash
python scripts/translate.py --checkpoint best_seq2seq_model.pt --tokenizer artifacts/mbart_shared_tokenizer.model
```

### 4. Evaluate on test set

```bash
python scripts/evaluate.py
```

Faster check on a subset:

```bash
python scripts/evaluate.py --max-samples 50
```

## Using an existing checkpoint

If you already have `best_seq2seq_model.pt` (e.g. from Colab) but no tokenizer files:

1. Place weights at `best_seq2seq_model.pt` (root) or `artifacts/best_seq2seq_model.pt`.
2. **You must have the matching SentencePiece model** (`.model` file). Either:
   - Copy `mbart_shared_tokenizer.model` from your Colab run into `artifacts/`, or
   - Re-run `python scripts/train_tokenizer.py` with the same settings (vocab 40k, unigram).

Then:

```bash
python scripts/translate.py --checkpoint best_seq2seq_model.pt
```

## Hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| Encoder/decoder embedding | 216 |
| Hidden dim | 255 |
| LSTM layers | 2 |
| Dropout | 0.5 |
| Max sequence length | 64 |
| Batch size | 32 |
| Vocab size | 40000 |

## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| `Tokenizer model not found` | Run `train_tokenizer.py` or pass `--tokenizer path/to/model` |
| `Checkpoint not found` | Pass `--checkpoint` or place file under `artifacts/` or project root |
| Out of memory | Lower `--batch-size` or reduce `max_seq_len` in `medev/config.py` |
| Slow evaluation | Use `--max-samples N` |
| METEOR errors | Run `nltk.download('wordnet')` and `nltk.download('punkt')` |
| CPU training very slow | Use GPU or `--max-samples` for development |

## Reference

Original notebook: `MedEV_translation.ipynb`
