from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    artifacts_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "artifacts")
    tokenizer_prefix: str = "mbart_shared_tokenizer"
    checkpoint_name: str = "best_seq2seq_model.pt"

    enc_emb_dim: int = 216
    dec_emb_dim: int = 216
    hid_dim: int = 255
    n_layers: int = 2
    enc_dropout: float = 0.5
    dec_dropout: float = 0.5

    batch_size: int = 32
    max_seq_len: int = 64
    n_epochs: int = 10
    clip: float = 1.0
    seed: int = 42
    vocab_size: int = 40000
    early_stopping_patience: int = 3

    @property
    def tokenizer_model_path(self) -> Path:
        return self.artifacts_dir / f"{self.tokenizer_prefix}.model"

    @property
    def tokenizer_prefix_path(self) -> Path:
        return self.artifacts_dir / self.tokenizer_prefix

    @property
    def checkpoint_path(self) -> Path:
        artifacts_ckpt = self.artifacts_dir / self.checkpoint_name
        if artifacts_ckpt.exists():
            return artifacts_ckpt
        root_ckpt = PROJECT_ROOT / self.checkpoint_name
        if root_ckpt.exists():
            return root_ckpt
        return artifacts_ckpt

    @property
    def combined_corpus_path(self) -> Path:
        return self.artifacts_dir / "combined_corpus.txt"


def get_device(device: str | None = None):
    import torch

    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
