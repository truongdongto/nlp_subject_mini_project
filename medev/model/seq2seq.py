from pathlib import Path

import torch
import torch.nn as nn
from torch import optim

from medev.config import Config


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        _, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        if input.dim() == 1:
            input = input.unsqueeze(1)

        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output)
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        hidden, cell = self.encoder(src)
        decoder_input = trg[:, :-1]
        outputs, _, _ = self.decoder(decoder_input, hidden, cell)
        return outputs


def model_hyperparams_from_config(config: Config, tokenizer) -> dict:
    vocab = tokenizer.get_piece_size()
    return {
        "input_dim": vocab,
        "output_dim": vocab,
        "enc_emb_dim": config.enc_emb_dim,
        "dec_emb_dim": config.dec_emb_dim,
        "hid_dim": config.hid_dim,
        "n_layers": config.n_layers,
        "enc_dropout": config.enc_dropout,
        "dec_dropout": config.dec_dropout,
    }


def build_model(tokenizer, device: torch.device, config: Config | None = None):
    config = config or Config()
    input_dim = tokenizer.get_piece_size()
    output_dim = tokenizer.get_piece_size()

    enc = Encoder(
        input_dim, config.enc_emb_dim, config.hid_dim, config.n_layers, config.enc_dropout
    )
    dec = Decoder(
        output_dim, config.dec_emb_dim, config.hid_dim, config.n_layers, config.dec_dropout
    )
    model = Seq2Seq(enc, dec, device).to(device)

    optimizer = optim.Adam(model.parameters())
    pad_idx = tokenizer.unk_id()
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    return model, optimizer, criterion


def _build_from_hyperparams(hp: dict, device: torch.device) -> Seq2Seq:
    enc = Encoder(
        hp["input_dim"], hp["enc_emb_dim"], hp["hid_dim"], hp["n_layers"], hp["enc_dropout"]
    )
    dec = Decoder(
        hp["output_dim"], hp["dec_emb_dim"], hp["hid_dim"], hp["n_layers"], hp["dec_dropout"]
    )
    model = Seq2Seq(enc, dec, device).to(device)
    return model


def load_checkpoint(
    checkpoint_path: Path | str,
    device: torch.device,
) -> tuple[Seq2Seq, dict]:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hp = checkpoint["model_hyperparams"]
    model = _build_from_hyperparams(hp, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint
