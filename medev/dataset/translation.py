import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader, Dataset


class TranslationDataset(Dataset):
    def __init__(self, source_data: list[list[int]], target_data: list[list[int]]):
        self.source_data = source_data
        self.target_data = target_data
        assert len(self.source_data) == len(self.target_data)

    def __len__(self) -> int:
        return len(self.source_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "source_ids": torch.tensor(self.source_data[idx], dtype=torch.long),
            "target_ids": torch.tensor(self.target_data[idx], dtype=torch.long),
        }


def encode_corpus(sentences: list[str], tokenizer) -> list[list[int]]:
    return [tokenizer.encode_as_ids(sent) for sent in sentences]


def make_collate_fn(tokenizer, max_seq_len: int = 64):
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    pad_id = tokenizer.unk_id()

    def collate_fn(batch):
        source_ids = [
            torch.tensor(item["source_ids"][:max_seq_len], dtype=torch.long) for item in batch
        ]
        target_ids = [
            torch.tensor(item["target_ids"][:max_seq_len], dtype=torch.long) for item in batch
        ]

        padded_source_ids = []
        padded_target_ids = []
        source_lengths = []
        target_lengths = []

        max_source_len = max(len(s) for s in source_ids)
        max_target_len = max(len(t) for t in target_ids)

        for s_ids, t_ids in zip(source_ids, target_ids):
            s_with_special = torch.cat(
                [
                    torch.tensor([bos_id], dtype=torch.long),
                    s_ids,
                    torch.tensor([eos_id], dtype=torch.long),
                ]
            )
            padded_s = torch.cat(
                [
                    s_with_special,
                    torch.full(
                        (max_source_len + 2 - len(s_with_special),), pad_id, dtype=torch.long
                    ),
                ]
            )
            padded_source_ids.append(padded_s)
            source_lengths.append(len(s_ids) + 2)

            t_with_special = torch.cat(
                [
                    torch.tensor([bos_id], dtype=torch.long),
                    t_ids,
                    torch.tensor([eos_id], dtype=torch.long),
                ]
            )
            padded_t = torch.cat(
                [
                    t_with_special,
                    torch.full(
                        (max_target_len + 2 - len(t_with_special),), pad_id, dtype=torch.long
                    ),
                ]
            )
            padded_target_ids.append(padded_t)
            target_lengths.append(len(t_ids) + 2)

        return {
            "source_ids": torch.stack(padded_source_ids),
            "source_lengths": torch.tensor(source_lengths, dtype=torch.long),
            "target_ids": torch.stack(padded_target_ids),
            "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        }

    return collate_fn


def build_dataloaders(
    med_dataset: DatasetDict,
    tokenizer,
    batch_size: int = 32,
    max_seq_len: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    collate = make_collate_fn(tokenizer, max_seq_len=max_seq_len)

    encoded_en = encode_corpus(med_dataset["train"]["en"], tokenizer)
    encoded_vi = encode_corpus(med_dataset["train"]["vi"], tokenizer)
    train_dataset = TranslationDataset(encoded_en, encoded_vi)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    encoded_en_val = encode_corpus(med_dataset["validation"]["en"], tokenizer)
    encoded_vi_val = encode_corpus(med_dataset["validation"]["vi"], tokenizer)
    validation_dataset = TranslationDataset(encoded_en_val, encoded_vi_val)

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, validation_dataloader
