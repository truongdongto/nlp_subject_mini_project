from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset

HF_DATASET = "nhuvo/MedEV"

SPLIT_FILES = {
    "train": ("train.en.txt", "train.vi.txt"),
    "validation": ("val.en.new.txt", "val.vi.new.txt"),
    "test": ("test.en.new.txt", "test.vi.new.txt"),
}


def _resolve_data_files(data_dir: Path, lang: str) -> dict[str, str]:
    """Build data_files dict for load_dataset; use local paths when present."""
    data_files: dict[str, str] = {}
    for split, (en_name, vi_name) in SPLIT_FILES.items():
        fname = en_name if lang == "en" else vi_name
        local = data_dir / fname
        data_files[split] = str(local) if local.exists() else fname
    return data_files


def load_medev_dataset(data_dir: Path | str | None = None) -> DatasetDict:
    data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data"

    data_files_en = _resolve_data_files(data_dir, "en")
    data_files_vi = _resolve_data_files(data_dir, "vi")

    en_data = load_dataset(HF_DATASET, data_files=data_files_en)
    vi_data = load_dataset(HF_DATASET, data_files=data_files_vi)

    en_data = en_data.rename_column("text", "en")
    vi_data = vi_data.rename_column("text", "vi")

    return DatasetDict(
        {
            "train": concatenate_datasets([en_data["train"], vi_data["train"]], axis=1),
            "validation": concatenate_datasets(
                [en_data["validation"], vi_data["validation"]], axis=1
            ),
            "test": concatenate_datasets([en_data["test"], vi_data["test"]], axis=1),
        }
    )
