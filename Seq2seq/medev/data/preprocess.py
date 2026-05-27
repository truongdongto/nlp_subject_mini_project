import re
import unicodedata

from datasets import DatasetDict


def clean_text(text: str, lang: str = "en") -> str:
    if not text:
        return ""
    s = unicodedata.normalize("NFC", text.strip())
    s = re.sub(r"[\u200b-\u200d\ufeff]", "", s)
    s = re.sub(r"[^\w\s.,?!'\"\-:;()]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def apply_cleaning(dataset: DatasetDict) -> DatasetDict:
    dataset = dataset.map(lambda x: {"en": clean_text(x["en"], lang="en")})
    dataset = dataset.map(lambda x: {"vi": clean_text(x["vi"], lang="vi")})
    return dataset
