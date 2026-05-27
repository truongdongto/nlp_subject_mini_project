import evaluate
from tqdm.auto import tqdm

from medev.inference.decode import translate_sentence


def evaluate_test_set(model, dataset, tokenizer, device, max_samples: int | None = None):
    model.eval()
    all_preds = []

    raw_refs = dataset["vi"]
    raw_sources = dataset["en"]

    if max_samples is not None:
        raw_sources = raw_sources[:max_samples]
        raw_refs = raw_refs[:max_samples]

    print(f"Generating translations for {len(raw_sources)} samples...")

    for src_sent in tqdm(raw_sources):
        pred_sent = translate_sentence(model, tokenizer, src_sent, device=device)
        all_preds.append(pred_sent)

    sacrebleu = evaluate.load("sacrebleu")
    ter = evaluate.load("ter")
    meteor = evaluate.load("meteor")

    formatted_refs = [[ref] for ref in raw_refs]

    bleu_results = sacrebleu.compute(predictions=all_preds, references=formatted_refs)
    ter_results = ter.compute(predictions=all_preds, references=raw_refs)
    meteor_results = meteor.compute(predictions=all_preds, references=raw_refs)

    metrics = {
        "sacrebleu": bleu_results["score"],
        "ter": ter_results["score"],
        "meteor": meteor_results["meteor"],
    }
    return metrics, all_preds, list(raw_refs)
