import torch

from medev.data.preprocess import clean_text


def beam_search_decoder(
    model,
    src_tensor,
    tokenizer,
    beam_width=3,
    max_len=64,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)

    beams = [(0, [tokenizer.bos_id()], hidden, cell, False)]

    for _ in range(max_len):
        new_beams = []
        for score, seq, h, c, finished in beams:
            if finished:
                new_beams.append((score, seq, h, c, True))
                continue

            trg_tensor = torch.tensor([seq[-1]], device=device)
            with torch.no_grad():
                output, next_h, next_c = model.decoder(trg_tensor, h, c)

            log_probs = torch.log_softmax(output.squeeze(1), dim=-1)
            topk_log_probs, topk_ids = log_probs.topk(beam_width)

            for i in range(beam_width):
                next_id = topk_ids[0][i].item()
                next_score = score + topk_log_probs[0][i].item()
                is_finished = next_id == tokenizer.eos_id()
                new_beams.append((next_score, seq + [next_id], next_h, next_c, is_finished))

        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        if all(b[4] for b in beams):
            break

    return beams[0][1]


def translate_sentence(
    model,
    tokenizer,
    src_sentence,
    max_len=64,
    beam_width=3,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    cleaned_src = clean_text(src_sentence, lang="en")
    src_tokens = tokenizer.encode_as_ids(cleaned_src)
    src_tensor = (
        torch.tensor([tokenizer.bos_id()] + src_tokens + [tokenizer.eos_id()])
        .unsqueeze(0)
        .to(device)
    )

    trg_indices = beam_search_decoder(
        model, src_tensor, tokenizer, beam_width, max_len, device
    )

    special = {tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.unk_id()}
    output_indices = [idx for idx in trg_indices if idx not in special]
    return tokenizer.decode_ids(output_indices)
