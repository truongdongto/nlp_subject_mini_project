import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_


def train_epoch(model, dataloader, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0.0
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    for batch in dataloader:
        src = batch["source_ids"].to(device)
        trg = batch["target_ids"].to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            output = model(src, trg)
            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg_flat)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)


def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0.0
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            src = batch["source_ids"].to(device)
            trg = batch["target_ids"].to(device)

            with autocast(enabled=use_amp):
                output = model(src, trg)
                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg_flat)

            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)
