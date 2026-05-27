import random
from time import perf_counter

import torch

from medev.config import Config
from medev.model.seq2seq import model_hyperparams_from_config
from medev.training.early_stopping import EarlyStopping
from medev.training.loop import eval_epoch, train_epoch


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def run_training(
    model,
    train_dataloader,
    validation_dataloader,
    optimizer,
    criterion,
    device,
    tokenizer,
    config: Config | None = None,
):
    config = config or Config()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)
    checkpoint_path = str(config.artifacts_dir / config.checkpoint_name)
    early_stopping = EarlyStopping(patience=config.early_stopping_patience, path=checkpoint_path)

    print(f"Start training on device: {device}")
    start_training = perf_counter()

    for epoch in range(config.n_epochs):
        start_time = perf_counter()

        train_loss = train_epoch(
            model, train_dataloader, optimizer, criterion, config.clip, device
        )
        valid_loss = eval_epoch(model, validation_dataloader, criterion, device)

        end_time = perf_counter()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(
            f"Epoch: {epoch + 1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s | "
            f"Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}"
        )

        model_hp = model_hyperparams_from_config(config, tokenizer)
        early_stopping(
            valid_loss, model, optimizer, epoch + 1, train_loss, model_hp
        )

        if early_stopping.early_stop:
            print("Early stopping triggered. Training finished.")
            break

    end_training = perf_counter()
    training_mins, training_seconds = divmod(end_training - start_training, 60)
    print(f"--> Total time: {int(training_mins)}m {int(training_seconds)}s")

    return model
