import torch


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0, path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, model, optimizer, epoch, train_loss, hp):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, epoch, train_loss, val_loss, hp)
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model, optimizer, epoch, train_loss, val_loss, hp):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "model_hyperparams": hp,
            },
            self.path,
        )
        print(
            f"Validation loss decreased ({self.best_loss:.3f}). "
            f"Saving best model to {self.path}\n"
        )
