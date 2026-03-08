import json
import argparse
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
from sklearn.model_selection import ParameterGrid
from thop import profile

from shubuun.hw1.datamodule import YesNoDataModule


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model: torch.nn.Module, sample_input: torch.Tensor) -> int:
    """Return FLOPs (MACs) for one forward pass with the given input shape."""
    model.eval()
    sample_input = sample_input[:1]  # [B, ...] -> [1, ...]
    with torch.no_grad():
        macs, _ = profile(model, inputs=(sample_input,))

    # FLOP is about MACs / 2
    return macs // 2


class EpochTimeAndFlopsCallback(Callback):
    """Log epoch training time and (once) model parameter count and FLOPs."""

    def __init__(self):
        self._epoch_start_time: float | None = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        device = next(pl_module.parameters()).device
        dm = trainer.datamodule
        batch = next(iter(dm.val_dataloader()))
        x = batch[0].to(device)
        n_params = count_parameters(pl_module)
        flops = count_flops(pl_module, x)
        pl_module.train()
        trainer.logger.log_metrics({"n_params": n_params, "flops": flops}, step=0)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self._epoch_start_time = time.perf_counter()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if self._epoch_start_time is not None:
            elapsed = time.perf_counter() - self._epoch_start_time
            trainer.logger.log_metrics({"epoch_time_sec": elapsed}, step=trainer.current_epoch)


class YesNoModel(L.LightningModule):
    """Binary classifier (yes/no). Replace self.net with your architecture."""

    def __init__(self, lr: float = 1e-3, n_mels: int = 80, groups: int = 1):
        super().__init__()
        self.lr = lr
        # Simple 1D CNN: input (B, 80, T) log-mel -> conv blocks -> global pool -> 2 logits.
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_mels, 16, kernel_size=25, padding=12, groups=groups),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(16, 32, kernel_size=25, padding=12, groups=groups),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(-1).float()
        logits = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.unsqueeze(-1).float()
        logits = self(x)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        preds = torch.sigmoid(logits)
        acc = (preds > 0.5) == y
        self.log("val_loss", loss)
        self.log("val_acc", acc.float().mean())
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def train_run(params: dict, data_dir: str, log_dir: str = "logs"):
    """Train one run; logs (CSV, etc.) are saved under log_dir."""
    dm = YesNoDataModule(
        data_dir,
        logMelParams={"n_mels": params["n_mels"]},
        batch_size=params["batch_size"],
        num_workers=4,
    )
    model = YesNoModel(lr=params["lr"], n_mels=params["n_mels"], groups=params["groups"])
    trainer = L.Trainer(
        max_epochs=params["max_epochs"],
        logger=CSVLogger(log_dir),
        callbacks=[EpochTimeAndFlopsCallback()],
    )
    trainer.fit(model, dm)


def grid_search(data_dir: str, base_log_dir: str = "logs/gridsearch"):
    """Train for each param combination; logs under base_log_dir/run_0, run_1, ..."""
    param_grid = {"n_mels":[16, 32, 64, 128], "max_epochs":[10], "batch_size":[32], "lr":[1e-3], "groups":[1, 2, 4, 8, 16]}
    total = len(ParameterGrid(param_grid))
    for i, params in enumerate(ParameterGrid(param_grid)):
        log_dir = f"{base_log_dir}/run_{i}"
        print("\033[96m" + "-"* 70 + "\033[0m")
        print(f"\033[96mRun {i}/{total}: {params} -> {log_dir}\033[0m")
        print("\033[96m" + "-"* 70 + "\033[0m")
        train_run(params, data_dir, log_dir=log_dir)
        
        params_file = f"{log_dir}/params.json"
        with open(params_file, "w") as f:
            json.dump(params, f)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="SPEECHCOMMANDS root")
    p.add_argument("--log_dir", default="logs", help="Folder to save run logs (CSV, etc.)")
    p.add_argument(
        "--grid",
        action="store_true",
        help="Run grid search over lr and batch_size; use --log_dir as base folder",
    )
    args = p.parse_args()

    if args.grid:
        grid_search(args.data_dir)
    else:
        train_params = {
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "lr": args.lr,
            "n_mels": 80,
        }
        train_run(train_params, args.data_dir, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
