"""Train yes/no classifier: python -m shubuun.hw1.train [--data_dir PATH]"""

import argparse
import time
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger
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
        batch = next(iter(dm.train_dataloader()))
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

    def __init__(self, lr: float = 1e-3):
        super().__init__()
        self.lr = lr
        # Simple 1D CNN: input (B, 80, T) log-mel -> conv blocks -> global pool -> 2 logits.
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(80, 32, kernel_size=25, padding=12),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(32, 64, kernel_size=25, padding=12),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(4),
            torch.nn.Conv1d(64, 128, kernel_size=25, padding=12),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(128, 1),
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data", help="SPEECHCOMMANDS root")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    dm = YesNoDataModule(
        args.data_dir,
        logMelParams={},
        batch_size=args.batch_size,
        num_workers=4,
    )
    model = YesNoModel(lr=args.lr)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger("logs"),
        callbacks=[EpochTimeAndFlopsCallback()],
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
