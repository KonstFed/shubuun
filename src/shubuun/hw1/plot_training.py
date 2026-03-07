"""Plot Lightning training metrics from CSV to a single PNG.

Usage:
  python -m shubuun.hw1.plot_training [--csv PATH] [--out PATH]

Defaults: --csv logs/lightning_logs/version_N/metrics.csv (latest version),
         --out logs/training_curves.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_latest_metrics_csv(logs_dir: Path) -> Path | None:
    """Return path to metrics.csv in the latest version_* under logs_dir."""
    lightning_logs = logs_dir / "lightning_logs"
    if not lightning_logs.exists():
        return None
    versions = sorted(
        (p for p in lightning_logs.iterdir() if p.is_dir() and p.name.startswith("version_")),
        key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else -1,
    )
    if not versions:
        return None
    csv_path = versions[-1] / "metrics.csv"
    return csv_path if csv_path.exists() else None


def plot_training_curves(csv_path: Path, out_path: Path) -> None:
    """Load metrics.csv and plot all training params into a single PNG."""
    df = pd.read_csv(csv_path)
    # Lightning leaves epoch empty when unchanged
    df["epoch"] = df["epoch"].ffill().fillna(0).astype(int)

    # Scalar metrics logged once (e.g. at step 0)
    scalars = {}
    for col in ("n_params", "flops"):
        if col in df.columns:
            val = df[col].dropna()
            if len(val):
                scalars[col] = int(val.iloc[0])

    # Epoch-level metrics (one value per epoch)
    epoch_cols = [
        c for c in ("train_loss_epoch", "val_loss", "val_acc", "epoch_time_sec") if c in df.columns
    ]
    # Step-level (many per epoch)
    step_cols = [c for c in ("train_loss_step",) if c in df.columns]

    n_plots = len(epoch_cols) + len(step_cols)
    if not n_plots:
        raise ValueError(f"No plottable metrics in {csv_path}. Columns: {list(df.columns)}")

    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 2.5 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Step-level: x = step
    for col in step_cols:
        ax = axes[idx]
        step = df["step"].values
        val = df[col].values
        mask = pd.notna(val)
        if mask.any():
            ax.plot(step[mask], val[mask], color="C0", alpha=0.8)
        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.set_title(col.replace("_", " ").title())
        idx += 1

    # Epoch-level: x = epoch
    for col in epoch_cols:
        ax = axes[idx]
        sub = df[["epoch", col]].dropna(subset=[col])
        if not sub.empty:
            # one row per epoch (take last if duplicated)
            by_epoch = sub.groupby("epoch", as_index=False).last()
            ax.plot(by_epoch["epoch"], by_epoch[col], marker="o", markersize=4, color="C0")
        ax.set_ylabel(col.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.set_title(col.replace("_", " ").title())
        idx += 1

    # Add scalar summary as text
    if scalars:
        text = "  |  ".join(f"{k}: {v:,}" for k, v in scalars.items())
        fig.suptitle(f"Training curves  ({text})", fontsize=10, y=1.02)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot Lightning training metrics to a single PNG.")
    p.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to metrics.csv (default: latest version in logs/lightning_logs)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("logs/training_curves.png"),
        help="Output PNG path (default: logs/training_curves.png)",
    )
    args = p.parse_args()

    csv_path = args.csv
    if csv_path is None:
        logs_dir = Path("logs")
        csv_path = find_latest_metrics_csv(logs_dir)
        if csv_path is None:
            raise SystemExit("No metrics.csv found. Run training first or pass --csv PATH.")
    else:
        csv_path = csv_path.resolve()
        if not csv_path.exists():
            raise SystemExit(f"File not found: {csv_path}")

    plot_training_curves(csv_path, args.out.resolve())


if __name__ == "__main__":
    main()
