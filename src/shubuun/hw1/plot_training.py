"""Plot gridsearch training metrics. Usage: python -m shubuun.hw1.plot_training [--grid-dir PATH] [--out-dir PATH]"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _latest_metrics_csv(run_dir: Path) -> Path | None:
    lightning = run_dir / "lightning_logs"
    if not lightning.exists():
        return None
    versions = sorted(
        (x for x in lightning.iterdir() if x.is_dir() and x.name.startswith("version_")),
        key=lambda x: int(x.name.split("_")[1]) if x.name.split("_")[1].isdigit() else -1,
    )
    if not versions:
        return None
    p = versions[-1] / "metrics.csv"
    return p if p.exists() else None


def _load_run(run_dir: Path) -> tuple[dict, pd.DataFrame, dict] | None:
    params = {}
    if (run_dir / "params.json").exists():
        with open(run_dir / "params.json") as f:
            params = json.load(f)
    csv_path = _latest_metrics_csv(run_dir)
    if not csv_path:
        return None
    df = pd.read_csv(csv_path)
    df["epoch"] = df["epoch"].ffill().fillna(0).astype(int)

    scalars = {}
    for col in ("n_params", "flops"):
        v = df[col].dropna()
        if len(v):
            scalars[col] = int(v.iloc[0])

    rows = []
    for ep in df["epoch"].dropna().unique():
        sub = df[df["epoch"] == ep]
        row = {"epoch": int(ep)}
        for c in ("train_loss_epoch", "val_acc", "val_loss", "epoch_time_sec"):
            if c in df.columns:
                v = sub[c].dropna()
                if len(v):
                    row[c] = v.iloc[-1]
        rows.append(row)
    epoch_df = pd.DataFrame(rows).sort_values("epoch") if rows else pd.DataFrame()
    return params, epoch_df, scalars


def _load_grid(grid_dir: Path) -> list[tuple[str, dict, pd.DataFrame, dict]]:
    out = []
    for p in sorted(Path(grid_dir).iterdir()):
        if not p.is_dir() or not p.name.startswith("run_"):
            continue
        r = _load_run(p)
        if r:
            out.append((p.name, *r))
    return out


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", type=Path, default=Path("logs/gridsearch"))
    p.add_argument("--out-dir", type=Path, default=Path("logs/gridsearch/plots"))
    args = p.parse_args()

    runs = _load_grid(args.grid_dir.resolve())
    if not runs:
        raise SystemExit(f"No runs in {args.grid_dir}")

    out = Path(args.out_dir)
    summary_rows = []
    for run_id, params, epoch_df, scalars in runs:
        summary_rows.append({
            "run_id": run_id,
            "groups": params.get("groups"),
            "n_mels": params.get("n_mels"),
            "n_params": scalars.get("n_params"),
            "flops": scalars.get("flops"),
            "mean_epoch_time_sec": epoch_df["epoch_time_sec"].mean() if "epoch_time_sec" in epoch_df.columns and len(epoch_df) else None,
            "final_val_acc": epoch_df["val_acc"].iloc[-1] if "val_acc" in epoch_df.columns and len(epoch_df) else None,
        })
    summary = pd.DataFrame(summary_rows)

    def label(run_id: str, params: dict) -> str:
        return f"{run_id} (g={params.get('groups')}, m={params.get('n_mels')})"

    # Train loss over all runs
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id, params, epoch_df, _ in runs:
        if "train_loss_epoch" in epoch_df.columns and len(epoch_df):
            ax.plot(epoch_df["epoch"], epoch_df["train_loss_epoch"], marker="o", markersize=3, label=label(run_id, params), alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train loss")
    ax.set_title("Train loss (all runs)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out, "train_loss_comparison.png")

    # Val acc over all runs
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_id, params, epoch_df, _ in runs:
        if "val_acc" in epoch_df.columns and len(epoch_df):
            ax.plot(epoch_df["epoch"], epoch_df["val_acc"], marker="o", markersize=3, label=label(run_id, params), alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val accuracy")
    ax.set_title("Val accuracy (all runs)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, out, "val_acc_comparison.png")

    # n_mels vs final val_acc
    v = summary.dropna(subset=["n_mels", "final_val_acc"])
    if len(v):
        fig, ax = plt.subplots(figsize=(7, 4))
        for g in v["groups"].dropna().unique():
            s = v[v["groups"] == g].sort_values("n_mels")
            ax.plot(s["n_mels"], s["final_val_acc"], marker="o", label=f"groups={g}")
        ax.set_xlabel("n_mels")
        ax.set_ylabel("Final val accuracy")
        ax.set_title("n_mels vs accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, out, "n_mels_vs_accuracy.png")

    # Epoch time vs groups
    v = summary.dropna(subset=["groups", "mean_epoch_time_sec"])
    if len(v):
        fig, ax = plt.subplots(figsize=(7, 4))
        for m in v["n_mels"].dropna().unique():
            s = v[v["n_mels"] == m].sort_values("groups")
            ax.plot(s["groups"], s["mean_epoch_time_sec"], marker="o", label=f"n_mels={m}")
        ax.set_xlabel("groups")
        ax.set_ylabel("Mean epoch time (sec)")
        ax.set_title("Epoch time vs groups")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, out, "epoch_time_vs_groups.png")

    # n_params vs groups and n_mels
    v = summary.dropna(subset=["n_params"])
    if len(v):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for m in v["n_mels"].dropna().unique():
            s = v[v["n_mels"] == m].sort_values("groups")
            ax1.plot(s["groups"], s["n_params"], marker="o", label=f"n_mels={m}")
        ax1.set_xlabel("groups")
        ax1.set_ylabel("Parameters")
        ax1.set_title("Parameters vs groups")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        for g in v["groups"].dropna().unique():
            s = v[v["groups"] == g].sort_values("n_mels")
            ax2.plot(s["n_mels"], s["n_params"], marker="o", label=f"groups={g}")
        ax2.set_xlabel("n_mels")
        ax2.set_ylabel("Parameters")
        ax2.set_title("Parameters vs n_mels")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, out, "n_params_vs_groups_and_mels.png")

    # FLOPs vs groups and n_mels
    v = summary.dropna(subset=["flops"])
    if len(v):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for m in v["n_mels"].dropna().unique():
            s = v[v["n_mels"] == m].sort_values("groups")
            ax1.plot(s["groups"], s["flops"], marker="o", label=f"n_mels={m}")
        ax1.set_xlabel("groups")
        ax1.set_ylabel("FLOPs")
        ax1.set_title("FLOPs vs groups")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        for g in v["groups"].dropna().unique():
            s = v[v["groups"] == g].sort_values("n_mels")
            ax2.plot(s["n_mels"], s["flops"], marker="o", label=f"groups={g}")
        ax2.set_xlabel("n_mels")
        ax2.set_ylabel("FLOPs")
        ax2.set_title("FLOPs vs n_mels")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, out, "flops_vs_groups_and_mels.png")


if __name__ == "__main__":
    main()
