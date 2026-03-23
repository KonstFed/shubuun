import csv
import multiprocessing as mp
import os
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
from tqdm import tqdm

from shubuun.hw2.wav2vec2decoder import Wav2Vec2Decoder

DATA_DIR = Path(__file__).parent
MANIFEST = DATA_DIR / "data/librispeech_test_other/manifest.csv"
LM_PATH = str(DATA_DIR / "lm/3-gram.pruned.1e-7.arpa.gz")
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
N_WORKERS = min(4, os.cpu_count() or 1)


def load_manifest():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def run(decoder, rows, method):
    """Sequential eval — used for single runs (task 1)."""
    refs, hyps = [], []
    for i, row in enumerate(rows):
        audio, sr = torchaudio.load(DATA_DIR / row["path"])
        assert sr == 16000
        hyps.append(decoder.decode(audio, method=method))
        refs.append(row["text"])
        print(f"\r  {i + 1}/{len(rows)}", end="", flush=True)
    print()
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)


# ---------------------------------------------------------------------------
# Multiprocessing sweep infrastructure
# ---------------------------------------------------------------------------

def _sweep_worker(args):
    """Top-level worker: loads decoder with given params, evaluates all rows."""
    rows_data, method, lm_path, beam_width, alpha, beta, temperature = args
    decoder = Wav2Vec2Decoder(
        lm_model_path=lm_path, beam_width=beam_width,
        alpha=alpha, beta=beta, temperature=temperature,
    )
    refs, hyps = [], []
    for path, text in rows_data:
        audio, sr = torchaudio.load(DATA_DIR / path)
        hyps.append(decoder.decode(audio, method=method))
        refs.append(text)
    return alpha, beta, temperature, beam_width, jiwer.wer(refs, hyps), jiwer.cer(refs, hyps)


def sweep(rows, combos, method, lm_path=None):
    """
    Run combos in parallel using N_WORKERS processes.
    combos: list of (alpha, beta, temperature, beam_width)
    returns: dict (alpha, beta, temperature, beam_width) -> (wer, cer)
    """
    rows_data = [(row["path"], row["text"]) for row in rows]
    args_list = [
        (rows_data, method, lm_path, bw, alpha, beta, temp)
        for alpha, beta, temp, bw in combos
    ]
    results = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS) as pool:
        for alpha, beta, temp, bw, wer, cer in tqdm(
            pool.imap_unordered(_sweep_worker, args_list),
            total=len(args_list),
            desc="  sweep",
        ):
            results[(alpha, beta, temp, bw)] = (wer, cer)
            tqdm.write(f"    alpha={alpha} beta={beta} T={temp} bw={bw}  WER={wer:.2%} CER={cer:.2%}")
    return results


# ---------------------------------------------------------------------------
# Task 1 — greedy
# ---------------------------------------------------------------------------

def task1(rows):
    print("\n=== Task 1: Greedy ===")
    decoder = Wav2Vec2Decoder(lm_model_path=None)
    wer, cer = run(decoder, rows, "greedy")
    print(f"  WER={wer:.2%}  CER={cer:.2%}")


# ---------------------------------------------------------------------------
# Task 2 — beam search, vary beam_width
# ---------------------------------------------------------------------------

def task2(rows):
    print("\n=== Task 2: Beam Search (varying beam_width) ===")
    widths = [1, 3, 10, 50]
    combos = [(1.0, 1.0, 1.0, w) for w in widths]
    results = sweep(rows, combos, "beam")

    rows_res = [(w, *results[(1.0, 1.0, 1.0, w)]) for w in widths]
    print(f"\n  {'beam_width':>12} {'WER':>8} {'CER':>8}")
    for w, wer, cer in rows_res:
        print(f"  {w:>12} {wer:>7.2%} {cer:>7.2%}")

    fig, ax = plt.subplots()
    ax.plot(widths, [r[1] * 100 for r in rows_res], marker="o", label="WER")
    ax.plot(widths, [r[2] * 100 for r in rows_res], marker="s", label="CER")
    ax.set_xscale("log")
    ax.set_xticks(widths)
    ax.set_xticklabels(widths)
    ax.set_xlabel("beam_width")
    ax.set_ylabel("%")
    ax.set_title("Beam Search: quality vs beam width")
    ax.legend()
    fig.savefig(PLOTS_DIR / "task2_beam_width.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  plot saved → plots/task2_beam_width.png")


# ---------------------------------------------------------------------------
# Task 3 — temperature sweep (greedy)
# ---------------------------------------------------------------------------

def task3(rows):
    print("\n=== Task 3: Temperature Sweep (greedy) ===")
    temps = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    combos = [(1.0, 1.0, t, 1) for t in temps]
    results = sweep(rows, combos, "greedy")

    rows_res = [(t, *results[(1.0, 1.0, t, 1)]) for t in temps]
    print(f"\n  {'T':>6} {'WER':>8} {'CER':>8}")
    for t, wer, cer in rows_res:
        print(f"  {t:>6} {wer:>7.2%} {cer:>7.2%}")

    fig, ax = plt.subplots()
    ax.plot(temps, [r[1] * 100 for r in rows_res], marker="o")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("WER %")
    ax.set_title("Temperature scaling: greedy WER (LibriSpeech)")
    fig.savefig(PLOTS_DIR / "task3_temperature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  plot saved → plots/task3_temperature.png")


# ---------------------------------------------------------------------------
# Task 4 — shallow LM fusion, sweep alpha & beta
# ---------------------------------------------------------------------------

def _lm_heatmap(wer_grid, alphas, betas, title, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(wer_grid, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([str(b) for b in betas])
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title(title)
    for i in range(len(betas)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{wer_grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="WER %")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def task4(rows):
    print("\n=== Task 4: Shallow LM Fusion (alpha/beta sweep) ===")
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    combos = [(a, b, 1.0, 50) for b in betas for a in alphas]
    results = sweep(rows, combos, "beam_lm", lm_path=LM_PATH)

    wer_grid = np.array([[results[(a, b, 1.0, 50)][0] * 100 for a in alphas] for b in betas])

    print(f"\n  WER(%) | " + " ".join(f"a={a:<5}" for a in alphas))
    for i, b in enumerate(betas):
        print(f"  b={b:<4} | " + "  ".join(f"{wer_grid[i, j]:5.2f}" for j in range(len(alphas))))

    _lm_heatmap(wer_grid, alphas, betas, "Shallow LM Fusion WER% (LibriSpeech)", PLOTS_DIR / "task4_lm_heatmap.png")
    print("  plot saved → plots/task4_lm_heatmap.png")

    best = np.unravel_index(wer_grid.argmin(), wer_grid.shape)
    print(f"  best: alpha={alphas[best[1]]}  beta={betas[best[0]]}  WER={wer_grid[best]:.2f}%")


# ---------------------------------------------------------------------------
# Task 6 — second-pass LM rescoring, sweep alpha & beta
# ---------------------------------------------------------------------------

def task6(rows):
    print("\n=== Task 6: LM Rescoring (alpha/beta sweep) ===")
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    combos = [(a, b, 1.0, 50) for b in betas for a in alphas]
    results = sweep(rows, combos, "beam_lm_rescore", lm_path=LM_PATH)

    wer_grid = np.array([[results[(a, b, 1.0, 50)][0] * 100 for a in alphas] for b in betas])

    print(f"\n  WER(%) | " + " ".join(f"a={a:<5}" for a in alphas))
    for i, b in enumerate(betas):
        print(f"  b={b:<4} | " + "  ".join(f"{wer_grid[i, j]:5.2f}" for j in range(len(alphas))))

    _lm_heatmap(wer_grid, alphas, betas, "LM Rescoring WER% (LibriSpeech)", PLOTS_DIR / "task6_rescore_heatmap.png")
    print("  plot saved → plots/task6_rescore_heatmap.png")

    best = np.unravel_index(wer_grid.argmin(), wer_grid.shape)
    print(f"  best: alpha={alphas[best[1]]}  beta={betas[best[0]]}  WER={wer_grid[best]:.2f}%")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_manifest()
    task1(rows)
    task2(rows)
    task3(rows)
    task4(rows)
    task6(rows)
