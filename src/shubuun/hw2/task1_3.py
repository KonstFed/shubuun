import csv
from pathlib import Path

import jiwer
import matplotlib.pyplot as plt
import numpy as np
import torchaudio

from shubuun.hw2.wav2vec2decoder import Wav2Vec2Decoder

DATA_DIR = Path(__file__).parent
MANIFEST = DATA_DIR / "data/librispeech_test_other/manifest.csv"
LM_PATH = str(DATA_DIR / "lm/3-gram.pruned.1e-7.arpa.gz")
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_manifest():
    with open(MANIFEST) as f:
        return list(csv.DictReader(f))


def run(decoder, rows, method):
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
    results = []
    for w in widths:
        print(f"  beam_width={w}")
        decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=w)
        wer, cer = run(decoder, rows, "beam")
        results.append((w, wer, cer))
        print(f"    WER={wer:.2%}  CER={cer:.2%}")

    # table
    print(f"\n  {'beam_width':>12} {'WER':>8} {'CER':>8}")
    for w, wer, cer in results:
        print(f"  {w:>12} {wer:>7.2%} {cer:>7.2%}")

    # plot
    ws = [r[0] for r in results]
    wers = [r[1] * 100 for r in results]
    cers = [r[2] * 100 for r in results]
    fig, ax = plt.subplots()
    ax.plot(ws, wers, marker="o", label="WER")
    ax.plot(ws, cers, marker="s", label="CER")
    ax.set_xlabel("beam_width")
    ax.set_ylabel("%")
    ax.set_title("Beam Search: quality vs beam width")
    ax.legend()
    ax.set_xscale("log")
    ax.set_xticks(ws)
    ax.set_xticklabels(ws)
    fig.savefig(PLOTS_DIR / "task2_beam_width.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved → plots/task2_beam_width.png")


# ---------------------------------------------------------------------------
# Task 3 — temperature sweep (greedy)
# ---------------------------------------------------------------------------

def task3(rows):
    print("\n=== Task 3: Temperature Sweep (greedy) ===")
    temps = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
    results = []
    for t in temps:
        print(f"  T={t}")
        decoder = Wav2Vec2Decoder(lm_model_path=None, temperature=t)
        wer, cer = run(decoder, rows, "greedy")
        results.append((t, wer, cer))
        print(f"    WER={wer:.2%}  CER={cer:.2%}")

    print(f"\n  {'T':>6} {'WER':>8} {'CER':>8}")
    for t, wer, cer in results:
        print(f"  {t:>6} {wer:>7.2%} {cer:>7.2%}")

    ts = [r[0] for r in results]
    wers = [r[1] * 100 for r in results]
    fig, ax = plt.subplots()
    ax.plot(ts, wers, marker="o")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("WER %")
    ax.set_title("Temperature scaling effect on greedy WER (LibriSpeech)")
    fig.savefig(PLOTS_DIR / "task3_temperature.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved → plots/task3_temperature.png")


# ---------------------------------------------------------------------------
# Task 4 — shallow LM fusion, sweep alpha & beta
# ---------------------------------------------------------------------------

def task4(rows):
    print("\n=== Task 4: Shallow LM Fusion (alpha/beta sweep) ===")
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]
    wer_grid = np.zeros((len(betas), len(alphas)))

    for i, beta in enumerate(betas):
        for j, alpha in enumerate(alphas):
            print(f"  alpha={alpha}  beta={beta}")
            decoder = Wav2Vec2Decoder(lm_model_path=LM_PATH, beam_width=50, alpha=alpha, beta=beta)
            wer, cer = run(decoder, rows, "beam_lm")
            wer_grid[i, j] = wer * 100
            print(f"    WER={wer:.2%}  CER={cer:.2%}")

    # print table
    print(f"\n  WER(%) | " + " ".join(f"a={a:<5}" for a in alphas))
    for i, beta in enumerate(betas):
        row_str = "  ".join(f"{wer_grid[i, j]:5.2f}" for j in range(len(alphas)))
        print(f"  b={beta:<4} | {row_str}")

    # heatmap
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(wer_grid, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([str(b) for b in betas])
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title("Shallow LM Fusion WER% (LibriSpeech test-other)")
    for i in range(len(betas)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{wer_grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, label="WER %")
    fig.savefig(PLOTS_DIR / "task4_lm_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot saved → plots/task4_lm_heatmap.png")

    best_idx = np.unravel_index(wer_grid.argmin(), wer_grid.shape)
    print(f"  best: alpha={alphas[best_idx[1]]}  beta={betas[best_idx[0]]}  WER={wer_grid[best_idx]:.2f}%")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rows = load_manifest()
    task1(rows)
    task2(rows)
    task3(rows)
    task4(rows)
