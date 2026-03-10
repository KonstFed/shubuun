# Shubuun

Course project and homework assignments (DSP / audio ML). Python ≥3.12.

## Setup

```bash
uv sync
```

Optional dev deps: `uv sync --extra dev`. Uses PyTorch, Lightning, torchaudio, thop (FLOPs), scikit-learn.

## HW1: LogMelFilterBanks + Yes/No CNN

- **LogMelFilterBanks** (`src/shubuun/hw1/melbanks.py`): `nn.Module` for log Mel filterbank energies — STFT with Hann window, power spectrum, then `F.melscale_fbanks()` and log. Matches `torch.log(torchaudio.transforms.MelSpectrogram(...)(x) + 1e-6)`.
- **Yes/No classifier**: Conv1d CNN on Speech Commands (Yes/No only), &lt;100K params. LogMelFilterBanks as front-end; training logs train loss, val accuracy, epoch time, n_params, FLOPs.

### Data

[Google Speech Commands](https://arxiv.org/abs/1804.03209) — dataset is downloaded automatically to `data/` on first run (via `torchaudio.datasets.SPEECHCOMMANDS`).

### Train

Single run (default `n_mels=80`, `groups=1`):

```bash
uv run python -m shubuun.hw1.train --data_dir data --log_dir logs
```

Grid search over `n_mels`, `groups`, etc. (logs under `logs/gridsearch/run_*`):

```bash
uv run python -m shubuun.hw1.train --data_dir data --grid
```

### Test

```bash
uv run pytest tests/
```

LogMelFilterBanks is validated against torchaudio in `tests/test_hw1/test_melbanks.py` over a parameter grid (shape and numerical closeness).

### Report

`report/report.md` and `report/plots/` — comparison plots, metrics vs `n_mels`/`groups`, conclusions.
