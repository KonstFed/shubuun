"""Tests for LogMelFilterBanks."""
from pathlib import Path

import pytest
import torch
import torchaudio

from shubuun.hw1.melbanks import LogMelFilterBanks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.mark.parametrize("wav_name", ["tuco-get-out.wav"])
@pytest.mark.parametrize("n_fft,hop_length,n_mels", [
    (400, 160, 80),
    (512, 256, 40),
    (256, 128, 64),
])
def test_log_mel_filter_to_torch(
    wav_name: str, n_fft: int, hop_length: int, n_mels: int
) -> None:
    """LogMelFilterBanks matches torchaudio MelSpectrogram for given params."""
    signal, _sr = torchaudio.load(str(DATA_DIR / wav_name))

    # our melspec
    melspec = torchaudio.transforms.MelSpectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )(signal)

    # gt melspec
    model = LogMelFilterBanks(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    logmelbanks = model(signal)

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
    assert model.n_fft == n_fft
    assert model.n_mels == n_mels
