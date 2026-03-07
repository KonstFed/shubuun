"""Tests for LogMelFilterBanks vs torchaudio."""
from itertools import product
from pathlib import Path

import pytest
import torch
import torchaudio
from torchaudio import functional as F

from shubuun.hw1.melbanks import LogMelFilterBanks

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def signal():
    wav, _ = torchaudio.load(str(DATA_DIR / "tuco-get-out.wav"))
    return wav


@pytest.mark.parametrize(
    "n_fft,hop_length,n_mels,samplerate,power,center,pad_mode,normalize_stft,"
    "f_min_hz,f_max_hz,norm_mel,mel_scale",
    product(
        [400, 512],        # n_fft
        [128, 160],             # hop_length
        [40, 80],               # n_mels
        [8000, 16000],          # samplerate
        [1.0, 2.0],            # power
        [True, False],          # center
        ["reflect", "constant"],# pad_mode
        [True, False],          # normalize_stft
        [0.0, 100.0],          # f_min_hz
        [None, 4000.0],        # f_max_hz
        [None, "slaney"],      # norm_mel
        ["htk", "slaney"],     # mel_scale
    ),
)
def test_forward(
    signal, n_fft, hop_length, n_mels, samplerate, power, center, pad_mode,
    normalize_stft, f_min_hz, f_max_hz, norm_mel, mel_scale,
):
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=samplerate, n_fft=n_fft, win_length=n_fft,
        hop_length=hop_length, f_min=f_min_hz, f_max=f_max_hz,
        n_mels=n_mels, power=power, normalized=normalize_stft,
        center=center, pad_mode=pad_mode,
        norm=norm_mel, mel_scale=mel_scale,
    )(signal)
    logmelbanks = LogMelFilterBanks(
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        samplerate=samplerate, power=power, center=center,
        pad_mode=pad_mode, normalize_stft=normalize_stft,
        f_min_hz=f_min_hz, f_max_hz=f_max_hz,
        norm_mel=norm_mel, mel_scale=mel_scale,
    )(signal)
    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
