"""Tests for LogMelFilterBanks."""

import pytest
import torch

from itmoaudio.hw1 import LogMelFilterBanks


def test_log_mel_filter_banks_instantiates():
    """LogMelFilterBanks can be created with default args."""
    model = LogMelFilterBanks()
    assert model.n_fft == 400
    assert model.n_mels == 80
