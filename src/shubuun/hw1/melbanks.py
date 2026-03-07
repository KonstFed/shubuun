from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        # Do correct initialization of stft params below:
        # hop_length, n_mels, center, return_complex, onesided, normalize_stft, pad_mode, power
        # ...
        # <YOUR CODE GOES HERE>
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power

        # Do correct initialization of mel fbanks params below:
        # f_min_hz, f_max_hz, norm_mel, mel_scale
        # ...
        # <YOUR CODE GOES HERE>
        self.f_min_hz = f_min_hz
        self.f_max_hz = samplerate / 2 if f_max_hz is None else f_max_hz
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        # finish parameters initialization
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        n_freqs = self.n_fft // 2 + 1 if self.onesided else self.n_fft
        return F.melscale_fbanks(
            n_freqs=n_freqs,
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,
            mel_scale=self.mel_scale,
        )

    def spectrogram(self, x):
        # # x - is an input signal
        # return torch.stft(
        #     # make a full and correct function call
        #     # <YOUR CODE GOES HERE>
        # )

        window = self.window.to(device=x.device, dtype=x.dtype)
        stft = torch.stft(
            input=x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_length,
            window=window,
            center=self.center,
            pad_mode=self.pad_mode,
            normalized=False, # torchaudio uses always False
            onesided=self.onesided,
            return_complex=self.return_complex,
        )

        if self.normalize_stft:
            stft = stft / window.pow(2.0).sum().sqrt()

        # Power spectrogram: |X|^power
        spec = torch.abs(stft) ** self.power
        return spec

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        spec = self.spectrogram(x)
        mel_fbanks = self.mel_fbanks.to(device=x.device, dtype=spec.dtype)
        mel_spec = spec.transpose(1, 2) @ mel_fbanks
        mel_spec = mel_spec.transpose(1, 2)
        log_mel_spec = torch.log(mel_spec + 1e-6)

        return log_mel_spec
