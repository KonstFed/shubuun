"""Lightning DataModule for Yes/No SPEECHCOMMANDS."""

import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from torchaudio.datasets import SPEECHCOMMANDS

from shubuun.hw1.melbanks import LogMelFilterBanks

LABELS = ("no", "yes")


class YesNoSpeechCommands(Dataset):
    """SPEECHCOMMANDS filtered to 'yes' and 'no' only. Returns (waveform, label_idx)."""

    def __init__(
        self, root: str, logMelParams: dict, download: bool = True, subset: str = "training"
    ):
        self._to_mel = LogMelFilterBanks(**logMelParams)
        self._ds = SPEECHCOMMANDS(root, download=download, subset=subset)
        self._indices = [i for i in range(len(self._ds)) if self._ds.get_metadata(i)[2] in LABELS]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        waveform, _, label, *_ = self._ds[self._indices[idx]]
        mel = self._to_mel(waveform)
        mel = mel[0]  # we have only one channel
        return mel, LABELS.index(label)


def collate_pad(batch):
    """Pad/crop waveforms to fixed length (longest in dataset)."""
    waveforms, labels = zip(*batch)
    out = []
    length = max(w.shape[-1] for w in waveforms)
    for w in waveforms:
        out.append(torch.nn.functional.pad(w, (0, length - w.shape[-1])))
    return torch.stack(out), torch.tensor(labels, dtype=torch.long)


class YesNoDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        logMelParams: dict,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.logMelParams = logMelParams

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = YesNoSpeechCommands(self.data_dir, self.logMelParams, subset="training")
        self.val_ds = YesNoSpeechCommands(self.data_dir, self.logMelParams, subset="validation")
        self.test_ds = YesNoSpeechCommands(self.data_dir, self.logMelParams, subset="testing")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_pad,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_pad,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_pad,
        )
