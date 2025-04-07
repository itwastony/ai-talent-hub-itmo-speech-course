import argparse

import torch
import torchaudio
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Timer
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from melbanks import LogMelFilterBanks
from thop import profile
import matplotlib.pyplot as plt
from typing import List, Tuple
import os
from pathlib import Path

from torchaudio.datasets import SPEECHCOMMANDS


class BinarySpeechCommandsDataset(SPEECHCOMMANDS):
    def __init__(self, subset: str, n_mels: int = 80, max_frames: int = 100):
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        super().__init__(root=data_dir, download=True, subset=subset)

        self.mel_transform = LogMelFilterBanks(n_mels=n_mels)
        self.max_frames = max_frames

    def __getitem__(self, index):
        waveform, _, label, _, _ = super().__getitem__(index)
        mel_spec = self.mel_transform(waveform)

        if mel_spec.shape[-1] > self.max_frames:
            mel_spec = mel_spec[:, :, : self.max_frames]
        elif mel_spec.shape[-1] < self.max_frames:
            pad_width = self.max_frames - mel_spec.shape[-1]
            mel_spec = torch.nn.functional.pad(
                mel_spec, (0, pad_width), mode="constant", value=0
            )

        binary_label = 1 if label in ["yes", "no"] else 0

        mel_spec = mel_spec.view(mel_spec.shape[1], mel_spec.shape[2])
        return mel_spec, binary_label


class SpeechCNN(pl.LightningModule):
    def __init__(self, n_mels: int = 80, groups: int = 1):
        super().__init__()
        self.save_hyperparameters()

        self.conv1 = nn.Conv1d(n_mels, 32, kernel_size=3, groups=groups)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, groups=groups)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, groups=groups)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool1d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool1d(x, 2)
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(1).float())
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(1).float())
        acc = (y_hat.squeeze() > 0).float().eq(y).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss, acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.unsqueeze(1).float())
        acc = (y_hat.squeeze() > 0).float().eq(y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return loss, acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_flops(self, input_size: tuple = None):
        input_tensor = torch.randn(input_size)
        flops, params = profile(self, inputs=(input_tensor,))
        return flops


def train_model(n_mels: int, groups: int = 1, max_epochs: int = 10):
    train_dataset = BinarySpeechCommandsDataset("training", n_mels=n_mels)
    val_dataset = BinarySpeechCommandsDataset("validation", n_mels=n_mels)
    test_dataset = BinarySpeechCommandsDataset("testing", n_mels=n_mels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(
        val_dataset, batch_size=32, num_workers=4, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, num_workers=4, persistent_workers=True
    )

    model = SpeechCNN(n_mels=n_mels, groups=groups)

    logger = TensorBoardLogger(
        "lightning_logs", name=f"n_mels_{n_mels}_groups_{groups}"
    )

    timer = Timer()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                save_top_k=1,
                filename="best-{epoch:02d}-{val_acc:.2f}",
            ),
            timer,
        ],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    train_time = timer.time_elapsed("train")
    print(f"Total training time: {train_time:.2f} seconds")

    return model, trainer, train_time


def plot_results(
    results: List[Tuple[int, float, float, float, int, float]], filename: str
):
    n_mels_list, train_times, params_list, flops_list, groups_list, test_accs = zip(
        *results
    )

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(n_mels_list, test_accs, "o-")
    ax1.set_title("Test Accuracy vs Number of Mel Bands")
    ax1.set_xlabel("Number of Mel Bands")
    ax1.set_ylabel("Test Accuracy")

    ax2.plot(groups_list, train_times, "o-")
    ax2.set_title("Training Time vs Groups")
    ax2.set_xlabel("Groups")
    ax2.set_ylabel("Training Time (s)")

    ax3.plot(groups_list, params_list, "o-")
    ax3.set_title("Number of Parameters vs Groups")
    ax3.set_xlabel("Groups")
    ax3.set_ylabel("Number of Parameters")

    ax4.plot(groups_list, flops_list, "o-")
    ax4.set_title("FLOPs vs Groups")
    ax4.set_xlabel("Groups")
    ax4.set_ylabel("FLOPs")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1)
    args = parser.parse_args()
    results = []

    # Experiment with different number of mel bands
    for n_mels in [20, 40, 80]:
        model, trainer, train_time = train_model(
            n_mels=n_mels, max_epochs=args.max_epochs
        )
        test_acc = trainer.callback_metrics["test_acc"]
        params = model.count_parameters()
        flops = model.count_flops(input_size=(1, n_mels, 100))
        results.append((n_mels, train_time, params, flops, 1, test_acc))

    plot_results(results, filename="n_mels_results.png")
    results = []
    # Experiment with different group values
    for groups in [2, 4, 8, 16]:
        model, trainer, train_time = train_model(
            n_mels=80, groups=groups, max_epochs=args.max_epochs
        )
        test_acc = trainer.callback_metrics["test_acc"]
        params = model.count_parameters()
        flops = model.count_flops(input_size=(1, 80, 100))
        results.append((80, train_time, params, flops, groups, test_acc))

    plot_results(results, filename="groups_results.png")
