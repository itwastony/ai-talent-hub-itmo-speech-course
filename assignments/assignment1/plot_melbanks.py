import torch
import torchaudio
import matplotlib.pyplot as plt
from melbanks import LogMelFilterBanks


def plot_melbanks_comparison(audio_path):
    signal, sr = torchaudio.load(audio_path)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        signal = resampler(signal)

    our_melbanks = LogMelFilterBanks()(signal)
    torchaudio_mel = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(
        signal
    )
    torchaudio_log_mel = torch.log(torchaudio_mel + 1e-6)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    im1 = ax1.imshow(
        our_melbanks[0].numpy(), aspect="auto", origin="lower", cmap="viridis"
    )
    ax1.set_title("Our LogMelFilterBanks Implementation")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Mel Frequency")
    plt.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(
        torchaudio_log_mel[0].numpy(), aspect="auto", origin="lower", cmap="viridis"
    )
    ax2.set_title("Torchaudio MelSpectrogram (with log)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Mel Frequency")
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig("melbanks_comparison.png")
    plt.close()


if __name__ == "__main__":
    plot_melbanks_comparison("applause_y.wav")
