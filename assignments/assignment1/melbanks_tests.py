import torch
import torchaudio
from melbanks import LogMelFilterBanks


def assert_log_mel_banks_equal(wav_path):
    signal, _ = torchaudio.load(wav_path)

    melspec = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(signal)
    logmelbanks = LogMelFilterBanks()(signal)

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)
    print(f"Test {wav_path} passed")


if __name__ == "__main__":
    assert_log_mel_banks_equal("applause_y.wav")
