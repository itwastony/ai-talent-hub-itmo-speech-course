import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torchaudio.datasets import LJSPEECH
from torchaudio.transforms import MelSpectrogram
from pathlib import Path
import random
import torchaudio
import itertools

from TTS.api import TTS
from TTS.tts.utils.synthesis import synthesis
from torch.nn.utils import weight_norm, remove_weight_norm


class TextToSpecConverter:
    def __init__(self, model_name: str = "tts_models/en/ljspeech/fast_pitch"):
        self.model_name = model_name
        self.tts_handler = TTS(model_name=model_name)
        self.model = self.tts_handler.synthesizer.tts_model
        self.config = self.tts_handler.synthesizer.tts_config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_cuda = True if self.device == "cuda" else False
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
        print(f"Model {model_name} loaded on {self.device}")

    def text2spec(self, text: str):
        outputs = synthesis(
            self.model,
            text,
            self.config,
            self.use_cuda,
            use_griffin_lim=False,
            do_trim_silence=False,
        )
        mel_spec = outputs["outputs"]["model_outputs"][0].detach().cpu()
        mel_spec_np = self.model.ap.denormalize(mel_spec.T.numpy()).T
        return mel_spec_np


class VocoderConfig:
    def __init__(self, tts_config):
        self.sample_rate = tts_config["audio"]["sample_rate"]
        self.n_fft = tts_config["audio"]["fft_size"]
        self.hop_length = tts_config["audio"]["hop_length"]
        self.win_length = tts_config["audio"]["win_length"]
        self.n_mels = tts_config["audio"]["num_mels"]

        # HiFi-GAN
        self.resblock_kernel_sizes = [3, 7, 11]
        self.resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        self.upsample_rates = [8, 8, 2, 2]
        self.upsample_initial_channel = 512
        self.upsample_kernel_sizes = [16, 16, 4, 4]

        self.segment_size = 8192
        self.batch_size = 96
        self.learning_rate = 2e-4
        self.adam_betas = (0.8, 0.99)
        self.lr_decay = 0.999

        self.lambda_mel = 45.0
        self.lambda_fm = 2.0

        self.total_steps = 50000
        self.steps_per_log = 200
        self.steps_per_checkpoint = 10000


class ResBlockV1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlockV1, self).__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=((kernel_size - 1) * d) // 2,
                    )
                )
                for d in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        padding=(kernel_size - 1) // 2,
                    )
                )
                for _ in range(len(dilation))
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)

        self.conv_pre = weight_norm(
            nn.Conv1d(config.n_mels, config.upsample_initial_channel, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(config.upsample_rates, config.upsample_kernel_sizes)
        ):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        config.upsample_initial_channel // (2**i),
                        config.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = config.upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(
                config.resblock_kernel_sizes, config.resblock_dilation_sizes
            ):
                self.resblocks.append(ResBlockV1(ch, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = self.ups[i](x)
            xs = 0
            for j in range(self.num_kernels):
                xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super(MultiPeriodDiscriminator, self).__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList(
            [self._create_discriminator(p) for p in self.periods]
        )

    def _create_discriminator(self, period):
        return nn.Sequential(
            weight_norm(nn.Conv2d(1, 32, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(32, 128, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(128, 512, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0))),
            weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))),
        )

    def forward(self, x):
        outputs = []
        features = []
        for d, p in zip(self.discriminators, self.periods):
            if x.shape[2] % p != 0:
                x = F.pad(x, (0, p - (x.shape[2] % p)), "reflect")
            x_reshaped = x.view(x.shape[0], 1, -1, p)

            feat_block = []
            for layer in d:
                x_reshaped = F.leaky_relu(layer(x_reshaped), 0.1)
                feat_block.append(x_reshaped)
            outputs.append(x_reshaped)
            features.append(feat_block[:-1])
        return outputs, features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                nn.Sequential(
                    weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                    weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                    weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                    weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                    weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                    weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
                    weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1)),
                ),
                nn.Sequential(
                    *[
                        weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                        weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                        weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                        weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                        weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                        weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
                        weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1)),
                    ]
                ),
                nn.Sequential(
                    *[
                        weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
                        weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                        weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                        weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                        weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                        weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
                        weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1)),
                    ]
                ),
            ]
        )
        self.pools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, x):
        outputs, features = [], []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pools[i - 1](x)
            feat_block = []
            for layer in d:
                x = F.leaky_relu(layer(x), 0.1)
                feat_block.append(x)
            outputs.append(x)
            features.append(feat_block[:-1])
        return outputs, features


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, x):
        mpd_outs, mpd_feats = self.mpd(x)
        msd_outs, msd_feats = self.msd(x)
        return mpd_outs + msd_outs, mpd_feats + msd_feats


def collate_fn(batch, config):
    mel_spectrogram_transform = MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    )
    waveforms, mels = [], []
    for waveform, _, _, _ in batch:
        if waveform.size(1) < config.segment_size:
            continue
        start = random.randint(0, waveform.size(1) - config.segment_size)
        segment = waveform[:, start : start + config.segment_size]
        waveforms.append(segment)
        mel = mel_spectrogram_transform(segment)
        mels.append(mel)
    if not waveforms:
        return None, None
    return torch.stack(waveforms), torch.stack(mels).squeeze(1)


# Helper to crop tensors to the same length for feature matching loss
def crop_to_min_size_dim(tensor_a, tensor_b, dim=-1):
    """Crop two tensors to the minimum size along a specified dimension."""
    min_size = min(tensor_a.size(dim), tensor_b.size(dim))
    # Create slicing objects
    slices_a = [slice(None)] * tensor_a.ndim
    slices_b = [slice(None)] * tensor_b.ndim
    slices_a[dim] = slice(0, min_size)
    slices_b[dim] = slice(0, min_size)
    return tensor_a[tuple(slices_a)], tensor_b[tuple(slices_b)]


def train(config, dataset, resume_from_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = Path("./checkpoints_hifigan")
    checkpoint_dir.mkdir(exist_ok=True)

    generator = Generator(config).to(device)
    discriminator = Discriminator().to(device)

    optim_g = optim.AdamW(
        generator.parameters(), lr=config.learning_rate, betas=config.adam_betas
    )
    optim_d = optim.AdamW(
        discriminator.parameters(), lr=config.learning_rate, betas=config.adam_betas
    )
    scheduler_g = ExponentialLR(optim_g, gamma=config.lr_decay)
    scheduler_d = ExponentialLR(optim_d, gamma=config.lr_decay)

    steps = 0
    if resume_from_checkpoint and Path(resume_from_checkpoint).exists():
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        ckpt = torch.load(resume_from_checkpoint, map_location=device)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        optim_g.load_state_dict(ckpt["optim_g"])
        optim_d.load_state_dict(ckpt["optim_d"])
        scheduler_g.load_state_dict(ckpt["scheduler_g"])
        scheduler_d.load_state_dict(ckpt["scheduler_d"])
        steps = ckpt["steps"]
        print(f"Loaded model at step {steps}. Resuming.")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, config),
        num_workers=8,
        pin_memory=True,
    )
    data_iterator = itertools.cycle(loader)
    mel_transform = MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
    ).to(device)

    generator.train()
    discriminator.train()

    while steps < config.total_steps:
        real_wav, mel_spec = next(data_iterator)
        if real_wav is None:
            continue
        real_wav, mel_spec = real_wav.to(device), mel_spec.to(device)

        optim_d.zero_grad()
        fake_wav = generator(mel_spec).detach()
        real_outputs, _ = discriminator(real_wav)
        fake_outputs, _ = discriminator(fake_wav)
        loss_d = 0
        for out_r, out_f in zip(real_outputs, fake_outputs):
            loss_d += torch.mean((1 - out_r) ** 2) + torch.mean(out_f**2)
        loss_d.backward()
        optim_d.step()

        optim_g.zero_grad()
        fake_wav = generator(mel_spec)
        _, real_features = discriminator(
            real_wav
        )  # real features again for generator update
        adv_outputs, fake_features = discriminator(
            fake_wav
        )  # fake features for generator update

        loss_adv = 0
        for out in adv_outputs:
            loss_adv += torch.mean((1 - out) ** 2)

        loss_fm = 0
        for i in range(len(real_features)):  # Iterate through MPD/MSD feature sets
            for j in range(
                len(real_features[i])
            ):  # Iterate through layers within each feature set
                real_feat = real_features[i][j].detach()  # Detach real features here
                fake_feat = fake_features[i][j]

                # Dynamically determine dimensions to crop based on feature map dimensionality
                # MPD features are 4D (batch, channels, time_per_period, period)
                # MSD features are 3D (batch, channels, time_frames)

                # Apply cropping to the relevant time/spatial dimensions
                if real_feat.ndim == 4:  # MPD features
                    real_feat, fake_feat = crop_to_min_size_dim(
                        real_feat, fake_feat, dim=2
                    )  # Crop time_per_period
                    real_feat, fake_feat = crop_to_min_size_dim(
                        real_feat, fake_feat, dim=3
                    )  # Crop period
                elif real_feat.ndim == 3:  # MSD features
                    real_feat, fake_feat = crop_to_min_size_dim(
                        real_feat, fake_feat, dim=2
                    )  # Crop time_frames

                loss_fm += F.l1_loss(real_feat, fake_feat)

        fake_wav_trimmed = fake_wav[:, :, : real_wav.size(2)]
        mel_fake = mel_transform(fake_wav_trimmed.squeeze(1))
        loss_mel = F.l1_loss(mel_spec, mel_fake)

        loss_g = (
            loss_adv + (config.lambda_fm * loss_fm) + (config.lambda_mel * loss_mel)
        )
        loss_g.backward()
        optim_g.step()

        scheduler_g.step()
        scheduler_d.step()

        if steps % config.steps_per_log == 0:
            print(
                f"Step {steps}/{config.total_steps}: D Loss: {loss_d.item():.4f}, G Loss: {loss_g.item():.4f} (Adv: {loss_adv.item():.4f}, Mel: {loss_mel.item():.4f}, FM: {loss_fm.item():.4f})"
            )

        if steps % config.steps_per_checkpoint == 0 and steps > 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{steps}.pth"
            torch.save(
                {
                    "steps": steps,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optim_g": optim_g.state_dict(),
                    "optim_d": optim_d.state_dict(),
                    "scheduler_g": scheduler_g.state_dict(),
                    "scheduler_d": scheduler_d.state_dict(),
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path}")

        steps += 1

    final_model_path = checkpoint_dir / "generator_final.pth"
    torch.save(generator.state_dict(), final_model_path)
    return generator.cpu()


def generate_samples(
    generator_weights_path, config, t2s, test_sentences_path, output_dir
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(config).to(device)
    print(f"Loading generator weights from: {generator_weights_path}")
    generator.load_state_dict(torch.load(generator_weights_path, map_location=device))
    generator.remove_weight_norm()
    generator.eval()

    output_dir.mkdir(exist_ok=True)
    with open(test_sentences_path, "r") as f:
        sentences = f.readlines()

    with torch.no_grad():
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            print(f"Generating for: '{sentence}'")
            mel_spec_np = t2s.text2spec(sentence)
            mel_spec = torch.from_numpy(mel_spec_np).unsqueeze(0).to(device)
            mel_spec = mel_spec.transpose(1, 2)
            generated_wav = generator(mel_spec).squeeze().cpu()
            output_path = output_dir / f"generated_sample_{i+1}.wav"
            torchaudio.save(
                str(output_path), generated_wav.unsqueeze(0), config.sample_rate
            )
            print(f"Saved generated audio to {output_path}")


if __name__ == "__main__":
    t2s = TextToSpecConverter()
    config = VocoderConfig(t2s.config)
    ljsspeech_path = Path("./")
    output_dir = Path("./generated_audio_hifigan")
    test_sentences_file = Path("test_sentences.txt")
    dataset = LJSPEECH(root=str(ljsspeech_path), download=False)

    train(config, dataset)

    # resume_path = "./checkpoints_hifigan/checkpoint_20000.pth"
    # train(config, dataset, resume_from_checkpoint=resume_path)

    final_weights = "./checkpoints_hifigan/generator_final.pth"
    if Path(final_weights).exists():
        generate_samples(final_weights, config, t2s, test_sentences_file, output_dir)
