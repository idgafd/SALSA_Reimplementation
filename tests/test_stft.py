import pytest
import torch

from foasalsa.stft import StftConfig, istft, stft


# a fixed seed keeps failures reproducible
torch.manual_seed(0)


@pytest.fixture
def config():
    return StftConfig()


def make_audio(n_batch=2, n_channels=4, n_samples=12000, dtype=torch.float32):
    """Random multichannel audio, the [N, C, T] shape the package works with."""
    return torch.randn(n_batch, n_channels, n_samples, dtype=dtype)


class TestStftConfig:
    def test_defaults_match_the_paper(self, config):
        # SALSA paper, Section V.C
        assert config.sample_rate == 24000
        assert config.n_fft == 512
        assert config.win_length == 512
        assert config.hop_length == 300
        # the paper reports 80 fps for these settings, which checks the rest
        assert config.frame_rate == 80.0

    def test_n_freqs(self, config):
        assert config.n_freqs == 257

    def test_frequencies_span_zero_to_nyquist(self, config):
        freqs = config.frequencies()

        assert freqs.shape == (config.n_freqs,)
        assert freqs[0] == 0.0
        assert freqs[-1] == config.sample_rate / 2

    def test_frequencies_follow_requested_device_and_dtype(self, config):
        freqs = config.frequencies(dtype=torch.float64)

        assert freqs.dtype == torch.float64

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sample_rate": 0},
            {"n_fft": 511},  # odd
            {"n_fft": -512},
            {"win_length": 1024},  # longer than n_fft
            {"win_length": 0},
            {"hop_length": 0},
            {"hop_length": 512},  # frames would not overlap, so NOLA fails
            {"hop_length": 600},
        ],
    )
    def test_rejects_broken_settings(self, kwargs):
        with pytest.raises(ValueError):
            StftConfig(**kwargs)

    def test_is_frozen(self, config):
        with pytest.raises(Exception):
            config.n_fft = 1024


class TestShapesAndDtypes:
    def test_stft_shape(self, config):
        audio = make_audio(n_batch=2, n_channels=4, n_samples=12000)

        spec = stft(audio, config)

        expected_frames = 1 + 12000 // config.hop_length
        assert spec.shape == (2, 4, config.n_freqs, expected_frames)

    def test_istft_gives_back_the_requested_length(self, config):
        audio = make_audio(n_samples=12000)

        restored = istft(stft(audio, config), config, length=12000)

        assert restored.shape == audio.shape

    def test_float32_audio_gives_complex64_spectrogram(self, config):
        audio = make_audio(dtype=torch.float32)

        spec = stft(audio, config)

        assert spec.dtype == torch.complex64
        assert istft(spec, config, length=audio.shape[-1]).dtype == torch.float32

    def test_float64_audio_gives_complex128_spectrogram(self, config):
        audio = make_audio(dtype=torch.float64)

        spec = stft(audio, config)

        assert spec.dtype == torch.complex128
        assert istft(spec, config, length=audio.shape[-1]).dtype == torch.float64

    def test_handles_a_single_item_batch(self, config):
        audio = make_audio(n_batch=1, n_channels=1, n_samples=4000)

        spec = stft(audio, config)

        assert spec.shape[:2] == (1, 1)


class TestRoundTrip:
    """512/300 is neither 50% nor 75% overlap, so exact reconstruction is
    worth checking rather than assuming.

    It works because torch.istft normalises the overlap-add by dividing by
    the sum of squared windows. That only requires the NOLA condition, that
    the sum stays non-zero, and not the stricter COLA condition.
    """

    def test_reconstructs_the_interior_of_the_signal(self, config):
        audio = make_audio(n_samples=12000)

        restored = istft(stft(audio, config), config, length=audio.shape[-1])

        # center=True pads the signal by reflection, which affects the first
        # and last n_fft samples, so compare away from the edges
        edge = config.n_fft
        interior = slice(edge, -edge)
        assert torch.allclose(restored[..., interior], audio[..., interior], atol=1e-5)

    def test_edges_are_close_too_just_less_exact(self, config):
        audio = make_audio(n_samples=12000)

        restored = istft(stft(audio, config), config, length=audio.shape[-1])

        assert torch.allclose(restored, audio, atol=1e-3)

    def test_survives_a_length_that_is_not_a_multiple_of_the_hop(self, config):
        audio = make_audio(n_samples=12345)

        restored = istft(stft(audio, config), config, length=12345)

        edge = config.n_fft
        interior = slice(edge, -edge)
        assert restored.shape == audio.shape
        assert torch.allclose(restored[..., interior], audio[..., interior], atol=1e-5)


class TestBatchIndependence:
    """Flattening [N, C, T] to [N*C, T] and back is the only place in this
    module where the axes can end up in the wrong order. A wrong reshape
    still produces numbers that look reasonable, so it needs its own test.
    """

    def test_each_channel_matches_its_own_transform(self, config):
        audio = make_audio(n_batch=3, n_channels=4, n_samples=6000)

        batched = stft(audio, config)

        for n in range(3):
            for c in range(4):
                alone = stft(audio[n : n + 1, c : c + 1], config)
                assert torch.allclose(batched[n, c], alone[0, 0], atol=1e-6)

    def test_channels_are_not_swapped(self, config):
        # two channels that are easy to tell apart, silence and noise
        audio = torch.zeros(1, 2, 6000)
        audio[0, 1] = torch.randn(6000)

        spec = stft(audio, config)

        assert spec[0, 0].abs().max() == 0.0
        assert spec[0, 1].abs().max() > 0.0

    def test_istft_keeps_channels_apart(self, config):
        audio = make_audio(n_batch=3, n_channels=4, n_samples=6000)

        restored = istft(stft(audio, config), config, length=6000)

        for n in range(3):
            for c in range(4):
                alone = istft(
                    stft(audio[n : n + 1, c : c + 1], config), config, length=6000
                )
                assert torch.allclose(restored[n, c], alone[0, 0], atol=1e-5)


class TestKnownSignals:
    def test_a_pure_tone_lands_in_its_own_bin(self, config):
        # a frequency exactly on a bin centre, so there is no leakage into
        # neighbouring bins
        bin_index = 50
        freq = config.frequencies()[bin_index].item()
        t = torch.arange(12000) / config.sample_rate
        audio = torch.sin(2 * torch.pi * freq * t).reshape(1, 1, -1)

        spec = stft(audio, config)

        # average over frames so the quieter edge frames do not matter
        energy_per_bin = spec.abs().mean(dim=-1)[0, 0]
        assert energy_per_bin.argmax().item() == bin_index

    def test_silence_stays_silent(self, config):
        audio = torch.zeros(1, 4, 6000)

        spec = stft(audio, config)

        assert torch.all(spec.abs() == 0.0)
