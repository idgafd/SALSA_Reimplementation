import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class StftConfig:
    """STFT settings shared by FoaConverter and FoaSalsa.

    Defaults come from the SALSA paper, Section V.C (Hyperparameters):
    "We used a sampling rate of 24 kHz, window length of 512 samples, hop
    length of 300 samples, Hann window, 512 FFT points". Those settings are
    what give the 80 fps frame rate reported in the same section.

    The window is always Hann, as in the paper, so it is not a parameter.

    Frozen because FoaConverter caches an encoding matrix built for these
    exact frequencies; mutating the config afterwards would silently
    desynchronise the two.
    """

    # samples per second; Nyquist limit = 12 kHz
    sample_rate: int = 24000
    # n of points for each frame -> 257 bins spaced by ~47 Hz
    n_fft: int = 512
    # n of audio samples seen by one STFT frame -> ~21 ms
    win_length: int = 512
    # n of samples between frame -> 12.5 ms hop -> 80 frames/s
    hop_length: int = 300
    # pad the signal so frame t is centred on sample t * hop_length;
    # keeps the feature time axis aligned with the waveform
    center: bool = True
    # how that padding is filled; only affects the first and last frames
    pad_mode: str = "reflect"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.n_fft <= 0 or self.n_fft % 2 != 0:
            raise ValueError(f"n_fft must be positive and even, got {self.n_fft}")
        if not 0 < self.win_length <= self.n_fft:
            raise ValueError(
                f"win_length must be in (0, n_fft={self.n_fft}], got {self.win_length}"
            )
        # frames must overlap, otherwise the squared window sum that istft
        # divides by hits zero at the frame boundaries (the NOLA condition)
        if not 0 < self.hop_length < self.win_length:
            raise ValueError(
                f"hop_length must be in (0, win_length={self.win_length}), "
                f"got {self.hop_length}"
            )

    @property
    def n_freqs(self) -> int:
        """Number of one-sided frequency bins, F. 257 for the paper's defaults."""
        return self.n_fft // 2 + 1

    @property
    def frame_rate(self) -> float:
        """STFT frames per second. 80 fps for the paper's defaults."""
        return self.sample_rate / self.hop_length

    def frequencies(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Centre frequency of every bin, [F], in Hz.

        Needed by both consumers: FoaConverter turns these into wavenumbers
        k = 2 * pi * f / c, and FoaSalsa uses them for the 50 Hz - 9 kHz band
        limits of the EIV channels.
        """
        return torch.fft.rfftfreq(
            self.n_fft,
            d=1.0 / self.sample_rate,
            device=device,
            dtype=dtype,
        )


def stft(audio: torch.Tensor, config: StftConfig) -> torch.Tensor:
    """
    Args:
        audio: [N, C, T]
            N - batch size
            C - number of audio / microphone channels
            T - number of time-domain samples

    Returns:
        stft_audio: [N, C, F, T_frames], complex
            F - number of frequency bins
                       for real-valued audio: F = n_fft // 2 + 1
            T_frames - number of STFT time frames
    
    We flatten [N, C, T] -> [N*C, T] because torch.stft processes
    a batch of 1D signals in shape [B, T] and reshape it back to
    [N, C, F, T_frames] after the STFT.
    """
    N, C, T = audio.shape

    # every channel of every batch is an independent signal
    audio_flat = audio.reshape(N * C, T)

    # Hann window of shape [win_length] smooths out the edges
    # of each frame to reduce spectral leakage
    window = torch.hann_window(
        config.win_length,
        periodic=True,
        device=audio.device,
        dtype=audio.dtype,
    )

    # [N*C, T] -> [N*C, F, T_frames], complex
    stft_audio_flat = torch.stft(
        audio_flat,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        pad_mode=config.pad_mode,
        # keep the raw FFT scale: log(|X|^2) is a feature, so it must not
        # silently pick up a 1/sqrt(n_fft) factor
        normalized=False,
        return_complex=True,
    )

    F, T_frames = stft_audio_flat.shape[-2:]

    # [N*C, F, T_frames] -> [N, C, F, T_frames]
    return stft_audio_flat.reshape(N, C, F, T_frames)


def istft(stft_audio: torch.Tensor, config: StftConfig, length: int) -> torch.Tensor:
    """
    Args:
        stft_audio: [N, C, F, T_frames], complex
            N - batch size
            C - number of audio / microphone channels
            F - number of frequency bins
            T_frames - number of STFT time frames

    Returns:
        audio: [N, C, T]
            T - number of time-domain samples
    
    Same flatteting logic as in stft() function.
    """
    N, C, F, T_frames = stft_audio.shape

    # every channel of every batch is an independent signal
    stft_audio_flat = stft_audio.reshape(N * C, F, T_frames)

    # Hann window must be the same as in stft() to reconstruct the original signal
    window = torch.hann_window(
        config.win_length,
        periodic=True,
        device=stft_audio.device,
        dtype=stft_audio.real.dtype, # must be real for Hann window
    )

    # [N*C, F, T_frames] -> [N*C, T]
    audio_flat = torch.istft(
        stft_audio_flat,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=config.center,
        length=length,
    )

    # [N*C, T] -> [N, C, T]
    return audio_flat.reshape(N, C, length)