"""Synthetic signals, source directions and array geometries.

Used by examples.py and by the tests. All sources are free-field plane waves,
which is the model the paper assumes in Eq. (1) and the model both encoders
are built on. There are no rooms, reverberation or HRTFs here.
"""

import torch


SPEED_OF_SOUND = 343.0


def direction_from_angles(azimuth_degrees: float,
                          elevation_degrees: float = 0.0,
                          dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor: # [3]
    """
    Unit vector pointing at a source, using the paper's angle convention.

    Args:
        azimuth_degrees: Angle in the horizontal plane, 0 along +x, 90 along +y.
        elevation_degrees: Angle above the horizontal plane.
        dtype: Floating-point dtype of the result.

    Returns:
        direction: [3]
            Cartesian [x, y, z], unit length.

    This is the [cos(phi)cos(theta), sin(phi)cos(theta), sin(theta)] of
    Eq. (6), which is the XYZ part of the ideal SN3D steering vector. A
    planar array cannot recover elevation, so most callers leave it at zero.
    """
    azimuth = torch.tensor(azimuth_degrees * torch.pi / 180, dtype=dtype)
    elevation = torch.tensor(elevation_degrees * torch.pi / 180, dtype=dtype)

    return torch.stack(
        [
            azimuth.cos() * elevation.cos(),
            azimuth.sin() * elevation.cos(),
            elevation.sin(),
        ]
    )


def angles_from_direction(direction: torch.Tensor, # [..., 3]
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read azimuth and elevation, in degrees, back out of a direction vector.

    Args:
        direction: [..., 3], need not be unit length.

    Returns:
        azimuth, elevation: [...], both in degrees.

    Used by the plots, because degrees are easier to read on an axis than
    Cartesian components.
    """
    x, y, z = direction[..., 0], direction[..., 1], direction[..., 2]

    azimuth = torch.rad2deg(torch.atan2(y, x))
    elevation = torch.rad2deg(torch.atan2(z, torch.sqrt(x.square() + y.square())))

    return azimuth, elevation


def steering_vector(mics: torch.Tensor, # [C, 3]
                    direction: torch.Tensor, # [3]
                    freqs: torch.Tensor, # [F]
                    speed_of_sound: float = SPEED_OF_SOUND,
    ) -> torch.Tensor: # [F, C]
    """
    What each microphone measures for a unit plane wave from `direction`.

    Args:
        mics: [C, 3]
            Microphone coordinates in meters. Centred internally, so the
            origin the caller measured from does not matter.

        direction: [3]
            Unit vector pointing at the source.

        freqs: [F]
            Frequencies in Hz.

        speed_of_sound:
            In meters per second.

    Returns:
        steering: [F, C], complex
            The magnitude is 1 everywhere. For omnidirectional microphones in
            a free field, all the direction information is in the phase.

    A microphone closer to the source hears the wave earlier, so its phase
    leads. That is the + in the exponent below. The paper writes the same
    thing with two minus signs in Eq. (7), because it measures a distance
    from a reference microphone instead of a projection onto a direction.

    plane_wave() below calls this function, so the two cannot end up using
    different sign conventions.
    """
    centred = mics - mics.mean(dim=0, keepdim=True)

    # wavenumber, the same k = 2*pi*f / c the encoder uses
    k = 2 * torch.pi * freqs / speed_of_sound # [F]

    # how far along the direction of travel each microphone sits
    projections = centred @ direction # [C]

    return torch.exp(1j * k[:, None] * projections[None, :])


def plane_wave(mics: torch.Tensor, # [C, 3]
               direction: torch.Tensor, # [3]
               signal: torch.Tensor, # [T]
               sample_rate: int = 24000,
               speed_of_sound: float = SPEED_OF_SOUND,
    ) -> torch.Tensor: # [1, C, T]
    """
    Record one signal on an array as a plane wave from `direction`.

    Args:
        mics: [C, 3], microphone coordinates in meters.
        direction: [3], unit vector pointing at the source.
        signal: [T], the waveform the source emits.
        sample_rate: Samples per second.
        speed_of_sound: In meters per second.

    Returns:
        audio: [1, C, T]
            Ready to hand to FoaConverter.convert.

    The delay is applied as a phase ramp over the signal's FFT bins, not by
    shifting whole samples. Across a 10 cm array at 24 kHz the delays are a
    couple of samples at most, so rounding them to integers would add a
    direction error of several degrees to the generated signal itself.
    """
    n_samples = signal.shape[-1]

    spectrum = torch.fft.rfft(signal) # [F']
    freqs = torch.fft.rfftfreq(n_samples, 1 / sample_rate, device=signal.device)

    # [F', C] -> [C, F'] so the transform runs along the last axis
    steering = steering_vector(mics, direction, freqs, speed_of_sound).T

    delayed = torch.fft.irfft(spectrum[None, :] * steering, n=n_samples)

    return delayed.unsqueeze(0)


def ideal_foa(direction: torch.Tensor, # [3]
              signal: torch.Tensor, # [T]
    ) -> torch.Tensor: # [1, 4, T]
    """
    Encode one signal directly into FOA, skipping the array entirely.

    Args:
        direction: [3], unit vector pointing at the source.
        signal: [T], the waveform the source emits.

    Returns:
        foa: [1, 4, T]
            Channels [W, X, Y, Z], SN3D.

    This is Eq. (6) applied directly: W carries the pressure, and XYZ carry
    the same signal scaled by the direction cosines. There is no array, so
    there is no spatial aliasing and no low-frequency droop.

    Useful for testing FoaSalsa on its own. If a feature is wrong on input
    built this way, the problem is in the feature extractor rather than in
    FoaConverter.
    """
    channels = [signal, *(component * signal for component in direction)]

    return torch.stack(channels).unsqueeze(0)


def doa_error_degrees(estimate: torch.Tensor, # [..., 3]
                      truth: torch.Tensor, # [..., 3]
                      eps: float = 1e-12,
    ) -> torch.Tensor: # [...]
    """
    Angle between two directions, in degrees.

    Args:
        estimate, truth: [..., 3], broadcastable. Need not be unit length.
        eps: Guard for zero-length inputs.

    Returns:
        error: [...], in degrees.

    This is the quantity SALSA depends on. Its EIV normalises the direction
    vector to unit length, so a gain error shared by all three channels does
    not affect it and only the pointing error matters.
    """
    estimate = estimate / torch.linalg.vector_norm(
        estimate, dim=-1, keepdim=True
    ).clamp(min=eps)
    truth = truth / torch.linalg.vector_norm(truth, dim=-1, keepdim=True).clamp(min=eps)

    cosine = (estimate * truth).sum(dim=-1).clamp(-1.0, 1.0)

    return torch.rad2deg(torch.arccos(cosine))


def regular_polygon(n_mics: int,
                    radius: float = 0.05,
                    dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor: # [C, 3]
    """
    Microphones evenly spaced around a circle in the z = 0 plane.

    Args:
        n_mics: Number of microphones.
        radius: Distance from the centre, in meters.
        dtype: Floating-point dtype of the result.

    Returns:
        mics: [n_mics, 3]

    n_mics=4 gives the square used in the tests. All microphones lie in the
    z = 0 plane, so anything encoded from this array has an empty Z channel.
    """
    if n_mics <= 0:
        raise ValueError("n_mics must be positive.")

    if radius <= 0:
        raise ValueError("radius must be positive.")

    angles = torch.arange(n_mics, dtype=dtype) * (2 * torch.pi / n_mics)

    return torch.stack(
        [radius * angles.cos(), radius * angles.sin(), torch.zeros_like(angles)],
        dim=-1,
    )


def random_planar(n_mics: int,
                  radius: float = 0.06,
                  seed: int = 0,
                  dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor: # [C, 3]
    """
    Microphones scattered inside a disc in the z = 0 plane.

    Args:
        n_mics: Number of microphones.
        radius: Radius of the disc, in meters.
        seed: Fixed so the examples redraw identically.
        dtype: Floating-point dtype of the result.

    Returns:
        mics: [n_mics, 3]

    Deliberately irregular. The least-squares encoder only beats the gradient
    one when the array has more microphones than unknowns and no symmetry. On
    a square or a hexagon the two encoders give identical directions and
    differ only by a common gain, which normalising removes.
    """
    if n_mics <= 0:
        raise ValueError("n_mics must be positive.")

    if radius <= 0:
        raise ValueError("radius must be positive.")

    generator = torch.Generator().manual_seed(seed)

    # taking the square root of the radius spreads the points evenly over
    # the disc instead of concentrating them near the centre
    angles = torch.rand(n_mics, generator=generator, dtype=dtype) * 2 * torch.pi
    radii = radius * torch.rand(n_mics, generator=generator, dtype=dtype).sqrt()

    return torch.stack(
        [radii * angles.cos(), radii * angles.sin(), torch.zeros_like(radii)],
        dim=-1,
    )


def tone(hz: float,
         n_samples: int,
         sample_rate: int = 24000,
    ) -> torch.Tensor: # [T]
    """A single sine wave at one frequency."""
    t = torch.arange(n_samples) / sample_rate

    return torch.sin(2 * torch.pi * hz * t)


def chirp(start_hz: float,
          end_hz: float,
          n_samples: int,
          sample_rate: int = 24000,
    ) -> torch.Tensor: # [T]
    """
    A sine sweeping linearly from start_hz to end_hz.

    Sweeping one source across the band shows where an array stops working.
    The direction is accurate at low frequency and breaks down above about
    c / (2 * spacing).
    """
    t = torch.arange(n_samples) / sample_rate
    duration = n_samples / sample_rate

    # the phase is the integral of the instantaneous frequency
    rate = (end_hz - start_hz) / duration
    phase = 2 * torch.pi * (start_hz * t + 0.5 * rate * t.square())

    return torch.sin(phase)


def white_noise(n_samples: int,
                seed: int = 0,
    ) -> torch.Tensor: # [T]
    """Gaussian noise, seeded so plots redraw identically."""
    generator = torch.Generator().manual_seed(seed)

    return torch.randn(n_samples, generator=generator)


def band_noise(low_hz: float,
               high_hz: float,
               n_samples: int,
               sample_rate: int = 24000,
               seed: int = 0,
    ) -> torch.Tensor: # [T]
    """
    Noise confined to one frequency band.

    Args:
        low_hz, high_hz: Edges of the band.
        n_samples: Length in samples.
        sample_rate: Samples per second.
        seed: Fixed so plots redraw identically.

    Returns:
        signal: [T]

    Two sources in separate bands show what SALSA is for. They overlap
    completely in time, so a frame-level feature must report a single
    direction for both. They never share a frequency bin, so a feature with a
    frequency axis can report both directions at the same time.
    """
    if not 0 <= low_hz < high_hz:
        raise ValueError(f"need 0 <= low_hz < high_hz, got {low_hz} and {high_hz}")

    generator = torch.Generator().manual_seed(seed)

    spectrum = torch.fft.rfft(torch.randn(n_samples, generator=generator))
    freqs = torch.fft.rfftfreq(n_samples, 1 / sample_rate)

    spectrum = spectrum * ((freqs >= low_hz) & (freqs <= high_hz))

    signal = torch.fft.irfft(spectrum, n=n_samples)

    # filtering removes most of the energy, so rescale to unit standard
    # deviation. Without this, two sources in different bands would not be
    # equally loud
    return signal / signal.std().clamp(min=1e-12)
