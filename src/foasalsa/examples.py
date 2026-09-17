"""Synthetic examples showing what SALSA does and where it stops working.

Run with:

    python -m foasalsa.examples <output_dir>

Figures 1 to 5 record sources on a planar microphone array and go through
FoaConverter, so they can only ever show azimuth. Figures 6 and 7 skip the
array and encode straight to FOA with Eq. (6), which isolates the feature
extractor from anything the geometry does.

    1-3  the pipeline works, and where it stops working
    4-5  the encoder: why 1/(jk) is needed, and which encoder to pick
    6-7  the feature: what it is for, and why it uses an eigenvector

By default the figures land in plots/ at the top of the repository, and the
README walks through them in this order.
"""

import sys
from pathlib import Path

import torch

from .encoding import FoaConverter, _gradient_matrix, _least_squares_matrix
from .encoding import _fibonacci_sphere
from .salsa import FoaSalsa, _covariance, _normalize_eigenvector
from .salsa import _principal_eigenvector, intensity_vector
from .stft import StftConfig, stft
from .synth import (
    SPEED_OF_SOUND,
    band_noise,
    diffuse_field,
    direction_from_angles,
    doa_error_degrees,
    ideal_foa,
    plane_wave,
    random_planar,
    regular_polygon,
    tone,
    white_noise,
)

try:
    import matplotlib.pyplot as plt

    from .plotting import EIV_COLORMAP, as_array, mark_frequency, save, time_frequency
except ImportError as error: # pragma: no cover - only hit without matplotlib
    raise SystemExit(
        "The examples need matplotlib, which is an optional dependency.\n"
        "Install it with:  pip install 'foasalsa[examples]'"
    ) from error


CONFIG = StftConfig()

# 5 cm radius, so 10 cm between opposite microphones
SQUARE = regular_polygon(4, radius=0.05)

SECOND = CONFIG.sample_rate


def estimated_direction(features, band=None):
    """
    Average the EIV over every bin the single-source tests kept.

    Args:
        features: [1, 7, F, T_frames] from FoaSalsa.compute.
        band: optional [F] boolean mask restricting which frequencies count.

    Returns:
        direction: [3], not normalised.

    The length of the result says how much the bins agreed with each other.
    Unit vectors all pointing the same way average to length 1; unit vectors
    pointing at random average to nearly zero.

    The band argument matters whenever the source does not fill the spectrum.
    A pure tone excites a handful of bins, and the rest of the passband holds
    only whatever noise survived the tests of Section II.D, so averaging over
    everything would drown the answer.

    Returns zeros when nothing was kept, which is itself a result worth
    plotting.
    """
    eiv = features[0, 4:] # [3, F, T_frames]

    if band is not None:
        eiv = eiv[:, band]

    live = eiv.abs().sum(dim=0) > 0

    if not live.any():
        return torch.zeros(3)

    return eiv[:, live].mean(dim=-1)


def pipeline_direction(mics, azimuth, signal, band=None, converter=None, salsa=None):
    """Run one source through the whole package and read the direction back."""
    converter = converter or FoaConverter(config=CONFIG)
    salsa = salsa or FoaSalsa(config=CONFIG)

    audio = plane_wave(mics, direction_from_angles(azimuth), signal)

    return estimated_direction(
        salsa.compute(converter.convert(audio, mics)), band=band
    )


def bins_around(hz, width=150.0):
    """Boolean mask over the frequency axis, centred on one tone."""
    freqs = CONFIG.frequencies()

    return (freqs - hz).abs() < max(width, 0.1 * hz)


def example_01_azimuth_sweep(directory):
    """Does the pipeline recover the direction it was given?

    A single tone is played from every azimuth in turn, recorded on the
    square array, encoded and turned into features. The averaged EIV should
    come back as [cos(azimuth), sin(azimuth), 0].

    At 500 Hz the wavelength is 69 cm against a 10 cm array, which is well
    inside the range where the encoder is accurate, so this is the baseline
    that the later examples break on purpose.
    """
    azimuths = torch.arange(0, 360, 10.0)
    signal = tone(500.0, SECOND)

    band = bins_around(500.0)

    estimates = torch.stack(
        [pipeline_direction(SQUARE, float(a), signal, band=band) for a in azimuths]
    )
    estimates = estimates / estimates.norm(dim=-1, keepdim=True)

    recovered = torch.rad2deg(torch.atan2(estimates[:, 1], estimates[:, 0])) % 360

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")

    left.plot(as_array(azimuths), as_array(recovered), "o", markersize=4, label="recovered")
    left.plot([0, 350], [0, 350], "--", color="grey", linewidth=1, label="exact")
    left.set_xlabel("true azimuth (degrees)")
    left.set_ylabel("recovered azimuth (degrees)")
    left.set_title("Recovered azimuth, 500 Hz tone")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    right.plot(as_array(azimuths), as_array(estimates[:, 0]), "o", markersize=4, label="EIV X")
    right.plot(as_array(azimuths), as_array(estimates[:, 1]), "s", markersize=4, label="EIV Y")
    radians = torch.deg2rad(azimuths)
    right.plot(as_array(azimuths), as_array(radians.cos()), "-", color="grey", linewidth=1, label="cos, sin")
    right.plot(as_array(azimuths), as_array(radians.sin()), "-", color="grey", linewidth=1)
    right.set_xlabel("true azimuth (degrees)")
    right.set_ylabel("EIV component")
    right.set_title("EIV against the SN3D steering vector of Eq. (6)")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)

    fig.suptitle(
        "1. The pipeline recovers azimuth: plane wave -> FoaConverter -> FoaSalsa",
        fontsize=11,
    )

    worst = max(
        abs(((r - a + 180) % 360) - 180) for r, a in zip(recovered.tolist(), azimuths.tolist())
    )

    return save(fig, directory, "01_azimuth_sweep.png"), f"worst error {worst:.2f} deg"


def example_02_planar_array_has_no_elevation(directory):
    """The first limitation, and it is a property of the array, not the code.

    Every microphone sits in the z = 0 plane, so the third column of the
    geometry matrix is zero and its pseudoinverse has an all-zero third row.
    The Z channel of the encoder output is therefore exactly zero, and so is
    the Z channel of the feature.

    Two consequences are plotted. The elevation of a source is not recovered
    at all, and two sources mirrored about the horizontal plane produce
    byte-identical features.
    """
    elevations = torch.arange(-60, 61, 10.0)
    signal = tone(500.0, SECOND)
    converter = FoaConverter(config=CONFIG)
    salsa = FoaSalsa(config=CONFIG)

    band = bins_around(500.0)
    azimuth = 30.0 # not 45, so that X and Y are different numbers on the plot

    measured = []
    for elevation in elevations:
        direction = direction_from_angles(azimuth, float(elevation))
        audio = plane_wave(SQUARE, direction, signal)
        features = salsa.compute(converter.convert(audio, SQUARE))
        measured.append(estimated_direction(features, band=band))

    measured = torch.stack(measured)
    measured = measured / measured.norm(dim=-1, keepdim=True).clamp(min=1e-12)

    truth = torch.stack(
        [direction_from_angles(azimuth, float(e)) for e in elevations]
    )

    recovered_azimuth = torch.rad2deg(torch.atan2(measured[:, 1], measured[:, 0]))

    # the mirror test: the same azimuth, elevation above and below the plane
    above = salsa.compute(
        converter.convert(
            plane_wave(SQUARE, direction_from_angles(azimuth, 30.0), signal), SQUARE
        )
    )
    below = salsa.compute(
        converter.convert(
            plane_wave(SQUARE, direction_from_angles(azimuth, -30.0), signal), SQUARE
        )
    )
    mirror_gap = (above - below).abs().max().item()

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")

    for index, name in enumerate("XYZ"):
        line = left.plot(as_array(elevations), as_array(measured[:, index]), "o-", markersize=4,
                         label=f"measured {name}")[0]
        left.plot(as_array(elevations), as_array(truth[:, index]), "--", linewidth=1,
                  color=line.get_color(), label=f"true {name}")
    left.set_xlabel("true elevation (degrees)")
    left.set_ylabel("EIV component")
    left.set_title("Z stays at zero; X and Y hold the azimuth")
    left.legend(fontsize=7, ncol=2)
    left.grid(alpha=0.3)

    right.plot(as_array(elevations), as_array(recovered_azimuth), "o-", markersize=4,
               label="recovered azimuth")
    right.axhline(azimuth, color="grey", linestyle="--", linewidth=1,
                  label="true azimuth")
    right.plot(as_array(elevations), as_array(torch.zeros_like(elevations)), "s-", markersize=4,
               label="recovered elevation")
    right.plot(as_array(elevations), as_array(elevations), "--", color="tab:green", linewidth=1,
               label="true elevation")
    right.set_xlabel("true elevation (degrees)")
    right.set_ylabel("recovered angle (degrees)")
    right.set_title("Azimuth survives, elevation is lost entirely")
    right.legend(fontsize=7)
    right.grid(alpha=0.3)
    right.annotate(
        f"features at +30 and -30 elevation\ndiffer by {mirror_gap:.0e}: "
        "the array\ncannot tell them apart",
        xy=(0.03, 0.05), xycoords="axes fraction", fontsize=7,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.suptitle(
        "2. A planar array cannot measure elevation, and says so with an empty Z channel",
        fontsize=11,
    )

    return (
        save(fig, directory, "02_planar_array_no_elevation.png"),
        f"Z channel max {measured[:, 2].abs().max():.1e}, mirror gap {mirror_gap:.1e}",
    )


def example_03_spatial_aliasing(directory):
    """The second limitation: an upper frequency limit set by the spacing.

    Both encoders assume the array is small compared with the wavelength.
    Once half a wavelength fits between the outer microphones, their phase
    difference stops identifying a direction, and the estimate falls apart.
    The limit is c / (2 * spacing).

    The other half of the trade-off only appears once the microphones have
    noise of their own. A wider array sees larger phase differences between
    its microphones, so at low frequency, where those differences are small,
    it is the more accurate of the two. Below about 400 Hz here the 24 cm
    array wins; above it the 8 cm array does.

    So the spacing sets both ends of the usable range, and no encoder can
    move either one. This is why the paper band-limits the EIV to 9 kHz in
    Section V.A: that is the aliasing frequency of the array it used.
    """
    arrays = {
        "8 cm across": regular_polygon(4, radius=0.04),
        "24 cm across": regular_polygon(4, radius=0.12),
    }
    frequencies = torch.logspace(
        torch.log10(torch.tensor(60.0)), torch.log10(torch.tensor(9000.0)), 26
    )
    azimuths = [11.0, 37.0, 73.0]

    # sensor noise, at a fixed level against a unit-amplitude tone. Without it
    # the wider array has no visible advantage anywhere and the figure would
    # show only half of the trade-off.
    noise_level = 0.05
    generator = torch.Generator().manual_seed(0)

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")

    converter = FoaConverter(config=CONFIG)
    salsa = FoaSalsa(config=CONFIG)

    for row, (label, mics) in enumerate(arrays.items()):
        errors = []
        for hz in frequencies:
            band = bins_around(float(hz))
            signal = tone(float(hz), SECOND)

            per_azimuth = []
            for azimuth in azimuths:
                truth = direction_from_angles(azimuth)
                audio = plane_wave(mics, truth, signal)
                audio = audio + noise_level * torch.randn(
                    audio.shape, generator=generator
                )

                features = salsa.compute(converter.convert(audio, mics))
                estimate = estimated_direction(features, band=band)
                per_azimuth.append(doa_error_degrees(estimate, truth).item())

            errors.append(sum(per_azimuth) / len(per_azimuth))

        line = ax.plot(as_array(frequencies), errors, "o-", markersize=3, label=label)[0]
        line.set_zorder(3)

        spacing = 2 * mics.norm(dim=-1).max().item() # opposite microphones
        aliasing = SPEED_OF_SOUND / (2 * spacing)
        mark_frequency(ax, aliasing, f"{label}: c/(2d) = {aliasing:.0f} Hz", row=row)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("direction error, averaged over 3 azimuths (degrees)")
    ax.set_title(
        "3. Microphone spacing sets both ends of the usable band\n"
        f"identical sensor noise on every microphone (std {noise_level})",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    return (
        save(fig, directory, "03_spatial_aliasing.png"),
        "wider array wins below ~400 Hz, narrower array holds on far higher",
    )


def example_04_gradient_compensation(directory):
    """Why FoaConverter divides by j*k, and what happens if it does not.

    Omnidirectional microphones measure pressure only, so the direction has
    to come from differences between them. A spatial difference measures
    grad(p), and for a plane wave grad(p) = -j*k*p. The result points the
    right way but is 90 degrees out of phase with W, and its magnitude rises
    with frequency.

    SALSA's EIV divides XYZ by W and keeps the real part. When that ratio is
    almost purely imaginary, the small real part left over is rounding error,
    and normalising it to unit length turns that error into a confident
    direction pointing nowhere in particular. Each bin then reports its own
    arbitrary direction.

    The right way to see this is not to count how many bins survive the two
    tests of Section II.D, because neither test looks at the real part. It is
    to ask how much the surviving bins agree with each other. Averaging unit
    vectors that all point the same way gives a vector of length 1; averaging
    unit vectors pointing at random gives a vector of length near 0.

    Note that the information is not destroyed, only moved somewhere the EIV
    does not look: the angle of X/W is a clean constant, 0 degrees with the
    compensation and -90 without. A feature reading the angle would see an
    offset it could subtract, which is what SALSA does for the MIC format in
    Section II.C.2. The failure belongs to the pairing of a quadrature
    gradient with a real-part reader, not to spatial features in general.
    """
    freqs = CONFIG.frequencies()
    truth = direction_from_angles(30.0)

    compensated = _gradient_matrix(SQUARE, freqs, compensate=True)
    raw = _gradient_matrix(SQUARE, freqs, compensate=False)

    # push the analytic steering vector through both, one frequency at a time
    def ratios(matrix):
        values = []
        for index, hz in enumerate(freqs):
            k = 2 * torch.pi * hz / SPEED_OF_SOUND
            steering = torch.exp(1j * k * (SQUARE @ truth)).to(matrix.dtype)
            out = matrix[index] @ steering
            values.append(out[1] / out[0] if out[0].abs() > 0 else torch.zeros(()))
        return torch.stack(values)

    with_fix = ratios(compensated)
    without_fix = ratios(raw)

    # this array aliases above c / (2 * 0.1) = 1715 Hz, so measure below that
    # and keep the example about the compensation rather than the geometry
    band = (freqs > 200.0) & (freqs < 1500.0)

    audio = plane_wave(SQUARE, truth, white_noise(SECOND, seed=3))
    salsa = FoaSalsa(config=CONFIG)

    agreement = {}
    for name, flag in (("with 1/(jk)", True), ("without", False)):
        converter = FoaConverter(config=CONFIG, compensate_gradient=flag)
        eiv = salsa.compute(converter.convert(audio, SQUARE))[0, 4:][:, band]
        live = eiv.abs().sum(dim=0) > 0
        agreement[name] = eiv[:, live].mean(dim=-1).norm().item()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")

    for ax, values, title in (
        (axes[0], with_fix, "with 1/(jk): X/W is real"),
        (axes[1], without_fix, "without: X/W is almost purely imaginary"),
    ):
        # slice first: leaving the aliased high end in the data would let
        # autoscale pick a range in which the plotted part is a flat line
        ax.plot(as_array(freqs[band]), as_array(values.real[band]), label="real part")
        ax.plot(as_array(freqs[band]), as_array(values.imag[band]), label="imaginary part")
        ax.axhline(truth[0].item(), color="grey", linestyle="--", linewidth=1,
                   label="cos(30 deg), the target")
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("X / W")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[2].bar(list(agreement), list(agreement.values()), width=0.5,
                color=["tab:blue", "tab:red"])
    axes[2].set_ylim(0, 1.15)
    axes[2].set_ylabel("length of the averaged EIV")
    axes[2].set_title("do the bins agree on a direction?", fontsize=10)
    axes[2].grid(alpha=0.3, axis="y")
    for index, value in enumerate(agreement.values()):
        axes[2].annotate(f"{value:.2f}", (index, value), ha="center",
                         va="bottom", fontsize=9)

    fig.suptitle(
        "4. EIV keeps only the real part of X/W, so an uncompensated gradient is unusable\n"
        "measured over 200-1500 Hz, below this array's aliasing limit",
        fontsize=11,
    )

    return (
        save(fig, directory, "04_gradient_compensation.png"),
        f"bin agreement {agreement['with 1/(jk)']:.2f} with, {agreement['without']:.2f} without",
    )


def example_05_gradient_versus_least_squares(directory):
    """Which encoder to use, and when the choice actually matters.

    FoaConverter offers two ways to build the same frequency-dependent
    matrix. "gradient" linearises exp(j*k*r) to 1 + j*k*r and inverts the
    geometry; "ls" fits the exact plane-wave response over a grid of
    directions, which is Eq. (2) of McCormack et al.

    On a symmetric array the two are indistinguishable. Symmetry means X can
    only be built from one opposing pair of microphones and Y from the other,
    so the encoders can differ by a single common gain, and normalising the
    direction removes it. The least-squares fit only helps when the array has
    more microphones than the three unknowns and no symmetry to exploit.

    In both cases the two agree at low frequency, because that is where the
    linearisation is accurate. The gradient encoder is the low-frequency
    limit of the least-squares one, not a different idea.
    """
    frequencies = torch.logspace(2, torch.log10(torch.tensor(6000.0)), 24)
    azimuths = torch.arange(0, 360, 10.0)

    arrays = {
        "symmetric square, 4 mics": regular_polygon(4, radius=0.05, dtype=torch.float64),
        "irregular, 8 mics": random_planar(8, radius=0.06, seed=1, dtype=torch.float64),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True, layout="constrained")

    for ax, (label, mics) in zip(axes, arrays.items()):
        grid = _fibonacci_sphere(642, mics.device, mics.dtype)
        # regularisation off, so the comparison is about the fit and not
        # about the low-frequency gain cap, which only the gradient mode has
        encoders = {
            "gradient": _gradient_matrix(
                mics, CONFIG.frequencies(dtype=torch.float64), regularize=False
            ),
            "least squares": _least_squares_matrix(
                mics, CONFIG.frequencies(dtype=torch.float64), SPEED_OF_SOUND,
                grid, beta=1e-6,
            ),
        }

        for name, matrix in encoders.items():
            errors = []
            for hz in frequencies:
                index = int(round(float(hz) / (CONFIG.sample_rate / CONFIG.n_fft)))
                k = 2 * torch.pi * CONFIG.frequencies(dtype=torch.float64)[index] / SPEED_OF_SOUND

                per_azimuth = []
                for azimuth in azimuths:
                    direction = direction_from_angles(
                        float(azimuth), dtype=torch.float64
                    )
                    steering = torch.exp(1j * k * (mics @ direction)).to(matrix.dtype)
                    out = matrix[index] @ steering
                    per_azimuth.append(
                        doa_error_degrees(out[1:].real, direction).item()
                    )

                errors.append(sum(per_azimuth) / len(per_azimuth))

            ax.plot(as_array(frequencies), errors, "o-", markersize=3, label=name)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("frequency (Hz)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

    axes[0].set_ylabel("mean direction error over 36 azimuths (degrees)")

    fig.suptitle(
        "5. The least-squares encoder only helps on an irregular, redundant array",
        fontsize=11,
    )

    return (
        save(fig, directory, "05_gradient_vs_least_squares.png"),
        "curves overlap on the square, separate on the irregular array",
    )


def example_06_salsa_features(directory):
    """What SALSA is actually for, in one picture.

    Two sources occupy different frequency bands and overlap in time for part
    of the clip. One is at -60 degrees in 300-1200 Hz, the other at +60
    degrees in 2500-5000 Hz, and they are both on between 1.2 and 1.8 s.

    Read the EIV panels against the spectrogram panels above them. During the
    overlap the EIV reports one direction in the lower band and a different
    one in the upper band, at the same instant. A frame-level feature such as
    GCC-PHAT has no frequency axis, so it would have to collapse both sources
    into a single direction per frame.

    Both sources switch on and off rather than running throughout, because
    the noise floor of Section II.D is a tracker: it creeps towards whatever
    is steady, so a source that never stops eventually becomes background and
    stops passing the magnitude test. That is intended behaviour for a
    feature built to find sound events.

    The white areas in the EIV panels are bins zeroed by Section II.D, either
    because they hold no foreground sound or because they failed the
    coherence test. Everything outside 50 Hz to 9 kHz is zeroed as well.
    """
    duration = 3 * SECOND

    def gate(start_seconds, end_seconds):
        """Switch a source on between two times, with short fades so the
        edges do not smear energy across the whole spectrum."""
        envelope = torch.zeros(duration)
        start, end = int(start_seconds * SECOND), int(end_seconds * SECOND)
        envelope[start:end] = 1.0

        fade = int(0.05 * SECOND)
        ramp = torch.linspace(0, 1, fade)
        envelope[start : start + fade] = ramp
        envelope[end - fade : end] = ramp.flip(0)

        return envelope

    low = band_noise(300.0, 1200.0, duration, seed=1) * gate(0.3, 1.8)
    high = band_noise(2500.0, 5000.0, duration, seed=2) * gate(1.2, 2.7)

    foa = (
        ideal_foa(direction_from_angles(-60.0), low)
        + ideal_foa(direction_from_angles(60.0), high)
        + 0.02 * torch.randn(1, 4, duration)
    )

    features = FoaSalsa(config=CONFIG).compute(foa)[0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.0), layout="constrained")

    spectrogram_top = features[:4].max().item()
    for index, name in enumerate(["W", "X", "Y", "Z"]):
        image = time_frequency(
            axes[0, index],
            features[index],
            CONFIG,
            f"log-spectrogram {name}",
            vmin=spectrogram_top - 14,
            vmax=spectrogram_top,
        )
    fig.colorbar(image, ax=axes[0, 3], fraction=0.05, label="log |X|^2")

    for index, name in enumerate(["X", "Y", "Z"]):
        image = time_frequency(
            axes[1, index],
            features[4 + index],
            CONFIG,
            f"EIV {name}",
            colormap=EIV_COLORMAP,
            vmin=-1,
            vmax=1,
        )

    kept = (features[4:].abs().sum(dim=0) > 0).float()
    time_frequency(
        axes[1, 3],
        kept,
        CONFIG,
        f"bins kept by Section II.D ({kept.mean() * 100:.0f}%)",
        colormap="Greys",
        vmin=0,
        vmax=1,
    )
    fig.colorbar(image, ax=axes[1, 3], fraction=0.05, label="EIV component")

    # only the outer panels need axis labels; repeating them eight times
    # leaves no room for the plots themselves
    for row in axes:
        for ax in row[1:]:
            ax.set_ylabel("")
    for ax in axes[0]:
        ax.set_xlabel("")

    fig.suptitle(
        "6. SALSA features: two sources, different bands, different directions\n"
        "-60 deg in 300-1200 Hz from 0.3 to 1.8 s; +60 deg in 2500-5000 Hz from 1.2 to 2.7 s; "
        "both are on between 1.2 and 1.8 s",
        fontsize=11,
    )

    return (
        save(fig, directory, "06_salsa_features.png"),
        f"{kept.mean() * 100:.0f}% of bins kept",
    )


def example_07_eigenvector_versus_intensity_vector(directory):
    """Why the paper takes an eigenvector instead of a cross-spectrum.

    Both features come from the same covariance matrix. The baseline of
    Section III.A reads its first column, which is the classic active
    intensity Re{W* [X, Y, Z]}. SALSA takes its principal eigenvector.

    With a single source they agree. With two sources in the same bins they
    do not: the first column returns sigma1^2 H1 + sigma2^2 H2, a weighted
    sum that points between the two sources and belongs to neither. The
    eigenvector approximates the dominant steering vector instead, so it
    stays closer to the louder source.

    A quiet source at amplitude a contributes a^2 in power, so the baseline
    should sit at about atan(a^2) degrees off the louder source. The dashed
    line is that prediction.
    """
    amplitudes = torch.linspace(0.0, 1.0, 11)
    duration = 2 * SECOND
    band = slice(20, 190)

    loud = white_noise(duration, seed=3)
    quiet = white_noise(duration, seed=4)

    eigenvector_bias, intensity_bias = [], []

    for amplitude in amplitudes:
        foa = ideal_foa(direction_from_angles(0.0), loud) + ideal_foa(
            direction_from_angles(90.0), quiet * float(amplitude)
        )

        cov = _covariance(stft(foa, CONFIG), window=7)
        vector, _, _ = _principal_eigenvector(cov, "eigh")

        for values, estimate in (
            (eigenvector_bias, _normalize_eigenvector(vector, 1e-10)),
            (intensity_bias, intensity_vector(cov)),
        ):
            mean = estimate[0, band].reshape(-1, 3).mean(dim=0)
            values.append(torch.rad2deg(torch.atan2(mean[1], mean[0])).item())

    predicted = torch.rad2deg(torch.atan(amplitudes.square()))

    fig, ax = plt.subplots(figsize=(8, 4.6), layout="constrained")

    ax.plot(as_array(amplitudes), eigenvector_bias, "o-", markersize=4,
            label="EIV, principal eigenvector (Section II.C.1)")
    ax.plot(as_array(amplitudes), intensity_bias, "s-", markersize=4,
            label="IV, first column of R (Section III.A)")
    ax.plot(as_array(amplitudes), as_array(predicted), "--", color="grey", linewidth=1,
            label="atan(a^2), the predicted IV bias")

    ax.set_xlabel("amplitude of the second source, relative to the first")
    ax.set_ylabel("reported azimuth (degrees)")
    ax.set_title(
        "7. With two sources the eigenvector stays nearer the louder one\n"
        "sources at 0 and 90 degrees, overlapping in every bin",
        fontsize=11,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # report the half-amplitude point, where the two differ most clearly;
    # at equal amplitude both correctly land midway between the sources
    half = len(amplitudes) // 2

    return (
        save(fig, directory, "07_eigenvector_vs_intensity_vector.png"),
        f"second source at half amplitude: EIV {eigenvector_bias[half]:.1f} deg, "
        f"IV {intensity_bias[half]:.1f} deg off the louder source",
    )


def example_08_diffuse_field(directory):
    """What the coherence test is actually for.

    Section II.A says equation (1) only holds for bins with a high
    direct-to-reverberant ratio, and beta_drr exists to reject the rest. Every
    other example here has a clean direct source, so the threshold never has
    to do its job. This one adds an isotropic diffuse field: a sum of 64
    uncorrelated plane waves from directions all over the sphere, which is the
    standard stand-in for a reverberant tail.

    The first panel is the point. The magnitude test is blind to
    reverberation, because diffuse energy is still energy, so its pass rate
    barely moves as the field gets more diffuse. The coherence test does all
    the work, dropping from 16 percent of bins to under 3.

    The second panel is a caveat on the first. Averaging over thousands of
    bins already suppresses most of the damage, so throwing the bad ones away
    changes the averaged direction only a little: 44 degrees becomes 40 at a
    direct-to-diffuse ratio of -5 dB. The value of the test is not in that
    average. It is that the feature handed to a network has 85 percent fewer
    bins claiming a direction they do not have, and a network reads every bin
    separately rather than averaging them.

    The third panel compares the split of bins against Fig. 3 of the paper.
    The numbers do not match, and the reason is worth stating rather than
    tuning away: the paper measures a dataset of dense real recordings, while
    this scene is three short events in silence, so far more bins fail the
    magnitude test here. The shape of the comparison is what is meaningful,
    not the agreement.

    The last panel puts a number on something the docstrings assert: with a
    one-frame covariance the matrix is rank one, sigma2 is zero, and the
    coherence test passes everything.
    """
    duration = 3 * SECOND
    truth = direction_from_angles(30.0)

    direct = ideal_foa(truth, white_noise(duration, seed=7))
    diffuse = diffuse_field(duration, n_waves=64, seed=0)

    freqs = CONFIG.frequencies()
    in_band = (freqs >= 50.0) & (freqs <= 9000.0)

    def live_mask(features):
        return (features[0, 4:].abs().sum(dim=0) > 0)[in_band]

    def direction_error(features):
        estimate = estimated_direction(features, band=in_band)
        if estimate.norm() == 0:
            return float("nan")
        return doa_error_degrees(estimate, truth).item()

    ratios_db = torch.tensor([20.0, 15.0, 10.0, 5.0, 0.0, -5.0, -10.0])

    magnitude_only = FoaSalsa(config=CONFIG, apply_coherence_test=False)
    both_tests = FoaSalsa(config=CONFIG)

    pass_magnitude, pass_both, error_with, error_without = [], [], [], []
    for ratio_db in ratios_db:
        scene = 10 ** (float(ratio_db) / 20) * direct + diffuse

        without = magnitude_only.compute(scene)
        with_both = both_tests.compute(scene)

        pass_magnitude.append(live_mask(without).float().mean().item() * 100)
        pass_both.append(live_mask(with_both).float().mean().item() * 100)
        error_without.append(direction_error(without))
        error_with.append(direction_error(with_both))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")

    axes[0, 0].plot(as_array(ratios_db), pass_magnitude, "o-", markersize=4,
                    label="magnitude test only")
    axes[0, 0].plot(as_array(ratios_db), pass_both, "s-", markersize=4,
                    label="both tests")
    axes[0, 0].set_xlabel("direct-to-diffuse ratio (dB)")
    axes[0, 0].set_ylabel("bins kept, in band (%)")
    axes[0, 0].set_title(
        "The magnitude test does not notice reverberation", fontsize=10
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].invert_xaxis()

    axes[0, 1].plot(as_array(ratios_db), error_with, "o-", markersize=4,
                    label="both tests")
    axes[0, 1].plot(as_array(ratios_db), error_without, "s-", markersize=4,
                    label="magnitude test only")
    axes[0, 1].set_xlabel("direct-to-diffuse ratio (dB)")
    axes[0, 1].set_ylabel("direction error of the surviving bins (degrees)")
    axes[0, 1].set_title(
        "Accuracy of the averaged estimate barely moves", fontsize=10
    )
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].invert_xaxis()

    # the three-way split of Fig. 3, on a scene of short events rather than
    # steady noise, which is closer to what the paper measured
    gate = torch.zeros(duration)
    for start in (0.3, 1.2, 2.1):
        gate[int(start * SECOND) : int((start + 0.4) * SECOND)] = 1.0

    events = 10 ** (5 / 20) * ideal_foa(truth, white_noise(duration, seed=7) * gate)
    scene = events + diffuse

    magnitude = live_mask(magnitude_only.compute(scene))
    survives = live_mask(both_tests.compute(scene))

    ours = [
        (~magnitude).float().mean().item() * 100,
        (magnitude & ~survives).float().mean().item() * 100,
        survives.float().mean().item() * 100,
    ]
    theirs = [35.0, 23.0, 40.0]
    labels = ["fail\nmagnitude", "pass magnitude,\nfail coherence", "pass\nboth"]

    positions = torch.arange(3, dtype=torch.float32)
    axes[1, 0].bar(as_array(positions - 0.2), ours, width=0.4, label="this scene")
    axes[1, 0].bar(as_array(positions + 0.2), theirs, width=0.4,
                   label="paper, Fig. 3 (FOA)")
    axes[1, 0].set_xticks(as_array(positions))
    axes[1, 0].set_xticklabels(labels, fontsize=8)
    axes[1, 0].set_ylabel("bins in band (%)")
    axes[1, 0].set_title(
        "Split of bins: three short events in a diffuse field", fontsize=10
    )
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.3, axis="y")

    windows = [1, 3, 5, 7, 11, 15]
    rates, errors = [], []
    for window in windows:
        coherence_only = FoaSalsa(
            config=CONFIG, cov_window=window, apply_magnitude_test=False
        )
        rates.append(live_mask(coherence_only.compute(direct + diffuse)).float().mean().item() * 100)
        errors.append(
            direction_error(FoaSalsa(config=CONFIG, cov_window=window).compute(direct + diffuse))
        )

    axes[1, 1].plot(windows, rates, "o-", markersize=4, color="tab:blue")
    axes[1, 1].set_xlabel("cov_window (the paper's 2 Tr + 1)")
    axes[1, 1].set_ylabel("coherence test pass rate (%)", color="tab:blue")
    axes[1, 1].tick_params(axis="y", labelcolor="tab:blue")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axvline(7, color="grey", linestyle="--", linewidth=1)
    axes[1, 1].annotate("paper: Tr = 3", xy=(7, max(rates)), fontsize=7,
                        color="grey", xytext=(4, -2), textcoords="offset points")

    twin = axes[1, 1].twinx()
    twin.plot(windows, errors, "s--", markersize=4, color="tab:red")
    twin.set_ylabel("direction error (degrees)", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")
    axes[1, 1].set_title(
        "One frame gives a rank-one covariance, so nothing is rejected",
        fontsize=10,
    )

    fig.suptitle(
        "8. The coherence test in a diffuse field",
        fontsize=11,
    )

    return (
        save(fig, directory, "08_diffuse_field.png"),
        f"at 0 dB direct-to-diffuse the coherence test keeps {pass_both[4]:.0f}% "
        f"of bins against {pass_magnitude[4]:.0f}%",
    )


EXAMPLES = [
    example_01_azimuth_sweep,
    example_02_planar_array_has_no_elevation,
    example_03_spatial_aliasing,
    example_04_gradient_compensation,
    example_05_gradient_versus_least_squares,
    example_06_salsa_features,
    example_07_eigenvector_versus_intensity_vector,
    example_08_diffuse_field,
]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv

    if len(argv) != 1:
        raise SystemExit("usage: python -m foasalsa.examples <output_dir>")

    directory = Path(argv[0])
    directory.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)

    for example in EXAMPLES:
        path, summary = example(directory)
        print(f"{path.name}: {summary}")

    print(f"\n{len(EXAMPLES)} figures written to {directory}")


if __name__ == "__main__":
    main()
