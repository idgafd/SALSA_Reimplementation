import time

import pytest
import torch

from foasalsa.encoding import FoaConverter
from foasalsa.salsa import (
    FoaSalsa,
    _covariance,
    _moving_average,
    _noise_floor,
    _normalize_eigenvector,
    _principal_eigenvector,
    intensity_vector,
)
from foasalsa.stft import StftConfig, stft
from foasalsa.synth import (
    direction_from_angles,
    ideal_foa,
    plane_wave,
    regular_polygon,
    tone,
)


METHODS = ["eigh", "svd", "power"]

SQUARE = regular_polygon(4, radius=0.05)


# a fixed seed keeps failures reproducible
torch.manual_seed(0)


@pytest.fixture
def config():
    return StftConfig()


def direction_from_azimuth(degrees):
    """Unit vector in the horizontal plane."""
    return direction_from_angles(degrees)


def scene(*sources, n_samples=24000, noise=0.0):
    """Sum of ideal plane waves, optionally plus spatially white noise.

    Each source is a (azimuth_degrees, amplitude) pair. Encoded analytically
    through Eq. (6) rather than recorded on an array, so there is no aliasing
    and no low-frequency droop: if a feature comes out wrong here, the fault
    is in the feature extractor and not in the geometry.
    """
    foa = torch.zeros(1, 4, n_samples)
    for azimuth, amplitude in sources:
        signal = torch.randn(n_samples) * amplitude
        foa = foa + ideal_foa(direction_from_azimuth(azimuth), signal)

    if noise:
        foa = foa + noise * torch.randn(1, 4, n_samples)

    return foa


def covariance_of(foa, config, window=7):
    return _covariance(stft(foa, config), window)


def average_eiv(features, band=slice(20, 190)):
    """Mean EIV direction over the bins the mask kept, ignoring band edges."""
    eiv = features[0, 4:, band]  # [3, F, T_frames]
    live = eiv.abs().sum(dim=0) > 0
    if not live.any():
        return torch.zeros(3)
    return eiv[:, live].mean(dim=-1)


def azimuth_of(vector):
    return torch.rad2deg(torch.atan2(vector[1], vector[0])).item()


class TestCovariance:
    def test_shape(self, config):
        spec = stft(scene((30, 1.0), n_samples=12000), config)

        cov = _covariance(spec, window=7)

        n_frames = spec.shape[-1]
        assert cov.shape == (1, config.n_freqs, n_frames, 4, 4)
        assert cov.dtype == torch.complex64

    def test_result_is_hermitian(self, config):
        cov = covariance_of(scene((30, 1.0), n_samples=12000), config)

        assert torch.allclose(cov, cov.mH, atol=1e-6)

    def test_diagonal_is_real_and_non_negative(self, config):
        cov = covariance_of(scene((30, 1.0), n_samples=12000), config)

        diagonal = cov.diagonal(dim1=-2, dim2=-1)

        # the diagonal is |X|^2, so it is real in exact arithmetic. Summing
        # seven complex products leaves a small imaginary part, and these
        # entries reach a few hundred, so the tolerance must be relative
        assert diagonal.imag.abs().max() < 1e-6 * diagonal.real.max()
        assert diagonal.real.min() >= 0

    def test_a_single_frame_window_is_exactly_rank_one(self, config):
        """This is why the paper averages over several frames. One frame gives
        X X^H, an outer product of a vector with itself, which has rank one
        no matter what the scene contained. The coherence test of Eq. (11)
        would then see a second eigenvalue of zero everywhere and reject
        nothing.
        """
        crowded = scene((0, 1.0), (90, 1.0), (200, 1.0), n_samples=12000, noise=0.5)

        cov = _covariance(stft(crowded, config), window=1)
        _, sigma1, sigma2 = _principal_eigenvector(cov, "eigh")

        assert sigma2.abs().max() < 1e-4 * sigma1.max()

    def test_averaging_lets_a_second_eigenvalue_appear(self, config):
        crowded = scene((0, 1.0), (90, 1.0), (200, 1.0), n_samples=12000, noise=0.5)

        cov = _covariance(stft(crowded, config), window=7)
        _, sigma1, sigma2 = _principal_eigenvector(cov, "eigh")

        assert sigma2.max() > 0.01 * sigma1.max()

    def test_it_averages_rather_than_accumulates(self, config):
        """Eq. (5) divides by the window size, and this checks that it does.

        Nothing downstream would notice if it did not. Scaling R by a
        constant scales both eigenvalues equally, so the eigenvector, the EIV
        and the sigma1/sigma2 ratio are all unchanged. The division only
        matters to code that reads the absolute values of R, which is why it
        needs its own test.
        """
        spec = stft(scene((30, 1.0), n_samples=12000), config)

        averaged = _covariance(spec, window=7)
        single = _covariance(spec, window=1)

        middle = slice(10, -10)
        assert averaged[:, :, middle].abs().mean() == pytest.approx(
            single[:, :, middle].abs().mean(), rel=0.3
        )

    def test_edge_frames_are_not_averaged_against_silence(self, config):
        # the input is steady noise, so every frame should carry about the
        # same energy. Padding with zeros instead of repeating the edge
        # frames would make the first few frames much quieter
        cov = covariance_of(scene((30, 1.0), n_samples=12000), config)

        energy = cov.diagonal(dim1=-2, dim2=-1).real.sum(dim=-1)  # [N, F, T]
        first = energy[..., 0].mean()
        middle = energy[..., 10:-10].mean()

        assert first > 0.5 * middle


class TestPrincipalEigenvector:
    @pytest.fixture
    def cov(self, config):
        return covariance_of(scene((30, 1.0), n_samples=12000), config)

    @pytest.mark.parametrize("method", METHODS)
    def test_eigenvalues_come_back_in_order(self, cov, method):
        _, sigma1, sigma2 = _principal_eigenvector(cov, method)

        assert (sigma1 >= sigma2 - 1e-6).all()

    @pytest.mark.parametrize("method", METHODS)
    def test_eigenvector_is_unit_norm(self, cov, method):
        vector, _, _ = _principal_eigenvector(cov, method)

        norms = torch.linalg.vector_norm(vector, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    @pytest.mark.parametrize("method", METHODS)
    def test_a_rank_one_covariance_returns_its_own_steering_vector(self, method):
        # build R = H H^H by hand, so the answer is known exactly
        steering = torch.tensor([1.0, 0.6, 0.8, 0.0], dtype=torch.complex64)
        cov = (steering[:, None] @ steering[None, :].conj()).reshape(1, 1, 1, 4, 4)

        vector, sigma1, sigma2 = _principal_eigenvector(cov, method)

        recovered = _normalize_eigenvector(vector, 1e-10)[0, 0, 0]
        assert torch.allclose(recovered, torch.tensor([0.6, 0.8, 0.0]), atol=1e-5)
        # H H^H has one non-zero eigenvalue, and it is |H|^2 = 1 + 0.36 + 0.64
        assert sigma1.item() == pytest.approx(2.0, rel=1e-4)
        assert sigma2.abs().item() < 1e-4

    @pytest.fixture
    def skewed_pair(self):
        """A rank-one covariance, and the same matrix with its upper triangle
        changed. The change is much larger than rounding would produce, so
        that the effect is visible in a test, but the mechanism is the same.
        """
        steering = torch.tensor([1.0, 0.6, 0.8, 0.0], dtype=torch.complex64)
        clean = (steering[:, None] @ steering[None, :].conj()).reshape(1, 1, 1, 4, 4)

        skewed = clean.clone()
        skewed[..., 0, 1] += 0.5
        skewed[..., 0, 2] -= 0.4

        return clean, skewed

    def test_the_upper_triangle_is_not_silently_discarded(self, skewed_pair):
        """This is what the one-line symmetrisation is for.

        torch.linalg.eigh reads one triangle and assumes the other mirrors
        it. Put 100 into the upper triangle and its eigenvalues do not change
        at all. The covariance is Hermitian, but only to floating point
        accuracy: summing seven complex products leaves the two triangles
        differing in the last digits. Averaging them first means both halves
        reach the decomposition instead of one being discarded.
        """
        clean, skewed = skewed_pair

        from_clean, _, _ = _principal_eigenvector(clean, "eigh")
        from_skewed, _, _ = _principal_eigenvector(skewed, "eigh")

        assert not torch.allclose(
            _normalize_eigenvector(from_clean, 1e-10),
            _normalize_eigenvector(from_skewed, 1e-10),
            atol=1e-3,
        )

    def test_eigh_and_svd_see_the_same_matrix(self, skewed_pair):
        # after symmetrising, the choice of routine cannot change the answer
        # even on an input that is not Hermitian
        _, skewed = skewed_pair

        from_eigh, _, _ = _principal_eigenvector(skewed, "eigh")
        from_svd, _, _ = _principal_eigenvector(skewed, "svd")

        assert torch.allclose(
            _normalize_eigenvector(from_eigh, 1e-10),
            _normalize_eigenvector(from_svd, 1e-10),
            atol=1e-4,
        )

    def test_rejects_an_unknown_method(self, cov):
        with pytest.raises(ValueError, match="method must be"):
            _principal_eigenvector(cov, "magic")


class TestMethodsAgree:
    """All three methods solve the same problem and must give the same
    answer. Measured on a 1-second single-item scene, 257 x 81 bins:

        eigh                18 ms
        svd                 39 ms
        power, 8 iterations 49 ms

    Power iteration is the slowest, which is the opposite of what its use of
    plain batched matmuls suggests. Eight iterations plus a deflation pass is
    sixteen passes over the tensor, while eigh does one pass in LAPACK. Power
    iteration is useful because it is differentiable and needs no LAPACK, not
    because it is fast.
    """

    @pytest.fixture
    def cov(self, config):
        return covariance_of(scene((30, 1.0), n_samples=12000), config)

    def test_eigh_and_svd_give_the_same_eiv(self, cov):
        from_eigh, _, _ = _principal_eigenvector(cov, "eigh")
        from_svd, _, _ = _principal_eigenvector(cov, "svd")

        assert torch.allclose(
            _normalize_eigenvector(from_eigh, 1e-10),
            _normalize_eigenvector(from_svd, 1e-10),
            atol=1e-4,
        )

    def test_eigh_and_svd_give_the_same_eigenvalues(self, cov):
        _, eigh1, eigh2 = _principal_eigenvector(cov, "eigh")
        _, svd1, svd2 = _principal_eigenvector(cov, "svd")

        assert torch.allclose(eigh1, svd1, rtol=1e-4)

        # sigma2 is the near-zero eigenvalue of a nearly rank-one matrix, so
        # the two routines disagree in the last few digits of a number that
        # is numerically noise. Judge it against the scale of the problem.
        assert (eigh2 - svd2).abs().max() < 1e-4 * eigh1.max()

    def test_power_iteration_agrees_on_the_bins_that_survive_the_mask(self, config):
        """Power iteration converges slowly where sigma1 and sigma2 are
        close, and those are exactly the bins the coherence test rejects.

        On a three-source scene the worst bins are still wrong after a dozen
        iterations. Convergence goes as (sigma2/sigma1)^n, and a low
        sigma1/sigma2 ratio is what the coherence test removes. On the bins
        that pass, the error drops by about a factor of ten per iteration:

            1 iter   1.5e-1      4 iters  5.7e-4
            2 iters  2.3e-2      8 iters  8.2e-7
        """
        crowded = scene((0, 1.0), (90, 1.0), (200, 1.0), n_samples=12000, noise=0.3)
        cov = _covariance(stft(crowded, config), window=7)

        exact, sigma1, sigma2 = _principal_eigenvector(cov, "eigh")
        approximate, _, _ = _principal_eigenvector(cov, "power", n_power_iterations=8)

        difference = (
            _normalize_eigenvector(exact, 1e-10)
            - _normalize_eigenvector(approximate, 1e-10)
        ).abs().max(dim=-1).values

        survives = sigma1 > 5.0 * sigma2.clamp(min=0)
        assert survives.any()
        assert difference[survives].max() < 1e-4
        # and, to show the caveat is real rather than rhetorical
        assert difference.max() > 0.1

    def test_eigh_is_the_cheapest_exact_route(self, cov):
        # the gap is about 2x, so this is not a close comparison. Take the
        # best of a few runs so a busy machine does not flip the result
        def best_of(method, repeats=3):
            timings = []
            for _ in range(repeats):
                start = time.perf_counter()
                _principal_eigenvector(cov, method)
                timings.append(time.perf_counter() - start)
            return min(timings)

        assert best_of("eigh") < best_of("svd")


class TestNormalizeEigenvector:
    def test_recovers_the_direction_from_a_steering_vector(self):
        direction = torch.tensor([0.6, 0.8, 0.0])
        vector = torch.tensor([1.0, 0.6, 0.8, 0.0], dtype=torch.complex64)

        eiv = _normalize_eigenvector(vector.reshape(1, 1, 1, 4), 1e-10)

        assert torch.allclose(eiv[0, 0, 0], direction, atol=1e-6)

    def test_an_arbitrary_complex_scale_makes_no_difference(self):
        # the eigenvector is only defined up to the source's loudness and
        # phase, and dividing by the W element removes both
        base = torch.tensor([1.0, 0.6, 0.8, 0.0], dtype=torch.complex64)
        scaled = base * (3.7 * torch.exp(1j * torch.tensor(1.1)))

        assert torch.allclose(
            _normalize_eigenvector(base.reshape(1, 1, 1, 4), 1e-10),
            _normalize_eigenvector(scaled.reshape(1, 1, 1, 4), 1e-10),
            atol=1e-5,
        )

    def test_result_is_unit_norm(self):
        vector = torch.randn(2, 5, 3, 4, dtype=torch.complex64)

        eiv = _normalize_eigenvector(vector, 1e-10)

        norms = torch.linalg.vector_norm(eiv, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_quadrature_xyz_collapse_to_nothing(self):
        """This is what an uncompensated encoder produces.

        A raw spatial difference between omnidirectional microphones measures
        grad(p), which for a plane wave is -j*k*p. XYZ are then 90 degrees
        out of phase with W, so the ratio below is purely imaginary and the
        real part this function keeps is zero.
        """
        quadrature = torch.tensor([1.0, -0.6j, -0.8j, 0.0], dtype=torch.complex64)

        eiv = _normalize_eigenvector(quadrature.reshape(1, 1, 1, 4), 1e-10)

        assert eiv.abs().max() == 0.0

    def test_an_empty_bin_stays_empty_instead_of_becoming_noise(self):
        # normalising rounding error would turn it into a unit vector
        # pointing in an arbitrary direction
        eiv = _normalize_eigenvector(torch.zeros(1, 1, 1, 4, dtype=torch.complex64), 1e-10)

        assert eiv.abs().max() == 0.0
        assert torch.isfinite(eiv).all()


class TestNoiseFloor:
    def test_shape_is_preserved(self):
        magnitude = torch.rand(2, 5, 40)

        floor = _noise_floor(magnitude, 1.02, 0.98, 5, 1e-10)

        assert floor.shape == magnitude.shape

    def test_it_climbs_towards_a_signal_that_stays_loud(self):
        magnitude = torch.cat([torch.full((1, 1, 5), 0.1), torch.full((1, 1, 60), 1.0)], -1)

        floor = _noise_floor(magnitude, 1.02, 0.98, 5, 1e-10)

        assert floor[0, 0, -1] > floor[0, 0, 0]

    def test_it_drifts_down_again_in_silence(self):
        magnitude = torch.cat([torch.full((1, 1, 5), 1.0), torch.zeros(1, 1, 60)], -1)

        floor = _noise_floor(magnitude, 1.02, 0.98, 5, 1e-10)

        assert floor[0, 0, -1] < floor[0, 0, 0]

    def test_it_moves_slowly_instead_of_following_the_signal(self):
        # one loud frame must not raise the floor to the level of the event,
        # otherwise the event would fail its own magnitude test
        magnitude = torch.full((1, 1, 40), 0.01)
        magnitude[0, 0, 20] = 100.0

        floor = _noise_floor(magnitude, 1.02, 0.98, 5, 1e-10)

        assert floor[0, 0, 21] < 1.0

    def test_digital_silence_does_not_drive_the_floor_to_zero(self):
        # if the floor reached zero, every later frame would pass the
        # magnitude test
        floor = _noise_floor(torch.zeros(1, 1, 500), 1.02, 0.98, 5, 1e-10)

        assert floor.min() > 0


class TestMovingAverage:
    def test_a_constant_signal_survives_unchanged(self):
        values = torch.full((1, 2, 30), 3.0)

        assert torch.allclose(_moving_average(values, 7), values)

    def test_window_of_one_is_a_no_op(self):
        values = torch.randn(1, 2, 30)

        assert torch.equal(_moving_average(values, 1), values)

    def test_it_smooths_a_spike(self):
        values = torch.zeros(1, 1, 30)
        values[0, 0, 15] = 7.0

        smoothed = _moving_average(values, 7)

        assert smoothed[0, 0, 15] == pytest.approx(1.0)
        assert smoothed[0, 0, 13] == pytest.approx(1.0)
        assert smoothed[0, 0, 11] == 0.0


class TestBinSelection:
    def test_out_of_band_bins_are_always_empty(self, config):
        features = FoaSalsa().compute(scene((30, 1.0), n_samples=12000))

        freqs = config.frequencies()
        below = freqs < 50.0
        above = freqs > 9000.0

        assert features[:, 4:, below].abs().max() == 0.0
        assert features[:, 4:, above].abs().max() == 0.0

    def test_silence_is_rejected_by_the_magnitude_test(self):
        silence = torch.zeros(1, 4, 12000)

        features = FoaSalsa().compute(silence)

        assert features[:, 4:].abs().max() == 0.0

    def test_two_equally_loud_sources_are_rejected_by_the_coherence_test(self):
        # the magnitude test is switched off so this measures the coherence
        # test alone. With both on, the magnitude test rejects so much of the
        # spectrum that the comparison no longer says anything about
        # coherence
        salsa = FoaSalsa(apply_magnitude_test=False)

        def live_fraction(*sources):
            features = salsa.compute(scene(*sources, n_samples=12000))
            return (features[0, 4:].abs().sum(dim=0) > 0).float().mean().item()

        one = live_fraction((0, 1.0))
        two = live_fraction((0, 1.0), (90, 1.0))
        three = live_fraction((0, 1.0), (90, 1.0), (200, 1.0))

        assert two < 0.5 * one
        assert three < two

    def test_turning_both_tests_off_keeps_the_whole_passband(self, config):
        crowded = scene((0, 1.0), (90, 1.0), n_samples=12000, noise=0.5)

        features = FoaSalsa(
            apply_magnitude_test=False, apply_coherence_test=False
        ).compute(crowded)

        freqs = config.frequencies()
        in_band = (freqs >= 50.0) & (freqs <= 9000.0)
        live = features[0, 4:].abs().sum(dim=0) > 0

        assert live[in_band].all()

    def test_a_single_frame_covariance_makes_the_coherence_test_useless(self):
        """When nothing is averaged, R is exactly rank one, so sigma2 is zero
        everywhere and every bin passes the coherence test regardless of how
        many sources are present.
        """
        crowded = scene((0, 1.0), (90, 1.0), (200, 1.0), n_samples=12000)

        def live_fraction(window):
            features = FoaSalsa(
                cov_window=window, apply_magnitude_test=False
            ).compute(crowded)
            return (features[0, 4:].abs().sum(dim=0) > 0).float().mean().item()

        assert live_fraction(1) > 0.9 * (9000 - 50) / 12000
        assert live_fraction(7) < 0.5 * live_fraction(1)


class TestEivVersusIntensityVector:
    """The baseline of Section III.A against the feature of Section II.C.1.
    Both are computed from the same covariance matrix: the baseline reads its
    first column, the feature takes its principal eigenvector.
    """

    def test_they_agree_when_there_is_one_source_and_white_noise(self, config):
        """Spatially white noise adds sigma_n^2 * I to the covariance. The
        first column of that is sigma_n^2 * e0, which affects only the W
        element, and the normalisation divides that out. The baseline is
        therefore unbiased here and the eigenvector has nothing to improve.

        The agreement is statistical rather than per-bin. Individual bins
        where the noise dominates still disagree, which is why the two tests
        of Section II.D exist.
        """
        cov = covariance_of(scene((30, 1.0), n_samples=12000, noise=0.2), config)

        vector, _, _ = _principal_eigenvector(cov, "eigh")
        eiv = _normalize_eigenvector(vector, 1e-10)
        iv = intensity_vector(cov)

        band = slice(20, 190)
        cosine = (eiv[:, band] * iv[:, band]).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.arccos(cosine))

        assert angle.mean() < 2.0
        assert (eiv[:, band] - iv[:, band]).abs().flatten().quantile(0.99) < 0.05

    def test_the_eigenvector_locks_onto_the_louder_of_two_sources(self, config):
        """The main claim of the paper, checked on synthetic data.

        With a second source at half the amplitude, the first column returns
        sigma1^2 H1 + sigma2^2 H2. That is a weighted sum pointing at neither
        source, about atan(0.25) = 14 degrees away from the louder one. The
        eigenvector approximates the dominant steering vector instead and
        lands closer, near 10 degrees.
        """
        cov = covariance_of(scene((0, 1.0), (90, 0.5), n_samples=24000), config)

        vector, _, _ = _principal_eigenvector(cov, "eigh")
        eiv = _normalize_eigenvector(vector, 1e-10)
        iv = intensity_vector(cov)

        band = slice(20, 190)
        eiv_bias = azimuth_of(eiv[0, band].reshape(-1, 3).mean(0))
        iv_bias = azimuth_of(iv[0, band].reshape(-1, 3).mean(0))

        assert 0 < eiv_bias < iv_bias
        assert iv_bias == pytest.approx(14.0, abs=3.0)


class TestFoaSalsa:
    def test_output_shape(self, config):
        foa = scene((30, 1.0), n_samples=24000)

        features = FoaSalsa().compute(foa)

        n_frames = 1 + 24000 // config.hop_length
        assert features.shape == (1, 7, config.n_freqs, n_frames)
        assert features.dtype == torch.float32

    def test_the_first_four_channels_are_log_spectrograms(self, config):
        foa = scene((30, 1.0), n_samples=12000)
        salsa = FoaSalsa()

        features = salsa.compute(foa)

        expected = torch.log(stft(foa, config).abs().square() + salsa.eps)
        assert torch.allclose(features[:, :4], expected, atol=1e-5)

    @pytest.mark.parametrize("azimuth", [0, 30, 90, 150, -60])
    def test_recovers_the_direction_of_an_ideal_source(self, azimuth):
        features = FoaSalsa().compute(scene((azimuth, 1.0), n_samples=24000))

        assert torch.allclose(
            average_eiv(features), direction_from_azimuth(azimuth), atol=0.05
        )

    def test_planar_array_leaves_the_z_channel_empty(self):
        """Runs through FoaConverter as well. The array cannot measure
        elevation, so the encoder produces a zero Z channel and the feature
        has nothing to report there.
        """
        audio = plane_wave(SQUARE, direction_from_azimuth(30), tone(500.0, 24000))
        foa = FoaConverter().convert(audio, SQUARE)

        features = FoaSalsa().compute(foa)

        assert features[:, 6].abs().max() == 0.0

    def test_recovers_the_azimuth_through_the_whole_pipeline(self):
        direction = direction_from_azimuth(30)
        audio = plane_wave(SQUARE, direction, tone(500.0, 24000))

        features = FoaSalsa().compute(FoaConverter().convert(audio, SQUARE))

        assert torch.allclose(average_eiv(features), direction, atol=0.05)

    @pytest.mark.parametrize("method", METHODS)
    def test_every_method_produces_the_same_features(self, method):
        foa = scene((30, 1.0), n_samples=12000)

        reference = FoaSalsa(method="eigh").compute(foa)
        features = FoaSalsa(method=method).compute(foa)

        assert torch.allclose(features, reference, atol=1e-4)

    def test_batch_items_do_not_leak_into_each_other(self):
        foa = torch.cat(
            [scene((0, 1.0), n_samples=12000), scene((90, 1.0), n_samples=12000)]
        )
        salsa = FoaSalsa()

        batched = salsa.compute(foa)

        for n in range(2):
            alone = salsa.compute(foa[n : n + 1])
            assert torch.allclose(batched[n], alone[0], atol=1e-4)

    def test_features_stay_finite_on_digital_silence(self):
        features = FoaSalsa().compute(torch.zeros(1, 4, 12000))

        assert torch.isfinite(features).all()


class TestFoaSalsaValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"method": "magic"},
            {"cov_window": 0},
            {"cov_window": 8},  # even, so it has no centre frame
            {"cov_window": -7},
            {"fmin": 9000.0, "fmax": 50.0},
            {"fmin": -1.0},
            {"alpha_snr": -1.0},
            {"beta_drr": 0.5},  # sigma1 >= sigma2 always, so a ratio below
                                # one would select every bin
            {"n_power_iterations": 0},
            {"noise_floor_rise": 0.9},  # would shrink on loud frames
            {"noise_floor_fall": 1.1},  # would grow on quiet ones
            {"n_init_frames": 0},
        ],
    )
    def test_rejects_broken_settings(self, kwargs):
        with pytest.raises(ValueError):
            FoaSalsa(**kwargs)

    def test_rejects_audio_that_is_not_three_dimensional(self):
        with pytest.raises(ValueError, match="foa must be"):
            FoaSalsa().compute(torch.randn(4, 12000))

    def test_rejects_a_channel_count_that_is_not_four(self):
        with pytest.raises(ValueError, match="4 channels"):
            FoaSalsa().compute(torch.randn(1, 6, 12000))
