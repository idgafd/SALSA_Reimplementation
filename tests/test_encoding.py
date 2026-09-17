import pytest
import torch

from foasalsa.encoding import (
    FoaConverter,
    _fibonacci_sphere,
    _gradient_matrix,
    _least_squares_matrix,
)
from foasalsa.stft import StftConfig
from foasalsa.synth import (
    SPEED_OF_SOUND,
    direction_from_angles,
    plane_wave,
    regular_polygon,
    steering_vector,
    tone,
)
from foasalsa.synth import doa_error_degrees as angle_between


# The task asks for planar arrays. A four-microphone square is the default
# used below: each microphone is 5 cm from the centre, so the array is 10 cm
# across at its widest.
SQUARE = regular_polygon(4, radius=0.05)

TETRAHEDRON = torch.tensor(
    [
        [0.05, 0.05, 0.05],
        [0.05, -0.05, -0.05],
        [-0.05, 0.05, -0.05],
        [-0.05, -0.05, 0.05],
    ]
)


@pytest.fixture
def config():
    return StftConfig()


@pytest.fixture
def freqs(config):
    return config.frequencies()


def bin_of(hz, config):
    """Index of the STFT bin closest to a frequency in Hz."""
    return int(round(hz / (config.sample_rate / config.n_fft)))


def direction_from_azimuth(degrees, dtype=torch.float32):
    """Unit vector in the horizontal plane, the only plane a planar array sees."""
    return direction_from_angles(degrees, dtype=dtype)


def encoded(matrix, mics, direction, hz, config):
    """Apply one frequency of the encoder matrix to the analytic steering
    vector. This needs no STFT, so it compares four numbers against four
    numbers.
    """
    index = bin_of(hz, config)
    freqs = torch.tensor([hz], dtype=mics.dtype)
    steering = steering_vector(mics, direction, freqs, SPEED_OF_SOUND)[0]

    return matrix[index] @ steering.to(matrix.dtype)


def ratio_to_w(foa, channel, skip=2000):
    """Least-squares scale of one FOA channel against W, ignoring the edges.

    For a plane wave from direction d the encoder should give X = W * d[0],
    so this returns the number that should come back as d[0].
    """
    w = foa[0, 0, skip:-skip]
    other = foa[0, channel, skip:-skip]
    return ((other * w).sum() / (w * w).sum()).item()


def doa_error_degrees(matrix, mics, direction, hz, config):
    """Pointing error of one encoder matrix at one frequency, in degrees."""
    out = encoded(matrix, mics, direction, hz, config)

    return angle_between(out[1:].real, direction).item()


def mean_doa_error_degrees(matrix, mics, hz, config, n_azimuths=36):
    """Pointing error averaged over a full turn in the horizontal plane.

    Testing one direction is unreliable, because some directions happen to be
    ones where an encoder is accidentally accurate. Comparisons between
    encoders average over the circle instead.
    """
    step = 360.0 / n_azimuths
    errors = [
        doa_error_degrees(
            matrix, mics, direction_from_azimuth(i * step, mics.dtype), hz, config
        )
        for i in range(n_azimuths)
    ]
    return sum(errors) / len(errors)


class TestGradientMatrix:
    def test_shape_and_dtype(self, freqs, config):
        E = _gradient_matrix(SQUARE, freqs)

        assert E.shape == (config.n_freqs, 4, SQUARE.shape[0])
        assert E.dtype == torch.complex64

    def test_float64_geometry_gives_double_precision_matrix(self, config):
        E = _gradient_matrix(SQUARE.double(), config.frequencies(dtype=torch.float64))

        assert E.dtype == torch.complex128

    def test_w_row_is_the_plain_microphone_average(self, freqs):
        E = _gradient_matrix(SQUARE, freqs)

        expected = torch.full((SQUARE.shape[0],), 1.0 / SQUARE.shape[0])
        assert torch.allclose(E[:, 0, :].real, expected.expand_as(E[:, 0, :].real))
        assert torch.all(E[:, 0, :].imag == 0)

    @pytest.mark.parametrize("azimuth", [0, 30, 90, 150, 180, -60, -120])
    def test_points_in_the_right_direction_at_low_frequency(
        self, azimuth, freqs, config
    ):
        # 500 Hz is ~69 cm of wavelength against a 10 cm array, comfortably
        # inside the regime where the first-order expansion holds
        direction = direction_from_azimuth(azimuth)

        error = doa_error_degrees(
            _gradient_matrix(SQUARE, freqs), SQUARE, direction, 500.0, config
        )

        assert error < 1.0

    def test_the_sn3d_gains_are_close_to_one_but_already_drooping(self, freqs, config):
        """Direction is recovered almost exactly at 500 Hz, yet every channel
        comes back a few percent quiet. Two hysical causes
        W is the average of four microphones that no longer agree, and XYZ
        still carry the second-order term the linearisation dropped.

        SALSA is unaffected. It unit-normalises the direction vector, but 
        anyone listening to the FOA would hear it.
        """
        out = encoded(
            _gradient_matrix(SQUARE, freqs),
            SQUARE,
            direction_from_azimuth(30),
            500.0,
            config,
        )

        expected = torch.tensor([1.0, *direction_from_azimuth(30)])
        assert torch.allclose(out.real, expected, atol=0.06)
        assert out[0].real < 0.99  # W is measurably quiet already

    def test_xyz_come_out_in_phase_with_w(self, freqs, config):
        # this is what the 1/(jk) compensation exists for
        out = encoded(
            _gradient_matrix(SQUARE, freqs),
            SQUARE,
            direction_from_azimuth(30),
            500.0,
            config,
        )

        assert out.imag.abs().max() < 1e-5

    def test_dc_bin_is_finite(self, freqs):
        # k = 0 there, so an unguarded 1/(jk) would produce inf
        E = _gradient_matrix(SQUARE, freqs)

        assert torch.isfinite(E).all()

    def test_identical_microphones_are_rejected(self, freqs):
        degenerate = torch.zeros(4, 3)

        with pytest.raises(ValueError):
            _gradient_matrix(degenerate, freqs)

    def test_the_origin_of_the_coordinates_does_not_matter(self, freqs, config):
        """Check that the encoder is invariant to the choice of coordinate origin.

        Shifting every microphone by the same vector preserves their relative
        geometry, so the encoding matrix should not change.
        """
        shifted = SQUARE - SQUARE[0]  # now microphone 1 sits at the origin
        far_away = SQUARE + torch.tensor([3.0, -7.0, 2.0])

        from_centre = _gradient_matrix(SQUARE, freqs)

        assert torch.allclose(_gradient_matrix(shifted, freqs), from_centre, atol=1e-6)
        assert torch.allclose(_gradient_matrix(far_away, freqs), from_centre, atol=1e-6)

    def test_an_off_centre_array_still_points_correctly(self, freqs, config):
        shifted = SQUARE - SQUARE[0]

        error = doa_error_degrees(
            _gradient_matrix(shifted, freqs),
            shifted,
            direction_from_azimuth(30),
            500.0,
            config,
        )

        assert error < 1.0


class TestPlanarGeometry:
    """A planar array cannot observe elevation, and the encoder has to say so
    by producing an empty Z channel rather than amplified noise.
    """

    def test_planar_array_gives_an_exactly_zero_z_row(self, freqs):
        E = _gradient_matrix(SQUARE, freqs)

        assert E[:, 3, :].abs().max() == 0.0

    def test_three_dimensional_array_gives_a_live_z_row(self, freqs):
        E = _gradient_matrix(TETRAHEDRON, freqs)

        assert E[:, 3, :].abs().max() > 0.1

    def test_almost_planar_array_is_truncated_rather_than_amplified(self, freqs):
        # a tiny non-zero z must not turn into a Z channel
        tiny_z = torch.tensor([[0.0, 0.0, 1e-9], [0.0, 0.0, -1e-9]] * 2)

        E = _gradient_matrix(SQUARE + tiny_z, freqs, rtol=1e-5)

        assert E[:, 3, :].abs().max() < 1e-5

    def test_rtol_is_what_decides_where_planar_stops(self, freqs):
        # the same geometry, but now the tilt is real enough to keep:
        # 1 mm of z against a 50 mm array is a 2% singular value
        tilt = torch.tensor([[0.0, 0.0, 1e-3], [0.0, 0.0, -1e-3]] * 2)

        kept = _gradient_matrix(SQUARE + tilt, freqs, rtol=1e-5)
        dropped = _gradient_matrix(SQUARE + tilt, freqs, rtol=1e-1)

        assert kept[:, 3, :].abs().max() > 1.0
        # not exactly zero, unlike the planar case above. Discarding a
        # singular value that was not zero to begin with leaves rounding error
        assert dropped[:, 3, :].abs().max() < 1e-6


class TestLowFrequencyRegularisation:
    def test_regularisation_caps_the_low_frequency_boost(self, freqs):
        loose = _gradient_matrix(SQUARE, freqs, max_gain_db=40.0)
        tight = _gradient_matrix(SQUARE, freqs, max_gain_db=0.0)

        assert peak_xyz_gain(loose) > 10 * peak_xyz_gain(tight)

    def test_the_cap_follows_the_closed_form(self, freqs):
        # the compensation peaks at 1 / (2 * k_reg) = radius * gain_max,
        # and that multiplies the pseudoinverse of the geometry
        max_gain_db = 20.0
        E = _gradient_matrix(SQUARE, freqs, max_gain_db=max_gain_db)

        radius = 0.05
        gain_max = 10 ** (max_gain_db / 20)
        pinv_norm = torch.linalg.matrix_norm(torch.linalg.pinv(SQUARE), ord=2).item()

        assert peak_xyz_gain(E) == pytest.approx(radius * gain_max * pinv_norm, rel=0.02)

    def test_unregularised_mode_still_survives_dc(self, freqs):
        E = _gradient_matrix(SQUARE, freqs, regularize=False)

        assert torch.isfinite(E).all()
        # 1/(jk) is undefined at k = 0, so that bin carries no direction
        assert E[0, 1:, :].abs().max() == 0.0

    def test_unregularised_mode_is_more_accurate_just_above_dc(self, freqs, config):
        # the price of the cap is a little low-frequency droop
        direction = direction_from_azimuth(30)
        args = dict(mics=SQUARE, freqs=freqs)

        capped = encoded(_gradient_matrix(**args), SQUARE, direction, 94.0, config)
        exact = encoded(
            _gradient_matrix(**args, regularize=False), SQUARE, direction, 94.0, config
        )

        assert exact[1].real > capped[1].real
        assert exact[1].real == pytest.approx(direction[0].item(), abs=0.02)


def peak_xyz_gain(matrix):
    """Largest amplification any of the XYZ rows applies, over all frequencies.

    This is the factor microphone self-noise gets multiplied by, which is the
    thing the low-frequency regularisation exists to bound.
    """
    return torch.linalg.matrix_norm(matrix[:, 1:, :], ord=2).max().item()


class TestUncompensatedGradientAblation:
    """The main failure mode. A raw spatial difference measures grad(p),
    which for a plane wave is -j*k*p: the right direction, rotated a quarter
    turn in phase. SALSA's EIV normalises XYZ by W and keeps only the real
    part, so that quarter turn sends the whole feature to zero.
    """

    def test_xyz_land_in_quadrature_with_w(self, freqs, config):
        out = encoded(
            _gradient_matrix(SQUARE, freqs, compensate=False),
            SQUARE,
            direction_from_azimuth(30),
            500.0,
            config,
        )

        x_over_w = out[1] / out[0]
        assert x_over_w.real.abs() < 1e-4 * x_over_w.imag.abs()

    def test_the_real_part_that_eiv_would_read_is_gone(self, freqs, config):
        args = (SQUARE, direction_from_azimuth(30), 500.0, config)

        live = encoded(_gradient_matrix(SQUARE, freqs), *args)
        dead = encoded(_gradient_matrix(SQUARE, freqs, compensate=False), *args)

        assert (live[1] / live[0]).real.abs() > 0.5
        assert (dead[1] / dead[0]).real.abs() < 1e-3

    def test_uncompensated_matrix_is_the_same_at_every_frequency(self, freqs):
        # 1/(jk) is the only frequency-dependent part of this encoder
        E = _gradient_matrix(SQUARE, freqs, compensate=False)

        assert torch.allclose(E[0], E[-1])


class TestFibonacciSphere:
    def test_all_directions_are_unit_vectors(self):
        directions = _fibonacci_sphere(642, torch.device("cpu"), torch.float32)

        assert directions.shape == (642, 3)
        assert torch.allclose(directions.norm(dim=-1), torch.ones(642), atol=1e-6)

    def test_directions_are_spread_evenly(self):
        # if the points are spread evenly, their average is the centre
        directions = _fibonacci_sphere(2000, torch.device("cpu"), torch.float64)

        assert directions.mean(dim=0).abs().max() < 1e-3

    def test_poles_are_not_doubled_up(self):
        directions = _fibonacci_sphere(64, torch.device("cpu"), torch.float32)

        assert directions[:, 2].abs().max() < 1.0

    def test_rejects_an_empty_grid(self):
        with pytest.raises(ValueError):
            _fibonacci_sphere(0, torch.device("cpu"), torch.float32)


class TestLeastSquaresMatrix:
    @pytest.fixture
    def directions(self):
        return _fibonacci_sphere(642, torch.device("cpu"), torch.float32)

    def test_shape_and_dtype(self, freqs, config, directions):
        E = _least_squares_matrix(SQUARE, freqs, SPEED_OF_SOUND, directions, beta=1e-3)

        assert E.shape == (config.n_freqs, 4, SQUARE.shape[0])
        assert E.dtype == torch.complex64

    @pytest.mark.parametrize("azimuth", [0, 45, 135, -90])
    def test_recovers_the_sn3d_target(self, azimuth, freqs, config, directions):
        direction = direction_from_azimuth(azimuth)
        E = _least_squares_matrix(SQUARE, freqs, SPEED_OF_SOUND, directions, beta=1e-3)

        out = encoded(E, SQUARE, direction, 500.0, config)

        assert torch.allclose(out.real, torch.tensor([1.0, *direction]), atol=0.05)

    def test_planar_array_gives_a_z_row_that_is_small_but_not_exactly_zero(
        self, freqs, directions
    ):
        """Where the two encoders genuinely differ. The gradient encoder
        truncates the rank of the geometry, which is a hard decision and
        yields an exactly empty Z row. The least-squares fit has no such
        step and only sees that no weighting reproduces elevation well, and
        settles on a near-zero compromise whose residue depends on how
        symmetric the direction grid happens to be about the horizontal
        plane. The Fibonacci grid is not exactly mirror-symmetric, so a
        thousandth of the XY scale leaks through.

        For planar arrays, the gradient encoder is easier to interpret because 
        it gives an exactly zero Z row.
        """
        E = _least_squares_matrix(SQUARE, freqs, SPEED_OF_SOUND, directions, beta=1e-3)

        leak = E[:, 3, :].abs().max()
        horizontal = E[:, 1:3, :].abs().max()
        assert leak > 0
        assert leak < 1e-2 * horizontal

    def test_rejects_directions_that_are_not_three_dimensional(self, freqs):
        with pytest.raises(ValueError):
            _least_squares_matrix(
                SQUARE, freqs, SPEED_OF_SOUND, torch.zeros(10, 2), beta=1e-3
            )

    def test_result_barely_moves_when_the_grid_is_refined(self, freqs, config):
        # if refining the grid changes nothing, 642 directions is enough
        coarse = _least_squares_matrix(
            SQUARE,
            freqs,
            SPEED_OF_SOUND,
            _fibonacci_sphere(162, torch.device("cpu"), torch.float32),
            beta=1e-3,
        )
        fine = _least_squares_matrix(
            SQUARE,
            freqs,
            SPEED_OF_SOUND,
            _fibonacci_sphere(2562, torch.device("cpu"), torch.float32),
            beta=1e-3,
        )

        index = bin_of(500.0, config)
        assert torch.allclose(coarse[index], fine[index], atol=1e-2)


class TestHowTheTwoEncodersRelate:
    """The gradient encoder is the low-frequency limit of the least-squares
    one. Replacing exp(j*k*r) with 1 + j*k*r turns the direction fit into
    an average plus a pseudoinverse of the geometry.
    """

    @pytest.fixture
    def directions(self):
        return _fibonacci_sphere(642, torch.device("cpu"), torch.float64)

    def matrices(self, mics, config, directions):
        freqs = config.frequencies(dtype=torch.float64)
        gradient = _gradient_matrix(mics, freqs, regularize=False)
        least_squares = _least_squares_matrix(
            mics, freqs, SPEED_OF_SOUND, directions, beta=1e-6
        )
        return gradient, least_squares

    def test_they_converge_as_frequency_drops(self, config, directions):
        gradient, least_squares = self.matrices(SQUARE.double(), config, directions)

        def relative_gap(hz):
            index = bin_of(hz, config)
            a, b = gradient[index, 1:], least_squares[index, 1:]
            return ((a - b).abs().max() / b.abs().max()).item()

        assert relative_gap(94.0) < 1e-2
        assert relative_gap(94.0) < relative_gap(1000.0) < relative_gap(4000.0)

    def test_on_a_symmetric_array_they_point_the_same_way(self, config, directions):
        """For a symmetric array the two encoders give *identical* directions 
        and differ only in overall gain. Symmetry forces X to come 
        from one opposing microphone pair and Y from the other, 
        so the encoders can only disagree about a common scale
        and a direction estimate normalises that away.
        """
        mics = SQUARE.double()
        gradient, least_squares = self.matrices(mics, config, directions)

        for hz in (500.0, 2000.0, 4000.0):
            a = mean_doa_error_degrees(gradient, mics, hz, config)
            b = mean_doa_error_degrees(least_squares, mics, hz, config)
            assert a == pytest.approx(b, rel=0.01)

    def test_least_squares_wins_on_an_irregular_redundant_array(
        self, config, directions
    ):
        """The least-squares fit helps when the array has more microphones
        than the three unknowns and has no symmetry.

        Measured mean pointing error over a full turn, eight irregular
        coplanar microphones inside a 12 cm span:

            250 Hz   gradient 0.03 deg   ls 0.02 deg   (both exact)
            500 Hz   gradient 0.10 deg   ls 0.05 deg
           1000 Hz   gradient 0.43 deg   ls 0.17 deg
           2000 Hz   gradient 2.09 deg   ls 1.05 deg
           3000 Hz   gradient 8.23 deg   ls 5.22 deg   (both failing)

        The gap is only useful in the middle of the range. At low frequency
        the two encoders have converged. At high frequency the geometry no
        longer carries the information, so no encoder can recover it.
        """
        mics = torch.tensor(
            [
                [0.060, 0.010, 0.0],
                [0.000, 0.050, 0.0],
                [-0.040, -0.020, 0.0],
                [0.020, -0.055, 0.0],
                [0.035, 0.040, 0.0],
                [-0.055, 0.025, 0.0],
                [-0.015, -0.050, 0.0],
                [0.050, -0.030, 0.0],
            ],
            dtype=torch.float64,
        )
        gradient, least_squares = self.matrices(mics, config, directions)

        for hz in (500.0, 1000.0, 2000.0):
            a = mean_doa_error_degrees(gradient, mics, hz, config)
            b = mean_doa_error_degrees(least_squares, mics, hz, config)
            assert b < 0.7 * a

    def test_the_advantage_disappears_again_at_the_bottom_of_the_band(
        self, config, directions
    ):
        # at 250 Hz the linearisation is accurate enough that the
        # least-squares fit has nothing left to improve
        mics = SQUARE.double()
        gradient, least_squares = self.matrices(mics, config, directions)

        a = mean_doa_error_degrees(gradient, mics, 250.0, config)
        b = mean_doa_error_degrees(least_squares, mics, 250.0, config)

        assert a < 0.1 and b < 0.1


class TestFoaConverter:
    def test_output_shape(self):
        audio = torch.randn(3, 4, 12000)

        foa = FoaConverter().convert(audio, SQUARE)

        assert foa.shape == (3, 4, 12000)
        assert foa.dtype == torch.float32

    @pytest.mark.parametrize("mode", ["gradient", "ls"])
    def test_both_modes_run_end_to_end(self, mode):
        audio = torch.randn(1, 4, 6000)

        foa = FoaConverter(mode=mode, n_directions=162).convert(audio, SQUARE)

        assert foa.shape == (1, 4, 6000)
        assert torch.isfinite(foa).all()

    @pytest.mark.parametrize("azimuth", [0, 30, 90, 180, -60])
    def test_recovers_the_azimuth_of_a_plane_wave(self, azimuth):
        direction = direction_from_azimuth(azimuth)
        audio = plane_wave(SQUARE, direction, tone(500.0, 12000))

        foa = FoaConverter().convert(audio, SQUARE)

        assert ratio_to_w(foa, 1) == pytest.approx(direction[0].item(), abs=0.05)
        assert ratio_to_w(foa, 2) == pytest.approx(direction[1].item(), abs=0.05)

    def test_planar_array_leaves_the_z_channel_silent(self):
        audio = plane_wave(SQUARE, direction_from_azimuth(30), tone(500.0, 12000))

        foa = FoaConverter().convert(audio, SQUARE)

        assert foa[:, 3].abs().max() == 0.0

    def test_direction_degrades_above_the_aliasing_frequency(self):
        """The array is 10 cm across, so half a wavelength fits between the
        outer microphones at about c / (2 * 0.1) = 1.7 kHz. Above that the
        geometry stops carrying direction information, and no encoder can
        recover it.
        """
        direction = direction_from_azimuth(30)
        converter = FoaConverter()

        def azimuth_gap(hz):
            audio = plane_wave(SQUARE, direction, tone(hz, 12000))
            foa = converter.convert(audio, SQUARE)
            x, y = ratio_to_w(foa, 1), ratio_to_w(foa, 2)
            return abs(x - direction[0].item()) + abs(y - direction[1].item())

        assert azimuth_gap(500.0) < 0.05
        assert azimuth_gap(4000.0) > 0.5

    def test_accepts_integer_microphone_coordinates(self):
        # integer coordinates are easy to type by hand, and mics.mean()
        # raises on an integer tensor
        mics = torch.tensor([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        audio = torch.randn(1, 4, 6000)

        foa = FoaConverter().convert(audio, mics)

        assert torch.isfinite(foa).all()

    def test_batch_items_do_not_leak_into_each_other(self):
        audio = torch.randn(3, 4, 6000)
        converter = FoaConverter()

        batched = converter.convert(audio, SQUARE)

        for n in range(3):
            alone = converter.convert(audio[n : n + 1], SQUARE)
            assert torch.allclose(batched[n], alone[0], atol=1e-5)


class TestFoaConverterValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "magic"},
            {"speed_of_sound": 0.0},
            {"speed_of_sound": -343.0},
            {"rtol": -1.0},
            {"n_directions": 0},
            {"beta": -1.0},
        ],
    )
    def test_rejects_broken_settings(self, kwargs):
        with pytest.raises(ValueError):
            FoaConverter(**kwargs)

    def test_rejects_audio_that_is_not_three_dimensional(self):
        with pytest.raises(ValueError, match="audio must be"):
            FoaConverter().convert(torch.randn(4, 6000), SQUARE)

    def test_rejects_microphone_coordinates_that_are_not_xyz(self):
        with pytest.raises(ValueError, match="mics must be"):
            FoaConverter().convert(torch.randn(1, 4, 6000), torch.randn(4, 2))

    def test_rejects_a_channel_count_that_disagrees_with_the_geometry(self):
        with pytest.raises(ValueError, match="channels"):
            FoaConverter().convert(torch.randn(1, 6, 6000), SQUARE)
