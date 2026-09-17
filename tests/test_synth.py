"""Tests for the synthetic signals and geometries.

These functions are the ground truth used by the other test files, so an
error here would make the package agree with itself and with nothing else.
"""

import pytest
import torch

from foasalsa.stft import StftConfig, stft
from foasalsa.synth import (
    SPEED_OF_SOUND,
    angles_from_direction,
    band_noise,
    chirp,
    direction_from_angles,
    doa_error_degrees,
    ideal_foa,
    plane_wave,
    random_planar,
    regular_polygon,
    steering_vector,
    tone,
    white_noise,
)



SQUARE = regular_polygon(4, radius=0.05)


class TestDirections:
    @pytest.mark.parametrize(
        "azimuth, expected",
        [
            (0, [1.0, 0.0, 0.0]),
            (90, [0.0, 1.0, 0.0]),
            (180, [-1.0, 0.0, 0.0]),
            (-90, [0.0, -1.0, 0.0]),
        ],
    )
    def test_azimuth_points_where_it_says(self, azimuth, expected):
        direction = direction_from_angles(azimuth)

        assert torch.allclose(direction, torch.tensor(expected), atol=1e-6)

    def test_elevation_lifts_out_of_the_plane(self):
        assert torch.allclose(
            direction_from_angles(0, 90), torch.tensor([0.0, 0.0, 1.0]), atol=1e-6
        )

    @pytest.mark.parametrize("azimuth", [0, 37, 90, 175, -60])
    @pytest.mark.parametrize("elevation", [0, 30, -45])
    def test_the_angles_survive_a_round_trip(self, azimuth, elevation):
        direction = direction_from_angles(azimuth, elevation)

        back_azimuth, back_elevation = angles_from_direction(direction)

        assert back_azimuth.item() == pytest.approx(azimuth, abs=1e-3)
        assert back_elevation.item() == pytest.approx(elevation, abs=1e-3)

    def test_result_is_unit_length(self):
        direction = direction_from_angles(37, 21)

        assert direction.norm().item() == pytest.approx(1.0, abs=1e-6)

    def test_dtype_is_respected(self):
        assert direction_from_angles(30, dtype=torch.float64).dtype == torch.float64


class TestDoaError:
    def test_identical_directions_are_zero_apart(self):
        direction = direction_from_angles(37)

        assert doa_error_degrees(direction, direction).item() == pytest.approx(0.0, abs=1e-3)

    def test_it_measures_the_angle_between_them(self):
        error = doa_error_degrees(direction_from_angles(0), direction_from_angles(30))

        assert error.item() == pytest.approx(30.0, abs=1e-3)

    def test_length_is_ignored(self):
        # SALSA normalises the EIV to unit length, so a gain error must not
        # be reported here as a pointing error
        direction = direction_from_angles(37)

        assert doa_error_degrees(direction * 17.0, direction).item() < 1e-3

    def test_opposite_directions_are_half_a_turn_apart(self):
        error = doa_error_degrees(direction_from_angles(0), direction_from_angles(180))

        assert error.item() == pytest.approx(180.0, abs=1e-2)


class TestSteeringVector:
    def test_shape_and_unit_magnitude(self):
        freqs = torch.linspace(0, 8000, 17)

        steering = steering_vector(SQUARE, direction_from_angles(30), freqs)

        assert steering.shape == (17, 4)
        assert torch.allclose(steering.abs(), torch.ones(17, 4), atol=1e-5)

    def test_at_dc_every_microphone_agrees(self):
        # at 0 Hz there is no phase difference between microphones
        steering = steering_vector(SQUARE, direction_from_angles(30), torch.zeros(1))

        assert torch.allclose(steering, torch.ones(1, 4, dtype=steering.dtype))

    def test_the_microphone_facing_the_source_leads_in_phase(self):
        """Checks the sign convention the rest of the package relies on.

        Microphone 0 of the square sits at +x, so for a source at azimuth 0
        it hears the wave first and its phase leads the array centre.
        """
        steering = steering_vector(
            SQUARE, direction_from_angles(0), torch.tensor([1000.0])
        )

        assert steering[0, 0].angle() > 0  # +x microphone, towards the source
        assert steering[0, 2].angle() < 0  # -x microphone, away from it

    def test_the_origin_of_the_coordinates_does_not_matter(self):
        freqs = torch.linspace(0, 8000, 17)
        direction = direction_from_angles(30)

        from_centre = steering_vector(SQUARE, direction, freqs)
        shifted = steering_vector(SQUARE + torch.tensor([3.0, -7.0, 2.0]), direction, freqs)

        assert torch.allclose(from_centre, shifted, atol=1e-4)


class TestPlaneWave:
    def test_shape(self):
        audio = plane_wave(SQUARE, direction_from_angles(30), tone(500.0, 12000))

        assert audio.shape == (1, 4, 12000)

    def test_every_microphone_hears_the_same_loudness(self):
        # omnidirectional microphones in a free field, so only the phase
        # differs between channels
        audio = plane_wave(SQUARE, direction_from_angles(30), tone(500.0, 12000))

        levels = audio[0, :, 2000:-2000].std(dim=-1)
        assert torch.allclose(levels, levels[0].expand(4), rtol=1e-3)

    def test_the_delays_match_the_steering_vector(self, ):
        """Checks that plane_wave and steering_vector use the same phase
        convention. If they were written separately, a sign error in both
        would cancel out and the tests would pass on wrong code.
        """
        config = StftConfig()
        direction = direction_from_angles(37)

        # a tone exactly on a bin centre, so the measured phase belongs to
        # that frequency and not to a neighbouring bin leaking in
        index = 21
        hz = config.frequencies()[index].item()

        spec = stft(plane_wave(SQUARE, direction, tone(hz, 24000)), config)
        measured = spec[0, :, index, 20]
        measured = measured / measured[0]

        expected = steering_vector(
            SQUARE, direction, torch.tensor([hz]), SPEED_OF_SOUND
        )[0]
        expected = expected / expected[0]

        assert torch.allclose(measured, expected, atol=1e-4)

    def test_a_source_straight_ahead_reaches_the_side_microphones_together(self):
        # at azimuth 0 the +y and -y microphones are the same distance from
        # the source, so there is no delay between them
        audio = plane_wave(SQUARE, direction_from_angles(0), tone(500.0, 12000))

        assert torch.allclose(audio[0, 1], audio[0, 3], atol=1e-4)


class TestIdealFoa:
    def test_shape_and_channel_meaning(self):
        signal = tone(500.0, 12000)
        direction = direction_from_angles(30)

        foa = ideal_foa(direction, signal)

        assert foa.shape == (1, 4, 12000)
        assert torch.allclose(foa[0, 0], signal)
        for channel in range(3):
            assert torch.allclose(foa[0, channel + 1], direction[channel] * signal)

    def test_it_matches_the_sn3d_steering_vector_of_equation_six(self):
        foa = ideal_foa(direction_from_angles(37, 21), tone(500.0, 12000))

        ratios = foa[0, 1:, 3000] / foa[0, 0, 3000]
        assert torch.allclose(ratios, direction_from_angles(37, 21), atol=1e-5)

    def test_it_carries_no_aliasing(self):
        """This is why ideal_foa exists next to plane_wave. There is no
        array, so the channel ratios are exact at every frequency and a
        feature test built on it cannot fail because of the geometry.
        """
        direction = direction_from_angles(30)

        for hz in (100.0, 1000.0, 9000.0):
            foa = ideal_foa(direction, tone(hz, 12000))
            ratios = foa[0, 1:, 3000] / foa[0, 0, 3000]
            assert torch.allclose(ratios, direction, atol=1e-4)


class TestArrays:
    @pytest.mark.parametrize("n_mics", [3, 4, 6, 8])
    def test_regular_polygon_is_planar_and_evenly_spread(self, n_mics):
        mics = regular_polygon(n_mics, radius=0.05)

        assert mics.shape == (n_mics, 3)
        assert torch.all(mics[:, 2] == 0)
        assert torch.allclose(mics.norm(dim=-1), torch.full((n_mics,), 0.05), atol=1e-6)
        assert mics.mean(dim=0).abs().max() < 1e-6

    def test_four_microphones_give_the_familiar_square(self):
        mics = regular_polygon(4, radius=0.05)

        expected = torch.tensor(
            [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [-0.05, 0.0, 0.0], [0.0, -0.05, 0.0]]
        )
        assert torch.allclose(mics, expected, atol=1e-8)

    def test_random_planar_stays_inside_its_disc(self):
        mics = random_planar(8, radius=0.06)

        assert mics.shape == (8, 3)
        assert torch.all(mics[:, 2] == 0)
        assert mics.norm(dim=-1).max() <= 0.06

    def test_random_planar_repeats_for_a_given_seed(self):
        assert torch.equal(random_planar(8, seed=3), random_planar(8, seed=3))
        assert not torch.equal(random_planar(8, seed=3), random_planar(8, seed=4))

    def test_random_planar_has_no_symmetry_to_lean_on(self):
        # the least-squares encoder only beats the gradient one when the
        # geometry has no symmetry, which is what this array is for
        mics = random_planar(8, radius=0.06)

        assert mics.mean(dim=0).abs().max() > 1e-4

    @pytest.mark.parametrize("bad", [{"n_mics": 0}, {"radius": 0.0}, {"radius": -1.0}])
    def test_arrays_reject_broken_settings(self, bad):
        with pytest.raises(ValueError):
            regular_polygon(**{"n_mics": 4, "radius": 0.05, **bad})

        with pytest.raises(ValueError):
            random_planar(**{"n_mics": 4, "radius": 0.05, **bad})


class TestSignals:
    def test_a_tone_lands_in_its_own_bin(self):
        config = StftConfig()
        index = 50
        hz = config.frequencies()[index].item()

        spec = stft(tone(hz, 12000).reshape(1, 1, -1), config)

        assert spec.abs().mean(dim=-1)[0, 0].argmax().item() == index

    def test_a_chirp_starts_low_and_ends_high(self):
        config = StftConfig()

        spec = stft(chirp(200.0, 6000.0, 24000).reshape(1, 1, -1), config)
        loudest = spec.abs()[0, 0].argmax(dim=0)  # [T_frames]

        assert loudest[5] < loudest[40] < loudest[-5]

    def test_a_chirp_actually_arrives_at_the_frequency_it_was_asked_for(self):
        """A chirp that sweeps to the wrong end frequency still looks like a
        chirp on a plot, so it needs checking directly. The factor of one
        half in the phase integral sets the end frequency. Without it the
        sweep runs to 2 * end_hz - start_hz instead.
        """
        config = StftConfig()
        start_hz, end_hz = 200.0, 6000.0

        spec = stft(chirp(start_hz, end_hz, 24000).reshape(1, 1, -1), config)
        loudest = spec.abs()[0, 0].argmax(dim=0)
        freqs = config.frequencies()

        # one STFT frame covers 21 ms, during which this sweep moves 123 Hz,
        # so the peak of the first and last frames is only approximately the
        # start and end frequency
        assert freqs[loudest[0]].item() == pytest.approx(start_hz, abs=150)
        assert freqs[loudest[-1]].item() == pytest.approx(end_hz, abs=150)

    def test_a_chirp_sweeps_at_a_steady_rate(self):
        # the sweep is linear, so halfway through the frequency is halfway
        # between the two ends
        config = StftConfig()

        spec = stft(chirp(200.0, 6000.0, 24000).reshape(1, 1, -1), config)
        loudest = spec.abs()[0, 0].argmax(dim=0)
        middle = config.frequencies()[loudest[len(loudest) // 2]].item()

        assert middle == pytest.approx(3100.0, abs=200)

    def test_a_chirp_of_constant_frequency_is_just_a_tone(self):
        # the two build the same phase in a different order. After half a
        # second the phase is about 1600 radians, where the spacing between
        # float32 values is already larger than 1e-4
        assert torch.allclose(
            chirp(500.0, 500.0, 12000), tone(500.0, 12000), atol=1e-3
        )

    def test_band_noise_keeps_its_energy_inside_the_band(self):
        config = StftConfig()

        spec = stft(band_noise(1000.0, 2000.0, 24000).reshape(1, 1, -1), config)
        power = spec.abs().square().mean(dim=-1)[0, 0]

        freqs = config.frequencies()
        inside = (freqs > 1100) & (freqs < 1900)
        outside = (freqs < 800) | (freqs > 2200)

        assert power[inside].mean() > 100 * power[outside].mean()

    def test_band_noise_comes_out_at_a_usable_level(self):
        # filtering removes most of the energy, so the result is rescaled.
        # Without that, two sources in different bands would not be equally
        # loud
        signal = band_noise(1000.0, 2000.0, 24000)

        assert signal.std().item() == pytest.approx(1.0, rel=0.05)

    def test_band_noise_rejects_a_backwards_band(self):
        with pytest.raises(ValueError):
            band_noise(2000.0, 1000.0, 24000)

    @pytest.mark.parametrize(
        "make",
        [
            lambda seed: white_noise(1000, seed=seed),
            lambda seed: band_noise(1000.0, 2000.0, 12000, seed=seed),
        ],
    )
    def test_noise_repeats_for_a_given_seed(self, make):
        assert torch.equal(make(3), make(3))
        assert not torch.equal(make(3), make(4))

    @pytest.mark.parametrize(
        "signal",
        [tone(500.0, 6000), chirp(200.0, 4000.0, 6000), white_noise(6000), band_noise(500.0, 1500.0, 6000)],
    )
    def test_every_generator_returns_a_plain_finite_waveform(self, signal):
        assert signal.ndim == 1
        assert signal.shape[0] == 6000
        assert torch.isfinite(signal).all()
