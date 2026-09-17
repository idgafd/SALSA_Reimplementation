"""Smoke tests for the example figures.

These do not check what the plots look like, only that every example runs to
completion and writes a file. The figures are the visible deliverable, so a
change elsewhere in the package that breaks one of them should not go
unnoticed until someone runs the command by hand.
"""

import pytest
import torch

from foasalsa.examples import EXAMPLES, bins_around, estimated_direction, main
from foasalsa.stft import StftConfig
from foasalsa.synth import direction_from_angles, ideal_foa, tone


class TestEstimatedDirection:
    def test_it_averages_the_live_bins(self):
        from foasalsa.salsa import FoaSalsa

        direction = direction_from_angles(30.0)
        foa = ideal_foa(direction, tone(500.0, 24000))

        estimate = estimated_direction(
            FoaSalsa().compute(foa), band=bins_around(500.0)
        )

        assert torch.allclose(
            estimate / estimate.norm(), direction, atol=0.02
        )

    def test_an_empty_mask_gives_zeros_rather_than_an_error(self):
        from foasalsa.salsa import FoaSalsa

        features = FoaSalsa().compute(torch.zeros(1, 4, 12000))

        assert torch.equal(estimated_direction(features), torch.zeros(3))

    def test_the_band_mask_covers_the_frequency_axis(self):
        mask = bins_around(500.0)

        assert mask.shape == (StftConfig().n_freqs,)
        assert mask.any()
        assert not mask.all()


class TestExamplesRun:
    @pytest.mark.parametrize("example", EXAMPLES, ids=lambda f: f.__name__)
    def test_one_example_writes_a_figure(self, example, tmp_path):
        path, summary = example(tmp_path)

        assert path.exists()
        assert path.stat().st_size > 10_000 # a blank png is far smaller
        assert summary

    def test_the_command_line_entry_point_writes_them_all(self, tmp_path):
        target = tmp_path / "plots"

        main([str(target)])

        written = sorted(p.name for p in target.glob("*.png"))
        assert len(written) == len(EXAMPLES)

    def test_it_creates_the_output_directory(self, tmp_path):
        target = tmp_path / "does" / "not" / "exist"

        main([str(target)])

        assert target.is_dir()

    @pytest.mark.parametrize("argv", [[], ["one", "two"]])
    def test_it_refuses_the_wrong_number_of_arguments(self, argv):
        with pytest.raises(SystemExit):
            main(argv)
