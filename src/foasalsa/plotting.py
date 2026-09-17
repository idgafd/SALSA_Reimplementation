"""Small matplotlib helpers shared by examples.py.

Deliberately plain. The figures use default matplotlib styling, and the only
colour choices made here are ones that carry meaning: a sequential map for
spectrogram magnitude, and a diverging map for EIV channels, which are signed
and centred on zero.
"""

from pathlib import Path

import matplotlib
import torch

# the examples write files and never open a window, so no display is needed
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


# signed data in [-1, 1], so a diverging map with white at zero. Bins that
# failed the single-source tests are exactly zero and show up as white.
EIV_COLORMAP = "RdBu_r"

MAGNITUDE_COLORMAP = "viridis"


def as_array(values):
    """Hand matplotlib a numpy array rather than a torch tensor.

    Matplotlib can consume a tensor, but it does so through a conversion path
    that numpy has deprecated, which fills the output with warnings.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()

    return values


def time_frequency(ax,
                   data,
                   config,
                   title,
                   colormap=MAGNITUDE_COLORMAP,
                   vmin=None,
                   vmax=None,
    ):
    """
    Draw one [F, T_frames] array with real units on both axes.

    Args:
        ax: Axes to draw on.
        data: [F, T_frames] tensor or array.
        config: StftConfig, used to label the axes in seconds and kHz.
        title: Short label above the panel.
        colormap, vmin, vmax: Passed to imshow.

    Returns:
        The image, so the caller can attach a colorbar.
    """
    n_frames = data.shape[-1]

    extent = [
        0.0,
        n_frames / config.frame_rate,
        0.0,
        config.sample_rate / 2000.0, # Nyquist in kHz
    ]

    image = ax.imshow(
        as_array(data),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.set_ylabel("frequency (kHz)", fontsize=8)
    ax.tick_params(labelsize=7)

    return image


def mark_frequency(ax, hz, label, row=0):
    """
    Vertical dashed line on a plot whose x axis is frequency in Hz.

    Args:
        ax: Axes to draw on.
        hz: Where to put the line.
        label: Text placed just under the top of the plot.
        row: Which line of text to use, so that several nearby markers do not
            print on top of each other.
    """
    ax.axvline(hz, color="grey", linestyle="--", linewidth=1)
    ax.annotate(
        label,
        xy=(hz, 1),
        xycoords=("data", "axes fraction"),
        xytext=(3, -12 - 11 * row),
        textcoords="offset points",
        fontsize=7,
        color="grey",
    )


def save(fig, directory, name):
    """Write a figure into `directory` and close it."""
    path = Path(directory) / name

    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)

    return path
