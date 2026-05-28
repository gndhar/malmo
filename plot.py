"""
Plotting helpers

imshow_phase
imshow_mag
"""

import numpy as np
from matplotlib.axes import Axes
from matplotlib.artist import Artist


def imshow_phase(
    im: np.ndarray,
    ax: Axes,
    extent: tuple[float, float, float, float] | None = None,
    xlabel: str = "pixels",
    ylabel: str = "pixels",
    title: str | None = None,
) -> Artist:
    data = np.angle(im) if np.iscomplexobj(im) else im

    im_obj = ax.imshow(
        data,
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
        extent=extent,
        aspect="equal",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    return im_obj


def imshow_mag(
    im: np.ndarray,
    ax: Axes,
    extent: tuple[float, float, float, float] | None = None,
    xlabel: str = "pixels",
    ylabel: str = "pixels",
    title: str | None = None,
) -> Artist:
    data = np.abs(im) if np.iscomplexobj(im) else im

    im_obj = ax.imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="equal",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    return im_obj
