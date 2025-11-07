"""Visualization helpers for eigenmodes and spectra."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from .geometry import Domain
from .solver import EigenResult, modes_to_frequencies


_DEFAULT_CMAP = "coolwarm"


def plot_domain(domain: Domain, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    mask = np.ma.masked_where(~domain.mask, domain.mask)
    im = ax.imshow(
        mask.T,
        origin="lower",
        extent=domain.grid.extent,
        cmap="gray",
        alpha=0.8,
    )
    ax.set_title(f"Domain: {domain.name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.8, label="inside Ω")
    return ax


def plot_mode(result: EigenResult, index: int = 0, ax: plt.Axes | None = None) -> plt.Axes:
    ax = ax or plt.gca()
    mode = result.mode(index)
    cmap = _DEFAULT_CMAP
    im = ax.imshow(
        mode.T,
        origin="lower",
        extent=result.domain.grid.extent,
        cmap=cmap,
    )
    val = result.values[index]
    ax.set_title(f"Mode {index + 1}, λ={val:.4f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.colorbar(im, ax=ax, shrink=0.8)
    return ax


def plot_spectrum(result: EigenResult, ax: plt.Axes | None = None, wave_speed: float = 340.0) -> plt.Axes:
    ax = ax or plt.gca()
    idx = np.arange(1, len(result.values) + 1)
    ax.stem(idx, result.values, linefmt="C0-", markerfmt="C0o", basefmt="C0-")
    ax.set_xlabel("Mode index")
    ax.set_ylabel("Eigenvalue λ")
    ax.set_title("Eigenvalue Spectrum")

    freq_ax = ax.twinx()
    freq = modes_to_frequencies(result.values, wave_speed=wave_speed)
    freq_ax.plot(idx, freq, "C3s--", label="Frequency (Hz)")
    freq_ax.set_ylabel("Frequency (Hz)")
    return ax


def animate_mode(
    result: EigenResult,
    index: int = 0,
    duration: float = 4.0,
    fps: int = 30,
    amplitude: float = 1.0,
    save_path: str | Path | None = None,
) -> animation.ArtistAnimation:
    mode = result.mode(index)
    val = float(result.values[index])
    omega = np.sqrt(max(val, 0.0))
    times = np.linspace(0.0, duration, int(duration * fps))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = result.domain.grid.X
    Y = result.domain.grid.Y
    artists = []

    for t in times:
        amplitude_field = amplitude * np.cos(omega * t) * mode
        surf = ax.plot_surface(X, Y, amplitude_field, cmap=_DEFAULT_CMAP, linewidth=0, antialiased=True)
        artists.append([surf])

    ax.set_zlim(-amplitude, amplitude)
    ax.set_title(f"Mode {index + 1} animation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u(x, y, t)")

    anim = animation.ArtistAnimation(fig, artists, interval=1000 / fps, blit=False)
    if save_path:
        anim.save(Path(save_path), fps=fps)
    return anim


def showcase_modes(result: EigenResult, indices: Iterable[int] | None = None, cols: int = 3) -> plt.Figure:
    values = list(indices) if indices is not None else list(range(len(result.values)))
    rows = int(np.ceil(len(values) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, mode_index in enumerate(values):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis("on")
        plot_mode(result, index=mode_index, ax=ax)

    fig.tight_layout()
    return fig
