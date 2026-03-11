"""
Plot surface sensible/latent heat flux time series (domain mean)
and optionally the Bowen ratio.

Usage from notebook:
    from cm1utils.workflows.plot_flx_bowen import plot_surface_fluxes
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import cm1utils.tools.constants as C
import cm1utils.tools.p as p


PathLike = Union[str, Path]


def plot_surface_fluxes(
    data_path: PathLike,
    figure_output_dir: Optional[PathLike] = None,
    figure_name: str = "fluxes",
    title: str = "",
    hour_interval: int = 4,
    ylim_flux: Optional[Tuple[float, float]] = None,
    ylim_bowen: Optional[Tuple[float, float]] = None,
    plots_bowen: bool = True,
    show: bool = True,
    save: bool = True,
):
    """
    Plot domain-mean sensible (hfx) and latent (qfx*LV) heat flux.
    Optionally plot Bowen ratio (H / L) in a second panel.

    Returns
    -------
    fig, axes, hf, lf, bowen
    """

    fluxes = xr.open_dataset(data_path)

    hf = fluxes.hfx.mean(dim=("xh", "yh"))
    lf = (fluxes.qfx * C.LV).mean(dim=("xh", "yh"))

    # Avoid division by zero
    bowen = xr.where(np.abs(lf) > 1e-12, hf / lf, np.nan)

    if plots_bowen:
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 7),
            sharex=True,
        )
        ax_flux, ax_bowen = axes
    else:
        fig, ax_flux = plt.subplots(figsize=(10, 4))
        ax_bowen = None

    # --- Flux panel ---
    hf.plot(ax=ax_flux, label="SHF", color="red")
    lf.plot(ax=ax_flux, label="LHF", color="blue")

    ax_flux.grid(True)
    ax_flux.legend()
    ax_flux.set_ylabel(r"$W\,m^{-2}$")

    if title:
        ax_flux.set_title(title)

    if ylim_flux is not None:
        ax_flux.set_ylim(*ylim_flux)

    # --- Bowen panel ---
    if plots_bowen:
        bowen.plot(ax=ax_bowen, color="black", label="Bowen ratio")
        ax_bowen.grid(True)
        ax_bowen.set_ylabel("Bowen ratio")
        ax_bowen.legend()

        if ylim_bowen is not None:
            ax_bowen.set_ylim(*ylim_bowen)

    # --- Time axis formatting (bottom axis only) ---
    target_ax = ax_bowen if plots_bowen else ax_flux

    target_ax.xaxis.set_major_locator(mdates.HourLocator(interval=hour_interval))
    target_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H"))
    target_ax.xaxis.set_minor_locator(mdates.DayLocator())
    target_ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%d-%b"))
    target_ax.set_xlabel("time [UTC]")

    fig.autofmt_xdate(rotation=0)

    if plots_bowen:
        import matplotlib.ticker as mticker

        ax_bowen.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax_bowen.axhline(0, linestyle="--", color="gray")

    if save:
        if figure_output_dir is None:
            raise ValueError("figure_output_dir must be provided when save=True")
        p.save_figure(fig, name=figure_name, path=Path(figure_output_dir))

    if show:
        plt.show()

    ax_out = axes if plots_bowen else ax_flux
    return fig, ax_out, hf, lf, bowen
