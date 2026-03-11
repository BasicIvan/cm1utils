"""
Virtual tower: simulations vs. observations.

Variables supported: ptsum, th_TM

The simulation calculates CO2 sum anomaly as: pt1+pt2+pt3

Usage:
    from cm1utils.workflows.virtual_tower import plot_vtower_obs_sim
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cm1utils.workflows.beromuenster_csv import get_observations
from cm1utils.tools import p
from metpy.units import units
import metpy.calc as mpcalc


PathLike = Union[str, Path]


DEFAULT_TOWER_LEVELS = [
    {"label": "12m",  "z": 12.5,  "color": "black"},
    {"label": "45m",  "z": 44.6,  "color": "tab:blue"},
    {"label": "72m",  "z": 71.5,  "color": "tab:green"},
    {"label": "132m", "z": 131.6, "color": "tab:red"},
    {"label": "212m", "z": 212.5, "color": "tab:orange"},
]

def compute_tower_potential_temperature(level, startDate: str, endDate: str):
    prs, _ = get_observations("pressure", start_date=startDate, end_date=endDate)
    tmp, _ = get_observations("temperature", start_date=startDate, end_date=endDate)

    pressure_vals = prs[level].to_numpy() * units.hPa
    temperature_vals = tmp[level].to_numpy() * units.degC

    theta = mpcalc.potential_temperature(pressure_vals, temperature_vals)
    # return a Series aligned to observation time index
    return pd.Series(np.asarray(theta), index=prs[level].index)

def plot_vtower_obs_sim(
    data_path: PathLike,
    variable: str = "ptsum",
    figure_output_dir: Optional[PathLike] = None,
    filename: str = "vt_vsOBS",
    spatial_mean: bool = True,
    subtract_baseline_ppm: float = 410.0,
    # observations:
    var_obs: str = "CO2_dry",
    obs_startDate = "2021-07-21 17:00:00",
    obs_endDate = "2021-07-23 17:00:00",
    levels: Sequence[dict] = DEFAULT_TOWER_LEVELS,
    # plotting
    title: str = "CO₂ [Δppm] at Beromünster tower — OBS vs. SIM",
    hour_tick_interval: Optional[int] = 4,
    show: bool = True,
    save: bool = True,
):
    """
    Plot observed (scatter) vs simulated (line) at multiple tower heights.
    """

    data_path = Path(data_path)
    ds1 = xr.open_dataset(data_path)

    z_levels = [lvl["z"] for lvl in levels]

    if spatial_mean:
        ds = ds1.interp(zh=z_levels).mean(dim=["xh", "yh"])
    else:
        ds = ds1.interp(xh=0.0, yh=0.0, zh=z_levels)

    if variable == "ptsum":
        targetVariable = (ds.pt1 + ds.pt2 + ds.pt3) * 1e6 - 3.0 * subtract_baseline_ppm
    else:
        targetVariable = ds[variable]

    ob, _ = get_observations(var_obs,start_date = obs_startDate, end_date = obs_endDate)
    t = pd.to_datetime(ds.time.values)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for lvl in levels:
        label = lvl["label"]
        z = lvl["z"]
        color = lvl["color"]

        # Observations (scatter)
        if variable == "ptsum":
            y_obs = ob[label] - subtract_baseline_ppm
            ylabel = "Δppm"
        elif variable.startswith("th"):
            y_obs = compute_tower_potential_temperature(label,startDate=obs_startDate, endDate=obs_endDate)
            ylabel = r"$\theta$ [K]"
        else:
            y_obs = ob[label]
            ylabel = str(variable)

        ax.scatter(
            t,
            y_obs[: len(t)],
            color=color,
            alpha=1.0,
            s=6,
        )

        # Simulation (line)
        y_sim = targetVariable.sel(zh=z)
        ax.plot(
            t,
            y_sim,
            color=color,
            alpha=0.9,
            linewidth=1.5,
            label=label,
        )

    # Legend for heights
    leg1 = ax.legend(
        title="Tower heights",
        ncol=2,
        frameon=False,
        loc="best",
    )

    # Legend for styles
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
               markersize=6, label="OBS"),
        Line2D([0], [0], color="black", linewidth=1.5, label="SIM"),
    ]
    leg2 = ax.legend(handles=legend_elements, loc="upper left", frameon=False)
    ax.add_artist(leg1)

    ax.set_title(title)
    ax.set_xlabel("time [UTC]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)

    # xaxis formatting
    p.format_time_axis_with_midnight_date(ax,hour_interval=hour_tick_interval)

    fig.tight_layout()

    if save:
        if figure_output_dir is None:
            raise ValueError("figure_output_dir must be provided when save=True")
        p.save_figure(fig, name=filename, path=figure_output_dir)

    if show:
        plt.show()

    return fig, ax


def plot_vtower_obs_only(
    data_path: PathLike,
    *,
    tower_file: str = "tower00.nc",
    figure_output_dir: Optional[PathLike] = None,
    filename: str = "OBS",
    var_obs: str = "CO2_dry",
    levels: Sequence[dict] = DEFAULT_TOWER_LEVELS,
    spatial_mean: bool = True,
    subtract_baseline_ppm: float | None = 410.0,
    obs_startDate:str = "2021-07-21 17:00:00",
    obs_endDate:str = "2021-07-23 17:00:00",
    title: str = "CO₂ [Δppm] at Beromünster tower — OBS",
    hour_tick_interval: Optional[int] = 3,
    show: bool = True,
    save: bool = True,
):
    """
    Plot observations (scatter) at multiple tower heights.
    Keeps the same plotting style/axis formatting as plot_vtower_obs_sim.
    """
    # Keep args for API compatibility, but we do not read simulation data here.
    _ = (data_path, tower_file, spatial_mean, hour_tick_interval
    )

    ob, _ = get_observations(var_obs,start_date=obs_startDate,end_date=obs_endDate)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    for lvl in levels:
        label = lvl["label"]
        color = lvl["color"]

        if var_obs == "CO2_dry":
            y_plot = ob[label] - subtract_baseline_ppm if subtract_baseline_ppm else ob[label]
            ylabel = "Δppm"
        elif var_obs == "temperature":
            y_plot = compute_tower_potential_temperature(label, obs_startDate, obs_endDate)
            ylabel = r"$\theta$ [K]"
        else:
            y_plot = ob[label]
            ylabel = str(var_obs)

        ax.scatter(
            y_plot.index,
            y_plot.to_numpy(),
            color=color,
            alpha=1.0,
            s=6,
            label=label,
        )
    # horizontal line at base level
    if subtract_baseline_ppm == None and var_obs=="CO2_dry":
        ax.axhline(410.0,linestyle="--",color="gray")

    # Legend for heights
    ax.legend(
        title="Tower heights",
        ncol=2,
        frameon=False,
        loc="best",
    )

    ax.set_title(title)
    ax.set_xlabel("time [UTC]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.4)

    # xaxis formatting (same helper)
    p.format_time_axis_with_midnight_date(ax,hour_interval=hour_tick_interval)

    fig.tight_layout()

    if save:
        if figure_output_dir is None:
            raise ValueError("figure_output_dir must be provided when save=True")
        p.save_figure(fig, filename, path=figure_output_dir)

    if show:
        plt.show()

    return fig, ax