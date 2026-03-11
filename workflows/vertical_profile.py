"""
in progress.
Functions that plot mean vertical profiles of a variable. If there are more variables, then wind is assumed.
Available:
- one-panel time-series of vertical profile on interpolated/non-interpolated grid for a variable: horizontal_wind, theta
- multiple-panel vertical profile of tracer at selected times with colored components
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional
import numpy as np
import xarray as xr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import cm1utils.tools.d as d
import cm1utils.tools.p as p
from cm1utils.tools.r import get_tke


# -----------------------
# Helper functions
# -----------------------


def select_vertical_profile(
    da: xr.DataArray,
    *,
    mean: bool = True,
    dims_to_mean: Sequence[str] = ("xh", "yh"),
) -> xr.DataArray:
    """
    Reduce a field to a vertical profile (per time): typically (time, zh).

    - If mean=True, average over horizontal dims.
    - Else select a single point (isel) at point_index for available horiz_dims.
    """
    vp = da.mean(dim=dims_to_mean,skipna=True) if mean else da.squeeze()

    if "zh" not in vp.dims and "zh" not in vp.coords:
        raise ValueError(
            f"Expected vertical coord/dim 'zh'. Got dims={vp.dims}, coords={list(vp.coords)}"
        )

    # Normalize ordering
    if "time" in vp.dims:
        vp = vp.transpose("time", "zh")
    else:
        vp = vp.transpose("zh")

    return vp


def subsample_time_by_hours(vp: xr.DataArray, hour_interval: int) -> xr.DataArray:
    """
    Subsample vp along time by approximately every `hour_interval` hours.

    Uses the numeric offset from the first timestamp. Keeps t=0 by construction.
    """
    if "time" not in vp.dims:
        raise ValueError(
            "hour_interval provided but DataArray has no 'time' dimension."
        )
    if hour_interval <= 0:
        raise ValueError("hour_interval must be a positive integer.")

    times = vp["time"].values
    if len(times) == 0:
        return vp

    # robust for numpy datetime64
    if np.issubdtype(times.dtype, np.datetime64):
        t0 = times[0]
        deltas_h = (times - t0) / np.timedelta64(1, "h")
        keep = np.isclose(deltas_h % hour_interval, 0.0, atol=1e-6) | (
            np.arange(len(times)) == 0
        )
        return vp.isel(time=np.where(keep)[0])

    # fallback: integer stride assuming 10-min model output (6 per hour)
    stride = hour_interval * 6
    return vp.isel(time=slice(None, None, stride))


def ensure_time_zh(
    da: xr.DataArray,
    *,
    zh_name: str = "zh",
) -> xr.DataArray:
    """
    Ensure DataArray is ordered (time, zh).

    - Drops any size-1 dimensions via squeeze().
    - If a remaining horizontal dim exists with size > 1, raises (profile must be 1D in z, varying in time).
    - Returns da transposed to (time, zh) (or (zh,) if time absent).

    Note: This function operates on a DataArray. If you want to compute horizontal wind from
    u_TM and v_TM, use `ensure_time_zh_from_ds(...)` below on a Dataset.
    """
    da = da.squeeze()

    # Basic dim checks
    dims = list(da.dims)
    if zh_name not in dims:
        raise ValueError(f"Expected '{zh_name}' in dims, got {da.dims}")

    # Any dims other than time and zh must be gone (or length-1 already squeezed)
    extra = [d for d in da.dims if d not in ("time", zh_name)]
    if len(extra) != 0:
        raise ValueError(
            f"Expected only dims ('time','{zh_name}') after squeeze, got extra dims: {extra} "
            f"(sizes: { {d: da.sizes[d] for d in extra} })"
        )

    # Return in (time, zh) order if time exists
    if "time" in da.dims:
        return da.transpose("time", zh_name)
    return da.transpose(zh_name)


def format_time_labels(
    vp: xr.DataArray, *, unit: str = "m", mode: str = "HH"
) -> list[str]:
    """
    Create simple labels from vp.time.

    mode:
      - "HH"     -> '17'
      - "HH:MM"  -> '17:00'
      - "full"   -> 'YYYY-MM-DDTHH:MM'
    """
    if "time" not in vp.dims:
        return []

    tvals = vp["time"].values
    if np.issubdtype(tvals.dtype, np.datetime64):
        s = [np.datetime_as_string(t, unit=unit) for t in tvals]
        if mode == "full":
            return s
        hhmm = [x.split("T")[-1] for x in s]
        if mode == "HH":
            return [x.split(":")[0] for x in hhmm]
        return hhmm

    return [f"{i}" for i in range(vp.sizes["time"])]


def build_line_colors(n: int, *, cmap=cm.Greys, lo: float = 0.25, hi: float = 0.95):
    """Return n RGBA colors sampled from a colormap."""
    if n <= 0:
        return []
    return cmap(np.linspace(lo, hi, n))


def z_km_from_zh(vp: xr.DataArray) -> np.ndarray:
    """Return vertical coordinate in km using vp['zh']."""
    return (vp["zh"] / 1000.0).to_numpy()

def get_horizontal_wind(ds: xr.Dataset, *, u_name: str = "u_TM", v_name: str = "v_TM") -> xr.DataArray:
    if u_name not in ds.data_vars or v_name not in ds.data_vars:
        raise KeyError(f"Need both '{u_name}' and '{v_name}'. Available: {list(ds.data_vars)}")

    u = ds[u_name]
    v = ds[v_name]

    hwind = np.hypot(u, v).rename("wind_h")
    hwind.attrs["long_name"] = "Horizontal wind speed"
    # preserve units if present and compatible
    if "units" in u.attrs:
        hwind.attrs["units"] = u.attrs["units"]
    hwind.attrs["computed_from"] = f"{u_name},{v_name}"
    return hwind

def get_cloud_water_gkg(
        ds: xr.Dataset
) -> xr.DataArray:
    qc = ds["qc"]
    qc_gkg = qc * 1000.0

    qc_cloudonly = qc_gkg.where(qc > 1e-6)

    qc_cloudonly.attrs["long_name"] = "Cloud water"
    qc_cloudonly.attrs["units"] = "g/kg"

    return qc_cloudonly

def convertToDataArray(obj: xr.DataArray | xr.Dataset, *, var: str | None = None) -> xr.DataArray:
    """
    If obj is a DataArray: return it.
    If obj is a Dataset:
      - if var is provided:
          - if var requests derived fields, compute them
          - else return ds[var]
      - else:
          - if TKE inputs exist, return resolved TKE
          - elif u_TM and v_TM exist, return horizontal wind
          - else raise.
    """
    if isinstance(obj, xr.DataArray):
        return obj
    if not isinstance(obj, xr.Dataset):
        raise TypeError(f"Expected xarray DataArray or Dataset, got {type(obj)}")

    ds = obj

    # Explicit request
    if var is not None:
        v = var.lower()
        if v in {"wind_h", "hwind", "horizontal_wind"}:
            return get_horizontal_wind(ds)
        if v in {"tke", "tke_res"}:
            return get_tke(ds, opt="res")
        if v in {"tke_tot", "tke_total"}:
            return get_tke(ds, opt="tot")
        if v in {"tke_ratio", "tke_sg_over_res", "tke_sg/res"}:
            return get_tke(ds, opt="ratio")
        if v == "qc":
            return get_cloud_water_gkg(ds)


        if var not in ds.data_vars:
            raise KeyError(f"'{var}' not in dataset. Available: {list(ds.data_vars)}")
        return ds[var]

    # Default: prefer TKE if available, else wind
    if {"u_u", "v_v", "w_w", "u_TM", "v_TM", "w_TM"}.issubset(ds.data_vars):
        return get_tke(ds, opt="res")

    if {"u_TM", "v_TM"}.issubset(ds.data_vars):
        return get_horizontal_wind(ds)

    raise ValueError(
        "Dataset input requires `var=...` unless it contains TKE inputs "
        "(u_u,v_v,w_w,u_TM,v_TM,w_TM) or (u_TM,v_TM) for horizontal wind."
    )

def plot_profiles_on_axis(
    ax: plt.Axes,
    vp: xr.DataArray,
    *,
    labels: Optional[Sequence[str]] = None,
    colors=None,
    linewidth: float = 1.2,
    legend_max: int = 12,
):
    """
    Plot vp on provided axis.

    vp should be (time, zh) or (zh,). x-axis is vp values; y-axis is zh [km].
    """
    z_km = z_km_from_zh(vp)

    if "time" in vp.dims:
        n = vp.sizes["time"]
        if colors is None:
            colors = build_line_colors(n)
        for i in range(n):
            lab = labels[i] if labels and i < len(labels) else None
            ax.plot(
                vp.isel(time=i).to_numpy(),
                z_km,
                color=colors[i],
                linewidth=linewidth,
                label=lab,
            )
            if i == 0:
                # put a marker for initial condition so it is more visible
                prof = vp.isel(time=i).dropna(dim="zh")
                if prof.sizes.get("zh",0) > 0:
                    ic = prof.isel(zh=0)
                    zhvalkm = ic.zh.values / 1000.
                    ax.scatter(ic.to_numpy(),
                            zhvalkm,
                            color="gray",
                            marker="*",
                            s=25,
                            zorder=2
                            )

        if labels and n <= legend_max:
            ax.legend(
                title="Time",
                fontsize=p.label_size,
                title_fontsize=p.label_size,
                frameon=False,
            )
    else:
        ax.plot(vp.to_numpy(), z_km, linewidth=linewidth)

def finalize_profile_axis(
    ax: plt.Axes,
    *,
    xlabel: str,
    title: str = "",
    grid: bool = True,
    interp: bool = False,
    xlimit: float | None = None,
    ylimit: float | None = None,
):
    """Standard axis formatting for vertical profiles."""
    if interp:
        ax.set_ylabel("z [km MSL]")
    else:
        ax.set_ylabel("z [km AGL]")
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(linewidth=0.6, linestyle=":", which="major")
        ax.grid(linewidth=0.3, linestyle=":", which="minor")

    ax.set_ylim(-0.01, ylimit)

    if xlimit:
        ax.set_xlim(-xlimit, xlimit)
        # adjust MultipleLocator to not be too dense
        major_mult = 10
        while xlimit / major_mult > 4:
            major_mult += 5
        # Major ticks every 10 units
        ax.xaxis.set_major_locator(MultipleLocator(major_mult))
        ax.tick_params(axis="x", which="major", length=4)

        # Minor ticks every 5 units
        ax.xaxis.set_minor_locator(MultipleLocator(major_mult / 2))
        ax.tick_params(axis="x", which="minor", length=2)

def plot_time_evolution_profile(
    variable: str | list,
    data_path: str,
    figure_output_dir: str | None = None,
    *,
    interp: bool = False,
    hour_interval: int | None = None,   
    mean: bool = True,
    title: str = "",
    xlabel: str = "",
    show: bool = True,
    save: bool = True,
    name: str | None = None,
    ylimit:float = None,
    xlimit:float | tuple = None,
):
    ds = d.open_data(data_path)
    try:
        #da = ds[variable]
        da = convertToDataArray(ds,var=variable)

        if hour_interval is not None:
            da = subsample_time_by_hours(da, hour_interval)

        vp = select_vertical_profile(da, mean=mean)

        vp = ensure_time_zh(vp)

        labels = format_time_labels(vp, mode="HH")  # simple hour labels like "17"
        colors = build_line_colors(vp.sizes.get("time", 1))

        fig, ax = plt.subplots(figsize=(10, 6))

        plot_profiles_on_axis(ax, vp, labels=labels, colors=colors)
        finalize_profile_axis(
            ax, xlabel=(xlabel if xlabel else variable), title=title, interp=interp
        )
        ax.set_ylim(ylimit)
        ax.set_xlim(xlimit)
        ax.axvline(0.0,linestyle="--",color="gray", linewidth=1.0)

        if save:
            out_name = name if name is not None else f"vProfile_{variable}"
            p.save_figure(fig, name=out_name, path=figure_output_dir)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax

    finally:
        try:
            ds.close()
        except Exception:
            pass

def plot_pt_components_profiles(
    data_path: str,
    figure_output_dir: str | None = None,
    *,
    hour_interval: int | None = None,
    mean: bool = False,
    baseline_ppm: float = 410.0,
    xlabel: str = "",
    ylimit: float | None = None,
    plot_between: bool = True,
    between_alpha: float = 1.0,
    show: bool = True,
    save: bool = True,
    name: str | None = None,
    figure_title: str | None = None,
    include_initial_conditions: bool = True,
):
    ds = d.open_data(data_path)
    variables = ["pt1", "pt2", "pt3"]
    if include_initial_conditions == False:
        ds = ds.isel(time=slice(1, None))

    try:
        if not all(v in ds for v in variables):
            raise ValueError(
                f"Passive tracers {variables} not found. Example vars: {list(ds.data_vars)[:20]}"
            )

        # Optional: time subsample on the Dataset
        if hour_interval is not None:
            # subsample_time_by_hours expects a DataArray; use time indices computed from any var
            tda = ds[variables[0]]
            tda_sub = subsample_time_by_hours(tda, hour_interval)
            ds = ds.isel(time=tda_sub["time"])

        # Build vertical profiles per component (keep as DataArray (time, zh))
        vp = {}
        for v in variables:
            vp[v] = select_vertical_profile(ds[v], mean=mean)  # (time, zh)
            vp[v] = vp[v] * 1e6 - baseline_ppm  # Δppm

        # Sum Δppm (time, zh)
        vp_sum = vp["pt1"] + vp["pt2"] + vp["pt3"]

        # Decide how many panels
        if "time" in vp_sum.dims:
            times = vp_sum["time"]
            n_times = vp_sum.sizes["time"]
        else:
            # No time dimension: plot single set of profiles
            times = None
            n_times = 1

        n_panels = min(8, n_times)  # because 2x4 layout
        nrows, ncols = 2, 4
        fig, axs = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(14, 6), constrained_layout=True
        )
        axs = np.array(axs).flatten()

        # search for maximum to set xlimits
        maxDeltaPPM = 0.1

        # Colors per component (single profile per panel)
        comp_colors = {
            "pt1": "purple",
            "pt2": "green",
            "pt3": "red",
            "sum": "black",
        }

        # Titles
        if times is not None:
            titles = format_time_labels(
                vp_sum.isel(time=slice(0, n_panels)), mode="HH:MM"
            )
        else:
            titles = [""]

        for i in range(n_panels):
            ax = axs[i]

            if "time" in vp_sum.dims:
                p1 = vp["pt1"].isel(time=i)
                p2 = vp["pt2"].isel(time=i)
                p3 = vp["pt3"].isel(time=i)
                ps = vp_sum.isel(time=i)
            else:
                p1 = vp["pt1"]
                p2 = vp["pt2"]
                p3 = vp["pt3"]
                ps = vp_sum

            # reference line at 0
            ax.axvline(0.0, linewidth=0.8, linestyle="-", color="0.3", zorder=0)

            # components (optionally filled)
            x1 = p1.to_numpy()
            x2 = p2.to_numpy()
            x3 = p3.to_numpy()

            # update maximum
            maxDeltaPPM = max(maxDeltaPPM, x1.max(), x2.max(), x3.max())

            # vertical values
            z_km = z_km_from_zh(vp_sum)

            ax.plot(
                x1,
                z_km,
                color=comp_colors["pt1"],
                label="GPP",
                linewidth=1.2,
                alpha=between_alpha,
            )
            ax.plot(
                x2,
                z_km,
                color=comp_colors["pt2"],
                label="RESP",
                linewidth=1.2,
                alpha=between_alpha,
            )
            ax.plot(
                x3,
                z_km,
                color=comp_colors["pt3"],
                label="Antr.",
                linewidth=1.2,
                alpha=between_alpha,
            )

            if plot_between:
                ax.fill_betweenx(
                    z_km, 0.0, x1, color=comp_colors["pt1"], alpha=between_alpha
                )
                ax.fill_betweenx(
                    z_km, 0.0, x2, color=comp_colors["pt2"], alpha=between_alpha
                )
                ax.fill_betweenx(
                    z_km, 0.0, x3, color=comp_colors["pt3"], alpha=between_alpha
                )

            # sum: line only
            ax.plot(
                ps.to_numpy(),
                z_km,
                color=comp_colors["sum"],
                label="sum",
                linewidth=2.0,
            )

            finalize_profile_axis(
                ax,
                xlabel=(xlabel if xlabel else "ΔCO₂ [ppm]"),
                title=titles[i] if titles else "",
                xlimit=maxDeltaPPM,  # you can remove this if you don't want same xlims
                ylimit=ylimit,
            )

            if i == 0:
                ax.legend(frameon=False, fontsize=p.label_size)

        # Turn off unused axes
        for j in range(n_panels, nrows * ncols):
            axs[j].axis("off")

        # set figure title
        fig.suptitle(figure_title)

        if save:
            out_name = name if name is not None else "vProfile_panels_pt"
            p.save_figure(fig, name=out_name, path=figure_output_dir)

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, axs[:n_panels]

    finally:
        try:
            ds.close()
        except Exception:
            pass
