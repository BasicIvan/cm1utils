"""
Currently available:
- vertical cross-sections (pt, theta, wind)
- horizontal cross-sections (pt components and sum), theta
"""

import os
import shutil
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
from cm1utils.tools import d
from cm1utils.workflows.virtual_tower import DEFAULT_TOWER_LEVELS

# -----------------------
# Helper functions
# -----------------------


def maybe_slice(
    ds: xr.Dataset, slice_dim: str | None, slice_index: int | None
) -> xr.Dataset:
    """
    Slice ds by index along slice_dim if that dim exists in ds.
    slice_index is 0-based.
    If ds is already sliced (dim not present), returns ds unchanged.
    """
    if slice_dim is None:
        return ds
    if slice_dim in ds.dims:
        if slice_index is None:
            raise ValueError("slice_index must be provided when slice_dim is used.")
        return ds.isel({slice_dim: slice_index})
    return ds


def ensure_zh_horiz(
    da: xr.DataArray,
    zh_name="zh",
) -> xr.DataArray:
    """
    Ensure DataArray is ordered (zh, horiz) where horiz is whatever non-time, non-zh dim remains.
    Any size-1 dimensions are dropped automatically.
    """
    da = da.squeeze()
    dims = list(da.dims)
    # drop time if present (we call this after selecting time)
    dims_wo_time = [d for d in dims if d != "time"]
    if zh_name not in dims_wo_time:
        raise ValueError(f"Expected '{zh_name}' in dims, got {da.dims}")

    horiz_dims = [d for d in dims_wo_time if d != zh_name]
    if len(horiz_dims) != 1:
        raise ValueError(
            f"Expected exactly one horizontal dim besides '{zh_name}', got {horiz_dims}"
        )

    horiz = horiz_dims[0]
    return da.transpose(zh_name, horiz)


def get_horizontal_dim(ds: xr.Dataset, zh_name="zh") -> str:
    """
    After slicing, expect dims include (time, zh, xh) OR (time, zh, yh).
    Returns "xh" or "yh".
    """
    for cand in ("xh", "yh"):
        if cand in ds.dims:
            return cand
    # some files may use x/y instead
    for cand in ("x", "y"):
        if cand in ds.dims:
            return cand
    raise ValueError(f"Could not find a horizontal dimension in {ds.dims}.")


def get_topo_profile_km(
    ds: xr.Dataset,
    zs_file: str | None,
    *,
    plots_topo: bool,
    slice_dim: str | None,
    slice_index: int | None,
    horiz_dim: str,
) -> xr.DataArray | None:
    """
    Returns 1D topo profile (km) matching horiz_dim.
    Priority:
      1) zs_file if provided
      2) ds['zs'] if present
      else error if plots_topo True
    """
    if not plots_topo:
        return None

    if zs_file is not None:
        zs_ds = xr.open_dataset(zs_file)
        zs = zs_ds["zs"]
    else:
        if "zs" not in ds:
            raise ValueError(
                "plotsTopo=True but no zs_file provided and variable 'zs' not in dataset."
            )
        zs = ds["zs"]
        zs_ds = None  # for symmetry

    # drop time if present
    d.drop_time(zs)

    # apply slice to topo if needed (only if topo still has that dim)
    if slice_dim is not None and slice_dim in zs.dims:
        zs = zs.isel({slice_dim: slice_index})

    # now zs should be 1D along horiz_dim; if not, try to reduce
    if horiz_dim in zs.dims and zs.ndim > 1:
        # if still 2D, average over the other horizontal dim
        other = [d for d in zs.dims if d != horiz_dim]
        zs = zs.mean(dim=other)

    if zs_ds is not None:
        zs_ds.close()

    return zs / 1000.0

def pick_horiz_wind(ds, fixed_dim: str):
    """
    fixed_dim:
      - "yh"  -> x-z cross-section (slice at yh=...), use u_TM
      - "xh"  -> y-z cross-section (slice at xh=...), use v_TM
    Returns (horiz_name, horiz_da, w_da) with dims ensured to be ("zh","xh").
    """
    ds = ensure_zh_horiz(ds)
    
    if fixed_dim == "yh":
        horiz_name = "v_TM"
    elif fixed_dim == "xh":
        horiz_name = "u_TM"
    else:
        raise ValueError("fixed_dim must be 'yh' or 'xh'.")

    if horiz_name not in ds or "w_TM" not in ds:
        return None, None, None

    h = ds[horiz_name]
    w = ds["w_TM"]

    return horiz_name, h, w


def _pick_var(ds: xr.Dataset, base: str) -> xr.DataArray:
    for cand in (f"{base}_TM", base):
        if cand in ds:
            return ds[cand]
    raise ValueError(f"Could not find '{base}_TM' or '{base}' in dataset variables.")

def _compute_tracer_sum(ds: xr.Dataset, *, baseline_ppm: float = 410.0) -> xr.DataArray:
    pt1 = _pick_var(ds, "pt1")
    pt2 = _pick_var(ds, "pt2")
    pt3 = _pick_var(ds, "pt3")
    return (pt1 + pt2 + pt3) * 1e6 - 3.0 * baseline_ppm

# -----------------------
# MAIN -- Vertical cross-sect.
# -----------------------
def make_vcs_gif(
    input_file: str,
    out_gif_path: str,
    *,
    missing_value=-999999.875,
    tmp_dir: str = "vcs_frames",
    step: int = 1,
    # slicing: choose one of slice_dim={"yh","xh"} and provide 0-based index
    # slice_dim is applied in ensure_zh_horiz if data has two horizontal dimensions
    topo_slice_dim: str | None = "yh",
    topo_slice_index: int | None = 150,
    # plotting options
    plotsTheta: bool = True,
    plotsTopo: bool = True,
    plotsQuiver: bool = True,
    zs_file: str | None = None,
    rz_km: tuple[float, float] = (0.0, 6.0),
    title_prefix: str = "",
    # tracer baseline
    baseline_ppm: float = 410.0,
    # quiver settings
    quiver_x_stride: int = 10,
    quiver_z_stride: int = 20,
    quiver_scale: float = 5.0,
    # gif settings
    duration_ms: float | int = 500,
    loop: int = 1,
):
    """
    Create an animated GIF of a CM1 vertical cross-section time series.

    The function reads a CM1 NetCDF (via `cm1utils.tools.d.open_data`), optionally slices the 3D
    fields along one horizontal dimension (e.g., fix `yh` to get an x–z section, or fix `xh` to
    get a y–z section), and then iterates over time to generate PNG frames which are assembled
    into a GIF.

    Each frame shows:
      - Filled contours of the summed passive tracer concentration anomaly (Δppm), computed as
        (pt1 + pt2 + pt3) * 1e6 - 3 * baseline_ppm, using either {pt1_TM, pt2_TM, pt3_TM} or
        {pt1, pt2, pt3}.
      - Optional potential temperature (th_TM) contour overlays.
      - Optional wind vectors (u_TM or v_TM for the horizontal component, plus w_TM) plotted
        as quivers, subsampled by `quiver_x_stride` and `quiver_z_stride`.
      - Optional terrain profile (zs) filled and outlined, taken from `zs_file` if provided,
        otherwise from `ds['zs']`.

    Frames are written into `tmp_dir` and removed after the GIF is created.

    Parameters
    ----------
    input_file : str
        Path to the CM1 NetCDF file containing time and cross-section variables.
    out_gif_path : str
        Output path for the generated GIF.
    missing_value : float, optional
        Value treated as missing and masked prior to plotting.
    tmp_dir : str, optional
        Temporary directory used to store PNG frames before GIF assembly.
    step : int, optional
        Time stride for frames (e.g., step=2 uses every second time index).
    slice_dim : {"yh","xh",None}, optional
        Dimension to slice (fix) to produce a 2D vertical cross-section. If None, no slicing is
        applied and the dataset must already be 2D in the horizontal.
    slice_index : int, optional
        0-based index used for slicing along `slice_dim` when that dimension exists.
    plotsTheta : bool, optional
        If True and `th_TM` is present, overlay theta contours.
    plotsTopo : bool, optional
        If True, plot terrain profile from `zs_file` or `ds['zs']`.
    plotsQuiver : bool, optional
        If True and required wind variables exist, overlay wind vectors.
    zs_file : str or None, optional
        Optional external NetCDF containing `zs` for topography. Useful if `zs` is not in
        `input_file` or if you want a dedicated terrain source.
    rz_km : (float, float), optional
        Requested z-limits in km (currently not enforced; the plot uses the dataset top height).
    title_prefix : str, optional
        Prefix added to each frame title (time string is appended).
    baseline_ppm : float, optional
        Baseline ppm used to form the tracer anomaly (Δppm) from the summed tracer mixing ratios.
    quiver_x_stride : int, optional
        Horizontal subsampling stride for quiver arrows.
    quiver_z_stride : int, optional
        Vertical subsampling stride for quiver arrows.
    quiver_scale : float, optional
        Matplotlib quiver scale (smaller values produce longer arrows).

    Returns
    -------
    None
        Writes `out_gif_path` and prints the output path on completion.

    Raises
    ------
    ValueError
        If required indices/variables are missing (e.g., slicing requested without `slice_index`,
        tracer variables not found, or topography requested but unavailable).
    RuntimeError
        If no PNG frames are generated (e.g., empty time dimension or I/O issues).
    """

    os.makedirs(tmp_dir, exist_ok=True)

    # Load + mask
    ds_all = d.open_data(input_file)

    # Optional slice (if ds )
    # ds_all = maybe_slice(ds_all, slice_dim=slice_dim, slice_index=slice_index)

    # Identify dims and coords
    horiz_dim = get_horizontal_dim(ds_all, zh_name="zh")

    # Coordinates in km (these are 1D)
    z_km = ds_all["zh"] / 1000.0
    x_km = ds_all[horiz_dim] / 1000.0
    rx_label = "x [km]" if horiz_dim in ("xh", "x") else "y [km]"

    # Topography profile (optional)
    zs_km = get_topo_profile_km(
        ds_all,
        zs_file,
        plots_topo=plotsTopo,
        slice_dim=topo_slice_dim,
        slice_index=topo_slice_index,
        horiz_dim=horiz_dim,
    )

    # Colormap specs
    specs = {
        "title": "Passive Tracer Conc.",
        "cmap": plt.get_cmap("RdBu_r"),
        "levels": np.linspace(-30, 150, 37),
        "extend": "both",
        "norm": mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=150),
    }

    ntime = ds_all.sizes.get("time", 1)

    # Frame loop
    for ti in range(0, ntime, step):
        ds = ds_all.isel(time=ti) if "time" in ds_all.dims else ds_all

        fig, ax = plt.subplots(figsize=(10, 6))

        # ----- passive tracer sum (Δppm) -----
        ds = ds.where(ds != missing_value)
        vr = _compute_tracer_sum(ds, baseline_ppm=baseline_ppm)
        vr = ensure_zh_horiz(vr, zh_name="zh")

        c = ax.contourf(
            x_km,
            z_km,
            vr,  # xarray is fine
            cmap=specs["cmap"],
            norm=specs["norm"],
            levels=specs["levels"],
            extend=specs["extend"],
            zorder=1,
        )

        # ----- optional: theta overlay -----
        if plotsTheta and ("th_TM" in ds):
            theta = ensure_zh_horiz(ds["th_TM"], zh_name="zh")
            cth = ax.contour(
                x_km,
                z_km,
                theta,
                levels=np.arange(250, 350, 1.0),
                colors="grey",
                linewidths=0.5,
                zorder=2,
            )
            ax.clabel(cth, fmt="%.0f", inline=True, fontsize=6)

        # ----- optional: quiver -----
        hname, hwind, wwind = pick_horiz_wind(ds, horiz_dim)
        if plotsQuiver and hwind is not None:
            # quiver is fussy: convert to numpy explicitly
            ax.quiver(
                x_km[::quiver_x_stride],
                z_km[::quiver_z_stride],
                hwind.values[::quiver_z_stride, ::quiver_x_stride],
                wwind.values[::quiver_z_stride, ::quiver_x_stride],
                angles="xy",
                color="black",
                scale=quiver_scale,  # smaller -> longer arrows
                scale_units="xy",
                width=0.003,
                alpha=0.9,
                zorder=3,
                pivot="mid",
            )

        # ----- optional: orography -----
        if plotsTopo and (zs_km is not None):
            # Ensure numpy for fill_between
            ax.fill_between(
                x_km.to_numpy(), zs_km.to_numpy(), y2=0, color="lightgray", zorder=0
            )
            ax.plot(
                x_km.to_numpy(), zs_km.to_numpy(), color="black", linewidth=1, zorder=4
            )

        # ax.set_ylim(rz_km)
        ax.set_ylim(0.0, z_km.values[-1])
        ax.set_ylabel("z [km]")
        ax.set_xlabel(rx_label)
        fig.colorbar(c, ax=ax, orientation="vertical")

        # Title time handling: no timezone modifications; format robustly
        if "time" in ds_all.coords:
            t = ds_all.time.values[ti]
            t_str = np.datetime_as_string(t, unit="m")  # YYYY-MM-DDTHH:MM
        else:
            t_str = f"t={ti}"

        full_title = f"{title_prefix}{t_str}"
        ax.set_title(full_title)

        out_png = os.path.join(tmp_dir, f"{ti:06d}.png")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    ds_all.close()

    # Build GIF
    frame_files = sorted(
        [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".png")]
    )
    if not frame_files:
        raise RuntimeError("No frames generated.")

    images = [Image.open(p) for p in frame_files]
    images[0].save(
        out_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
    )

    # cleanup
    shutil.rmtree(tmp_dir)

    print(f"Wrote GIF: {out_gif_path}")


# -----------------------
# MAIN -- Horizontal cross-sect. for passive tracer (CO2) panels
# -----------------------
def make_hcs_gif(
    input_file: str,
    out_gif_path: str,
    *,
    tmp_dir: str = "hcs_frames",
    step: int = 1,
    # vertical level selection (0-based index into zh)
    zh_index: int = 0,
    # missing/masking
    missing_value=-999999.875,
    # plotting toggles
    plotsTheta: bool = False,
    theta_name: str = "th_TM",
    theta_sigma: float = 1.0,  # smoothing sigma for theta anomaly
    plotsTopo: bool = True,
    topo_levels_m: int | float = 100,  # contour interval in meters
    zs_file: str | None = None,
    baseline_ppm: float = 410.0,
    # panel specs: either a dict (shared) or list/tuple of 4 dicts (pt1, pt2, pt3, sum)
    specs=None,
    # labels/titles
    suptitle_suffix: str = " LT",
    title_prefix: str = "",
    # gif settings
    duration_ms: int = 500,
    loop: int = 1,
):
    """
    Create an animated GIF of 2x2 horizontal cross-sections (pt1, pt2, pt3, sum) at a fixed zh level.

    Reads a CM1 NetCDF (via `cm1utils.tools.d.open_data`), selects `zh_index`, then loops over time
    to produce 2x2 panels:
      - pt1 anomaly (ppm): pt1*1e6 - baseline_ppm
      - pt2 anomaly (ppm): pt2*1e6 - baseline_ppm
      - pt3 anomaly (ppm): pt3*1e6 - baseline_ppm
      - sum anomaly (ppm): (pt1+pt2+pt3)*1e6 - 3*baseline_ppm

    Optional overlays:
      - theta anomaly contours (relative to time=0), smoothed with a Gaussian filter
      - orography contours from `zs_file` or from `ds['zs']`

    Frames are written into `tmp_dir` and removed after the GIF is created.
    """
    import os
    import shutil
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from PIL import Image

    # Optional dependency only if theta overlay is used
    try:
        from scipy.ndimage import gaussian_filter
    except Exception:
        gaussian_filter = None

    def panel_spec(local_specs, i):
        # Accept a single dict (shared) or a list/tuple of 4 dicts (per-panel)
        if isinstance(local_specs, (list, tuple)):
            return local_specs[i]
        return local_specs

    # Default specs if none provided (shared)
    if specs is None:
        specs = {
            "cmap": plt.get_cmap("RdBu_r"),
            "levels": np.linspace(-30, 150, 37),
            "extend": "both",
            "norm": mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=150),
            "cbar_label": "ΔCO₂ [ppm]",
        }

    os.makedirs(tmp_dir, exist_ok=True)

    # Load
    ds_all = d.open_data(input_file)

    # Select zh level if present
    if "zh" in ds_all.dims:
        ds_all = ds_all.isel(zh=zh_index)

    # Mask missing values (only where possible)
    ds_all = ds_all.where(ds_all != missing_value)

    # Coordinates [km]
    if "xh" not in ds_all.coords or "yh" not in ds_all.coords:
        raise ValueError(
            f"Expected xh/yh coordinates for HCS, got coords={list(ds_all.coords)}"
        )

    x_km = (ds_all["xh"] / 1000.0).to_numpy()
    y_km = (ds_all["yh"] / 1000.0).to_numpy()
    X, Y = np.meshgrid(x_km, y_km)

    rx = (float(x_km[0]), float(x_km[-1]))
    ry = (float(y_km[0]), float(y_km[-1]))

    # Topography field for contours (2D)
    topo_zs = None
    if plotsTopo:
        if zs_file is not None:
            zs_ds = xr.open_dataset(zs_file)
            if "zs" not in zs_ds:
                zs_ds.close()
                raise ValueError("zs_file provided but variable 'zs' not found.")
            topo_zs = zs_ds["zs"].squeeze()
            zs_ds.close()
        else:
            if "zs" not in ds_all:
                raise ValueError(
                    "plotsTopo=True but no zs_file provided and variable 'zs' not in dataset."
                )
            topo_zs = ds_all["zs"].squeeze()
            

        # Ensure (yh, xh) for contouring
        if set(("yh", "xh")).issubset(set(topo_zs.dims)) and tuple(topo_zs.dims) != (
            "yh",
            "xh",
        ):
            
            topo_zs = topo_zs.transpose("yh", "xh")

    # Theta baseline (time=0), only if requested
    theta0 = None
    if plotsTheta:
        if theta_name not in ds_all:
            raise ValueError(
                f"plotsTheta=True but '{theta_name}' not found in dataset."
            )
        if gaussian_filter is None:
            raise ImportError(
                "plotsTheta=True requires scipy (scipy.ndimage.gaussian_filter)."
            )

        if "time" in ds_all.dims:
            theta0 = ds_all[theta_name].isel(time=0)
        else:
            theta0 = ds_all[theta_name]

        # Ensure (yh, xh)
        if tuple(theta0.dims) != ("yh", "xh"):
            theta0 = theta0.transpose("yh", "xh")

    ntime = ds_all.sizes.get("time", 1)

    # Frame loop
    for ti in range(0, ntime, step):
        ds = ds_all.isel(time=ti) if "time" in ds_all.dims else ds_all

        # Tracers (2D), ensure (yh, xh)
        pt1 = _pick_var(ds, "pt1")
        pt2 = _pick_var(ds, "pt2")
        pt3 = _pick_var(ds, "pt3")

        if tuple(pt1.dims) != ("yh", "xh"):
            pt1 = pt1.transpose("yh", "xh")
        if tuple(pt2.dims) != ("yh", "xh"):
            pt2 = pt2.transpose("yh", "xh")
        if tuple(pt3.dims) != ("yh", "xh"):
            pt3 = pt3.transpose("yh", "xh")

        # Build panels (ppm anomalies)
        data_panels = [
            pt1 * 1e6 - baseline_ppm,
            pt2 * 1e6 - baseline_ppm,
            pt3 * 1e6 - baseline_ppm,
            (pt1 + pt2 + pt3) * 1e6 - 3.0 * baseline_ppm,
        ]
        panel_titles = ["GPP (pt1)", "Resp. (pt2)", "Anthrop. CO₂ (pt3)", "Sum"]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        axs = axs.flatten()

        for i, (ax, da, ptitle) in enumerate(zip(axs, data_panels, panel_titles)):
            sp = panel_spec(specs, i)

            c = ax.contourf(
                X,
                Y,
                da.to_numpy(),
                cmap=sp["cmap"],
                norm=sp.get("norm", None),
                levels=sp["levels"],
                extend=sp.get("extend", "neither"),
                zorder=1,
            )

            # Theta overlay (anomaly vs time=0), smoothed
            if plotsTheta:
                theta = ds[theta_name]
                if tuple(theta.dims) != ("yh", "xh"):
                    theta = theta.transpose("yh", "xh")

                theta_anom = (theta - theta0).to_numpy()
                theta_smooth = gaussian_filter(theta_anom, sigma=theta_sigma)

                vmin = float(np.nanmin(theta_smooth))
                vmax = float(np.nanmax(theta_smooth))
                if np.isfinite(vmin) and np.isfinite(vmax) and (vmax - vmin) > 0.45:
                    levels = np.linspace(vmin, vmax, num=5)
                    ax.contour(
                        X,
                        Y,
                        theta_smooth,
                        levels=levels,
                        cmap="coolwarm",
                        linewidths=0.8,
                        zorder=3,
                    )

            # Orography contours
            if plotsTopo and topo_zs is not None:
                topo_np = topo_zs.to_numpy()
                maxz = float(np.nanmax(topo_np))
                if np.isfinite(maxz) and maxz > 0:
                    levs = np.arange(0, maxz + topo_levels_m, topo_levels_m)
                    cont = ax.contour(
                        X,
                        Y,
                        topo_np,
                        levels=levs,
                        linewidths=0.8,
                        colors="black",
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.clabel(
                        cont, levels=cont.levels[::2], inline=True, fmt="%d", fontsize=6
                    )

            ax.set_xlim(rx)
            ax.set_ylim(ry)
            ax.set_title(ptitle)
            ax.set_xlabel("x [km]")
            ax.set_ylabel("y [km]")

            # Per-panel colorbar only if per-panel specs
            if isinstance(specs, (list, tuple)):
                cb = fig.colorbar(
                    c, ax=ax, orientation="vertical", fraction=0.046, pad=0.02
                )
                if "cbar_label" in sp:
                    cb.set_label(sp["cbar_label"])

        # Title time handling (robust, no timezone changes)
        if "time" in ds_all.coords:
            t = ds_all.time.values[ti]
            t_str = np.datetime_as_string(t, unit="m")  # YYYY-MM-DDTHH:MM
        else:
            t_str = f"t={ti}"

        fig.suptitle(f"{title_prefix}{t_str}{suptitle_suffix}")

        out_png = os.path.join(tmp_dir, f"{ti:06d}.png")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    ds_all.close()

    # Build GIF
    frame_files = sorted(
        [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".png")]
    )
    if not frame_files:
        raise RuntimeError("No frames generated.")

    images = [Image.open(p) for p in frame_files]
    images[0].save(
        out_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
    )

    shutil.rmtree(tmp_dir)
    print(f"Wrote GIF: {out_gif_path}")

# -----------------------
# MAIN -- Horizontal cross-sect. - virtual towers over the whole domain - PTSUM
# -----------------------
def make_hcs_vtower_gif(
    input_file: str,
    out_gif_path: str,
    *,
    levels=DEFAULT_TOWER_LEVELS,
    zh_name: str = "zh",
    x_name: str = "xh",
    y_name: str = "yh",
    baseline_ppm: float = 410.0,
    missing_value: float = -999999.875,
    tmp_dir: str = "vtower_domain_frames",
    step: int = 1,
    # shared tracer-sum color scale
    vmin: float = -30.0,
    vmax: float = 100.0,
    levels_count: int = 37,
    cmap: str = "RdBu_r",
    duration_ms: int = 500,
    loop: int = 1,
    title_prefix: str = "",
    suptitle_suffix: str = "",
    plotsTopo:bool=True,
    zs_file:str|None=None,
    topo_levels_m: int | float = 100,  # contour interval in meters
):
    """
    GIF of 5-panel horizontal cross-sections of tracer-sum anomaly (Δppm)
    at the Beromünster tower sampling heights (AGL).

    Plan implemented:
      1) load full 3D dataset (time, zh, yh, xh) with tracer components
      2) compute tracer sum and interpolate to sampling heights
      3) render frames (2x3 layout; 6th panel hidden)
      4) assemble into GIF

    Uses ONE shared colorbar for all 5 panels.
    """

    os.makedirs(tmp_dir, exist_ok=True)

    ds_all = d.open_data(input_file)

    # Basic checks
    for req in (zh_name, x_name, y_name):
        if (req not in ds_all.dims) and (req not in ds_all.coords):
            ds_all.close()
            raise ValueError(
                f"Expected '{req}' in dims/coords, got dims={ds_all.dims}, coords={list(ds_all.coords)}"
            )

    z_targets = [lvl["z"] for lvl in levels]
    z_labels = [lvl["label"] for lvl in levels]

    # Mask missing values (best effort)
    ds_all = ds_all.where(ds_all != missing_value)

    # 1) tracer sum on full 3D field
    tracer_sum = _compute_tracer_sum(ds_all, baseline_ppm=baseline_ppm)

    if zh_name not in tracer_sum.dims:
        ds_all.close()
        raise ValueError(f"Tracer sum has no '{zh_name}' dim; dims={tracer_sum.dims}")

    # 2) interpolate to tower heights (AGL)
    tracer_sum_i = tracer_sum.interp({zh_name: z_targets})

    # coords [km]
    x_km = (ds_all[x_name] / 1000.0).to_numpy()
    y_km = (ds_all[y_name] / 1000.0).to_numpy()
    X, Y = np.meshgrid(x_km, y_km)
    rx = (float(x_km[0]), float(x_km[-1]))
    ry = (float(y_km[0]), float(y_km[-1]))

    # Topography field for contours (2D)
    topo_zs = None
    if plotsTopo:
        if zs_file is not None:
            zs_ds = xr.open_dataset(zs_file)
            if "zs" not in zs_ds:
                zs_ds.close()
                raise ValueError("zs_file provided but variable 'zs' not found.")
            topo_zs = zs_ds["zs"].squeeze()
            zs_ds.close()
        else:
            if "zs" not in ds_all:
                raise ValueError(
                    "plotsTopo=True but no zs_file provided and variable 'zs' not in dataset."
                )
            topo_zs = ds_all["zs"].squeeze()
            

        # Ensure (yh, xh) for contouring
        if set(("yh", "xh")).issubset(set(topo_zs.dims)) and tuple(topo_zs.dims) != (
            "yh",
            "xh",
        ):
            
            topo_zs = topo_zs.transpose("yh", "xh")

    # shared color scale
    color_levs = np.linspace(vmin, vmax, levels_count)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap_obj = plt.get_cmap(cmap)

    ntime = ds_all.sizes.get("time", 1)

    # 3) frames
    for ti in range(0, ntime, step):
        da_t = tracer_sum_i.isel(time=ti) if "time" in tracer_sum_i.dims else tracer_sum_i

        fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        axs = np.asarray(axs).ravel()

        mappable = None

        for pi, (z_lab, z_val) in enumerate(zip(z_labels, z_targets)):
            ax = axs[pi]

            slab = da_t.sel({zh_name: z_val}, method="nearest").squeeze()

            # enforce (yh, xh) for plotting against X,Y
            if set((y_name, x_name)).issubset(slab.dims) and tuple(slab.dims) != (y_name, x_name):
                slab = slab.transpose(y_name, x_name)

            mappable = ax.contourf(
                X,
                Y,
                slab.to_numpy(),
                cmap=cmap_obj,
                norm=norm,
                levels=color_levs,
                extend="both",
                zorder=1,
            )

            # Orography contours
            if plotsTopo and topo_zs is not None:
                topo_np = topo_zs.to_numpy()
                maxz = float(np.nanmax(topo_np))
                if np.isfinite(maxz) and maxz > 0:
                    topo_levs = np.arange(0, maxz + topo_levels_m, topo_levels_m)
                    cont = ax.contour(
                        X,
                        Y,
                        topo_np,
                        levels=topo_levs,
                        linewidths=0.8,
                        colors="black",
                        alpha=0.7,
                        zorder=2,
                    )
                    ax.clabel(
                        cont, levels=cont.levels[::2], inline=True, fmt="%d", fontsize=6
                    )

            ax.set_xlim(rx)
            ax.set_ylim(ry)
            ax.set_aspect("equal")
            ax.set_title(f"{z_lab}")
            ax.set_xlabel("x [km]")
            ax.set_ylabel("y [km]")

        # hide unused 6th panel
        axs[5].set_visible(False)

        # one shared colorbar
        if mappable is not None:
            cb = fig.colorbar(mappable, ax=axs[:5], orientation="vertical", shrink=0.9)
            cb.set_label("ΔCO₂ [ppm]")

        # title
        if "time" in ds_all.coords:
            t = ds_all.time.values[ti]
            t_str = np.datetime_as_string(t, unit="m")
        else:
            t_str = f"t={ti}"
        fig.suptitle(f"{title_prefix}{t_str}{suptitle_suffix}")

        out_png = os.path.join(tmp_dir, f"{ti:06d}.png")
        fig.savefig(out_png, bbox_inches="tight")
        plt.close(fig)

    ds_all.close()

    # 4) GIF assembly
    frame_files = sorted(
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir) if f.endswith(".png")
    )
    if not frame_files:
        raise RuntimeError("No frames generated.")

    images = [Image.open(p) for p in frame_files]
    images[0].save(
        out_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
    )

    shutil.rmtree(tmp_dir)
    print(f"Wrote GIF: {out_gif_path}")