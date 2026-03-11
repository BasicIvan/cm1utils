import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def _infer_xy_dims(da: xr.DataArray) -> tuple[str, str]:
    dims = list(da.dims)
    if len(dims) != 2:
        raise ValueError(f"Expected 2D DataArray, got dims {da.dims}")
    # assume (y, x)
    return dims[1], dims[0]


def _xy_coords_km_if_meters(da: xr.DataArray, x_dim: str, y_dim: str):
    x = da.coords.get(x_dim, xr.DataArray(np.arange(da.sizes[x_dim]), dims=(x_dim,)))
    y = da.coords.get(y_dim, xr.DataArray(np.arange(da.sizes[y_dim]), dims=(y_dim,)))

    # Heuristic: if coords look like meters, plot km (matches your landuse plot behavior)
    try:
        if np.nanmax(np.abs(x.values)) > 5_000 and np.nanmax(np.abs(y.values)) > 5_000:
            return x / 1000.0, y / 1000.0, "x [km]", "y [km]"
    except Exception:
        pass
    return x, y, x_dim, y_dim


def plot_landuse_on_ax(
    ax: plt.Axes,
    ds: xr.Dataset,
    *,
    lu_name: str = "lu",
    x_name: str | None = None,
    y_name: str | None = None,
    landuse_dict: dict[int, tuple[str, str]],
    title: str | None = None,
):
    if lu_name not in ds:
        raise KeyError(
            f"Dataset has no variable '{lu_name}'. Available: {list(ds.data_vars)}"
        )

    lu = ds[lu_name].squeeze()

    if x_name is None or y_name is None:
        xd, yd = _infer_xy_dims(lu)
        x_name = x_name or xd
        y_name = y_name or yd

    if x_name not in lu.dims or y_name not in lu.dims:
        raise ValueError(
            f"Expected dims '{y_name}', '{x_name}' in '{lu_name}', got {lu.dims}"
        )

    categories = list(landuse_dict.keys())
    colors = [landuse_dict[k][1] for k in categories]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories))

    x, y, xlabel, ylabel = _xy_coords_km_if_meters(lu, x_name, y_name)

    ax.pcolormesh(
        x,
        y,
        lu.transpose(y_name, x_name),
        cmap=cmap,
        norm=norm,
        shading="auto",
        rasterized=True,
    )

    unique_indices = np.unique(lu.values)
    legend_patches = [
        Patch(
            color=landuse_dict[int(i)][1], label=f"{int(i)}: {landuse_dict[int(i)][0]}"
        )
        for i in unique_indices
        if int(i) in landuse_dict
    ]
    ax.legend(
        handles=legend_patches,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0.0,
        frameon=False,
        fontsize=8,
    )

    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = lu.attrs.get("long_name", lu_name)
    ax.set_title(title)


# ---------------------------------------
# MAIN - plot 2d map at one time instance
# ---------------------------------------
def plot_2d_from_ds(
    ds: xr.Dataset,
    var: str,
    *,
    x_name: str | None = None,
    y_name: str | None = None,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 6),
    show: bool = True,
    add_colorbar: bool = True,
    landuse_dict: dict[int, tuple[str, str]] | None = None,
    lu_name: str = "lu",
):
    """
    Plot a single 2D variable from a Dataset.

    - If var == lu_name: uses categorical landuse plot + legend (requires landuse_dict).
    - Otherwise: pcolormesh with provided cmap + optional colorbar.
    - Title is taken from DataArray.attrs['long_name'] (fallback: var).
    """
    if var not in ds.data_vars:
        raise KeyError(
            f"Variable '{var}' not in dataset. Available: {list(ds.data_vars)}"
        )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    if var == lu_name:
        if landuse_dict is None:
            raise ValueError("landuse_dict must be provided when plotting landuse.")
        plot_landuse_on_ax(
            ax,
            ds,
            lu_name=lu_name,
            x_name=x_name,
            y_name=y_name,
            landuse_dict=landuse_dict,
        )
    else:
        da = ds[var].squeeze()
        if x_name is None or y_name is None:
            xd, yd = _infer_xy_dims(da)
            x_name = x_name or xd
            y_name = y_name or yd

        if x_name not in da.dims or y_name not in da.dims:
            raise ValueError(
                f"Expected dims '{y_name}', '{x_name}' in '{var}', got {da.dims}"
            )

        x, y, xlabel, ylabel = _xy_coords_km_if_meters(da, x_name, y_name)

        im = ax.pcolormesh(
            x,
            y,
            da.transpose(y_name, x_name),
            shading="auto",
            cmap=cmap,
            rasterized=True,
        )

        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(da.attrs.get("long_name", var))

        if add_colorbar:
            cbar = fig.colorbar(im, ax=ax, shrink=0.9)
            cbar.ax.tick_params(labelsize=8)

            units = da.attrs.get("units", None)
            if units:
                cbar.set_label(f"${units}$")

    if show:
        plt.show()

    return fig, ax


def _resolve_var_3d(
    ds: xr.Dataset,
    var: str,
) -> xr.DataArray:
    """
    Return a 3D DataArray (time, z, y, x) for the requested variable.

    Special case:
      - var == "ptsum" -> pt1 + pt2 + pt3
    """
    if var == "ptsum":
        needed = ["pt1", "pt2", "pt3"]
        missing = [v for v in needed if v not in ds.data_vars]
        if missing:
            raise KeyError(
                f"'ptsum' requested, but missing variables: {missing}. "
                f"Available: {list(ds.data_vars)}"
            )
        da = (ds["pt1"] + ds["pt2"] + ds["pt3"]) * 1e6
        da.name = "ptsum"
        da.attrs["long_name"] = "ptsum"
        da.attrs["units"] = "Δppm"
        return da

    if var not in ds.data_vars:
        raise KeyError(
            f"Variable '{var}' not in dataset. Available: {list(ds.data_vars)}"
        )

    return ds[var]


def _infer_z_dim(da: xr.DataArray, x_dim: str, y_dim: str, time_dim: str) -> str:
    z_dims = [d for d in da.dims if d not in {x_dim, y_dim, time_dim}]
    if len(z_dims) != 1:
        raise ValueError(
            f"Could not infer vertical dimension uniquely for '{da.name}'. "
            f"Expected exactly one non-(time,y,x) dim, got {da.dims}"
        )
    return z_dims[0]


def plot_variability_at_height(
    ds: xr.Dataset,
    var: str,
    height: float,
    *,
    z_name: str = "zh",
    x_name: str | None = None,
    y_name: str | None = None,
    time_name: str = "time",
    cmap: str = "viridis",
    figsize: tuple[float, float] = (8, 6),
    show: bool = True,
    add_colorbar: bool = True,
    title_prefix: str = "",
    plotsTopo: bool = True,
    zs_file: str | None = None,
    topo_levels_m: int | float = 100,  # contour interval in meters
):
    """
    Plot 2D variability at a chosen height, defined as:

        variability = max(time) - min(time)

    on each (y, x) grid point.

    With removed outliers

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing a time dimension and one 3D variable.
    var : str
        Variable to plot. Special case:
          - 'ptsum' -> pt1 + pt2 + pt3 + baseline_ppm
    height : float
        Target height to select/interpolate to from z_name.
    z_name : str
        Vertical coordinate name, default 'zh'.
    x_name, y_name : str | None
        Horizontal dimension names. If omitted, inferred after height selection.
    time_name : str
        Time dimension name.
    baseline_ppm : float
        Used only when var == 'ptsum'.
    """
    da = _resolve_var_3d(ds, var)

    if time_name not in da.dims:
        raise ValueError(
            f"Variable '{var}' must have time dim '{time_name}', got {da.dims}"
        )

    # Infer x/y after selecting a single height slice if needed
    if x_name is None or y_name is None:
        if z_name in da.dims:
            sample_2d = (
                da.isel({time_name: 0})
                .sel({z_name: height}, method="nearest")
                .squeeze()
            )
        else:
            # try to infer z dim first if not named z_name
            tmp_x, tmp_y = None, None
            # infer x/y from a likely 2D slice later after z inference
            # for now assume time is present and there are 3 remaining dims
            rem = [d for d in da.dims if d != time_name]
            if len(rem) != 3:
                raise ValueError(
                    f"Expected dims (time,z,y,x)-like for '{var}', got {da.dims}"
                )
            # pick z as non-horizontal later after infer_xy_dims on final slice
            z_dim = rem[0]
            sample_2d = (
                da.isel({time_name: 0}).sel({z_dim: height}, method="nearest").squeeze()
            )

        xd, yd = _infer_xy_dims(sample_2d)
        x_name = x_name or xd
        y_name = y_name or yd

    # Infer vertical dimension if needed
    if z_name not in da.dims:
        z_name = _infer_z_dim(da, x_name, y_name, time_name)

    needed_dims = {time_name, z_name, y_name, x_name}
    if not needed_dims.issubset(set(da.dims)):
        raise ValueError(
            f"Expected dims including {needed_dims} for '{var}', got {da.dims}"
        )

    # Select requested height
    da_h = da.sel({z_name: height}, method="nearest").squeeze()

    # Variability = max - min over time
    variability = da_h.max(dim=time_name) - da_h.min(dim=time_name)

    if tuple(variability.dims) != (y_name, x_name):
        variability = variability.transpose(y_name, x_name)

    x, y, xlabel, ylabel = _xy_coords_km_if_meters(variability, x_name, y_name)

    # remove outliers
    vmin = np.nanpercentile(variability, 1)
    vmax = np.nanpercentile(variability, 99)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    im = ax.pcolormesh(
        x,
        y,
        variability,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
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
            if "zs" not in ds:
                raise ValueError(
                    "plotsTopo=True but no zs_file provided and variable 'zs' not in dataset."
                )
            topo_zs = ds["zs"].squeeze()

        # Ensure (yh, xh) for contouring
        if set(("yh", "xh")).issubset(set(topo_zs.dims)) and tuple(topo_zs.dims) != (
            "yh",
            "xh",
        ):

            topo_zs = topo_zs.transpose("yh", "xh")

        topo_np = topo_zs.to_numpy()
        maxz = float(np.nanmax(topo_np))
        if np.isfinite(maxz) and maxz > 0:
            topo_levs = np.arange(0, maxz + topo_levels_m, topo_levels_m)
            cont = ax.contour(
                x,
                y,
                topo_np,
                levels=topo_levs,
                linewidths=0.8,
                colors="black",
                alpha=0.7,
                zorder=2,
            )
            ax.clabel(cont, levels=cont.levels[::2], inline=True, fmt="%d", fontsize=6)

    z_val = float(da_h[z_name].values) if z_name in da_h.coords else float(height)

    ax.set_aspect("equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    long_name = da.attrs.get("long_name", var)
    ax.set_title(f"{title_prefix} {long_name} variability (max-min) at z={z_val:.1f} m")

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.9)
        cbar.ax.tick_params(labelsize=8)
        units = da.attrs.get("units", None)
        if units:
            cbar.set_label(f"max-min [{units}]")
        else:
            cbar.set_label("max-min")

    if show:
        plt.show()

    return fig, ax, variability
