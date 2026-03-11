""" CALCULATION TOOLS - IVAN BASIC
"""

import numpy as np
import xarray as xr

def get_gauss_filtered_field(field, sigma=10.0, truncate=4.0):
    """
    Mask-aware horizontal Gaussian filter across (yh, xh), broadcasting over
    any other dims (e.g., zh, time, or both). Original NaNs are preserved.

    Supported dims:
      ('yh','xh'), ('zh','yh','xh'), ('time','yh','xh'), ('time','zh','yh','xh')
    """
    import numpy as np
    import xarray as xr
    from scipy.ndimage import gaussian_filter

    if not {'yh', 'xh'}.issubset(field.dims):
        raise ValueError("Field must contain 'yh' and 'xh' dimensions.")

    # Work in float for safety; remember original dtype to cast back later
    orig_dtype = np.float32 if field.dtype == np.float32 else np.float64

    # Data with NaNs -> 0 for numerator; weights are 1 for valid, 0 for NaN
    data = field.fillna(0).astype(orig_dtype)
    wts  = field.notnull().astype(orig_dtype)

    filter_kwargs = dict(sigma=sigma, truncate=truncate, mode="wrap")

    # Apply gaussian_filter over (yh,xh), vectorized over remaining dims; dask-friendly
    num = xr.apply_ufunc(
        gaussian_filter,
        data,
        kwargs=filter_kwargs,
        input_core_dims=[["yh", "xh"]],
        output_core_dims=[["yh", "xh"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[orig_dtype],
    )
    den = xr.apply_ufunc(
        gaussian_filter,
        wts,
        kwargs=filter_kwargs,
        input_core_dims=[["yh", "xh"]],
        output_core_dims=[["yh", "xh"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[orig_dtype],
    )

    # Weighted average: only valid neighbors contribute
    eps = np.finfo(orig_dtype).eps
    filtered = xr.where(den > eps, num / den, np.nan)

    # Preserve original NaNs strictly (keep masked regions as NaN)
    filtered = filtered.where(field.notnull())

    return filtered.astype(orig_dtype)

def diagnose_ABL_height_tracer(
    tracer_da,
    z_da,
    *,
    surface_exclude=5,
    window=2,
    tolerance=0.05,
    return_agl=False,
    zs_data_path:str|None=None,
):
    """
    Adaptive tracer-based BL height using a sliding mean and a fractional drop criterion.

    Parameters
    ----------
    tracer_da : xr.DataArray
        Dimensions expected: ('time', 'zh', 'yh', 'xh').
    z_da : xr.DataArray or np.ndarray
        Vertical coordinate with dim ('zh',) or shape (zh,).
        Interpreted as height above sea level (ASL).
    surface_exclude : int
        Number of lowest levels to skip (from surface).
    window : int
        Vertical rolling-mean window size along 'zh'.
    tolerance : float
        Fractional drop threshold relative to the mixed-layer mean.
    return_agl : bool, optional
        If False (default), return BL height above sea level (ASL).
        If True, subtract zs from zs.nc and return BL height above ground level (AGL).

    Returns
    -------
    bl_height : xr.DataArray
        Boundary-layer height with dims ('time','yh','xh').
        ASL by default; AGL if return_agl=True.
    """
    assert {'time','zh','yh','xh'}.issubset(set(tracer_da.dims)), \
        "tracer_da must have dims ('time','zh','yh','xh')"

    # vertical coordinate
    z_vals = z_da.values if isinstance(z_da, xr.DataArray) else np.asarray(z_da)
    z_core = xr.DataArray(
        z_vals[surface_exclude:],
        dims=['zh'],
        coords={'zh': tracer_da.zh.isel(zh=slice(surface_exclude, None))}
    )

    # exclude lowest levels
    tracer_core = tracer_da.isel(zh=slice(surface_exclude, None))  # (time, zh, yh, xh)

    # rolling mean over zh
    cum_mean = tracer_core.rolling(zh=window, min_periods=window).mean()

    # adaptive mixed-layer mean
    max_cum = cum_mean.max(dim='zh')                       # (time, yh, xh)
    valid_mask = cum_mean > (max_cum * (1 - tolerance))    # (time, zh, yh, xh)
    mixed_layer_mean = cum_mean.where(valid_mask).mean(dim='zh', skipna=True)

    # drop threshold
    drop_threshold = mixed_layer_mean * (1 - tolerance)    # (time, yh, xh)

    # first drop index along zh
    drop_mask = tracer_core < drop_threshold               # (time, zh, yh, xh)
    first_drop_idx = drop_mask.argmax(dim='zh')            # (time, yh, xh)
    has_drop = drop_mask.any(dim='zh')
    last_idx = xr.zeros_like(first_drop_idx) + (tracer_core.sizes['zh'] - 1)
    first_drop_idx = xr.where(has_drop, first_drop_idx, last_idx)

    # map index → height (ASL)
    bl_height_vals = xr.apply_ufunc(
        np.take,
        z_core,               # (zh)
        first_drop_idx,       # (time, yh, xh)
        input_core_dims=[['zh'], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[z_core.dtype]
    )

    bl_height = xr.DataArray(
        bl_height_vals,
        dims=('time', 'yh', 'xh'),
        coords={'time': tracer_da['time'], 'yh': tracer_da['yh'], 'xh': tracer_da['xh']},
        name='tracer_bl_height'
    )

    # convert to AGL if requested
    if return_agl:
        if zs_data_path:
            orography= xr.open_dataset(zs_data_path).squeeze()
        else:
            raise("terrain dataarray not provided")
        bl_height = (bl_height - orography).rename('tracer_bl_height_agl')

    return bl_height

def diagnose_ABL_height_parcel(theta_da, z, threshold=0.5):
    """
    Parcel-method ABL height for (time, zh, yh, xh) input, robust to NaNs.
    Returns (time, yh, xh) (or (yh, xh) if no 'time') in same units as z.

    - Uses the first non-NaN level per column as the 'surface'
    - Only searches at/above that level
    - If no exceed, returns the last valid level
    - All-NaN columns -> NaN
    """
    if not isinstance(theta_da, xr.DataArray):
        raise TypeError("theta_da must be an xarray.DataArray")
    if 'zh' not in theta_da.dims:
        raise ValueError("theta_da must have a 'zh' dimension")

    has_time = 'time' in theta_da.dims

    # IMPORTANT: make zh a single chunk so it can be used as core dimension
    theta_da = theta_da.chunk(dict(zh=-1))

    # z levels as DataArray aligned on 'zh'
    z_da = xr.DataArray(np.asarray(z), dims=['zh'])
    if z_da.sizes['zh'] != theta_da.sizes['zh']:
        raise ValueError("len(z) must match theta_da.sizes['zh']")

    nz = theta_da.sizes['zh']
    zh_idx = xr.DataArray(np.arange(nz), dims=['zh'])

    # Valid (non-NaN) mask per level
    valid = theta_da.notnull()
    any_valid = valid.any(dim='zh')  # columns that have at least one valid level

    # First valid level index from the bottom (surface) and last valid from the top
    first_valid_idx = valid.argmax(dim='zh')  # 0 if all False; fix later with any_valid
    last_valid_from_top = valid.isel(zh=slice(None, None, -1)).argmax(dim='zh')
    last_valid_idx = (nz - 1) - last_valid_from_top

    # Helper: take along axis 0 (the 'zh' core dim)
    def take0(a, i):
        return np.take(a, i, axis=0)

    # Baseline theta at the first valid level (column-wise)
    theta_surf = xr.apply_ufunc(
        take0, theta_da, first_valid_idx,
        input_core_dims=[['zh'], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[theta_da.dtype],
    )

    # Δθ relative to that column's surface
    delta_theta = theta_da - theta_surf

    # Only consider levels at/above the surface level
    above_surface = zh_idx >= first_valid_idx
    # Consider only valid levels too
    cond = (delta_theta >= threshold) & above_surface & valid

    # First exceed index (0 if none -> we’ll fix using 'cond.any')
    first_exceed_idx = cond.argmax(dim='zh')
    exceeded_any = cond.any(dim='zh')

    # If never exceeded, choose last valid level; else the first exceed level
    chosen_idx = xr.where(exceeded_any, first_exceed_idx, last_valid_idx)

    # Map index -> height
    abl_height = xr.apply_ufunc(
        take0, z_da, chosen_idx,
        input_core_dims=[['zh'], []],
        output_core_dims=[[]],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[z_da.dtype],
    )

    # All-NaN columns -> NaN
    abl_height = abl_height.where(any_valid)

    # Attach coords and order dims
    if has_time:
        abl_height = abl_height.assign_coords(
            time=theta_da['time'], yh=theta_da['yh'], xh=theta_da['xh']
        ).transpose('time', 'yh', 'xh')
    else:
        abl_height = abl_height.assign_coords(
            yh=theta_da['yh'], xh=theta_da['xh']
        ).transpose('yh', 'xh')

    return abl_height.rename('abl_height')

def net_radiation(ds, mean=True):
    SWup = ds.swupb
    SWdn = ds.swdnb
    LWup = ds.lwupb
    LWdn = ds.lwdnb

    # Net radiation
    netRadiation = (LWdn + SWdn) - (LWup + SWup)

    # Area average over horizontal dimensions
    if mean:
        netRadiation_mean = netRadiation.mean(dim=["xh", "yh"])
        return netRadiation_mean
    else:return netRadiation

def find_nearest_index(arr: np.ndarray, target: float):
    tolerance = 5.0
    difference = np.abs(arr - target)
    if np.any(difference <= tolerance):
        return difference.argmin()
    else:
        return None


def xrinterpolate(ds, heights, method="linear"):
    # example usage:
    # interpHeights = [entry["height"] for entry in inletHeights.values()]
    # ds = xrinterpolate(ds, interpHeights)
    return ds.interp(zh=heights, method=method)


def naninterpolate(arr):
    """Interpolate the NaN values in ndArray"""
    # Find indices of NaN values
    nan_indices = np.isnan(arr)
    # Interpolate NaN value using linear interpolation
    arr[nan_indices] = np.interp(
        np.flatnonzero(nan_indices), np.flatnonzero(~nan_indices), arr[~nan_indices]
    )

def get_tke(
    ds: xr.Dataset,
    *,
    opt: str = "res",  # "res" | "tot" | "ratio"
    uu_name: str = "u_u",
    vv_name: str = "v_v",
    ww_name: str = "w_w",
    u_name: str = "u_TM",
    v_name: str = "v_TM",
    w_name: str = "w_TM",
    tke_sg_name: str = "tke_sg",
) -> xr.DataArray:
    """
    Compute TKE from CM1-style second moments.

    res  = 0.5*(u_u + v_v + w_w) - 0.5*(u_TM^2 + v_TM^2 + w_TM^2)
    tot  = res + tke_sg
    ratio = tke_sg / res

    Notes:
    - Assumes u_u, v_v, w_w are total second moments (e.g., <u^2>, <v^2>, <w^2>)
      and u_TM, v_TM, w_TM are means.
    """
    needed = {uu_name, vv_name, ww_name, u_name, v_name, w_name}
    missing = [n for n in needed if n not in ds.data_vars]
    if missing:
        raise KeyError(f"Missing for TKE: {missing}. Available: {list(ds.data_vars)}")

    uu = ds[uu_name]
    vv = ds[vv_name]
    ww = ds[ww_name]
    u = ds[u_name]
    v = ds[v_name]
    w = ds[w_name]

    tke_res = 0.5 * (uu + vv + ww) - 0.5 * (u**2 + v**2 + w**2)

    if opt == "res":
        out = tke_res.rename("tke_res")
        out.attrs["long_name"] = "Resolved turbulent kinetic energy"
        out.attrs["units"] = "m^2 s^-2"
        out.attrs["computed_from"] = f"{uu_name},{vv_name},{ww_name},{u_name},{v_name},{w_name}"
        return out

    if tke_sg_name not in ds.data_vars:
        raise KeyError(f"opt='{opt}' requires '{tke_sg_name}'. Available: {list(ds.data_vars)}")

    tke_sg = ds[tke_sg_name]

    if opt == "tot":
        out = (tke_res + tke_sg).rename("tke_tot")
        out.attrs["long_name"] = "Total TKE (resolved + subgrid)"
        out.attrs["units"] = "m^2 s^-2"
        out.attrs["computed_from"] = f"tke_res + {tke_sg_name}"
        return out

    if opt == "ratio":
        out = (tke_sg / tke_res).rename("tke_sg_over_res")
        out.attrs["long_name"] = "Subgrid-to-resolved TKE ratio"
        out.attrs["units"] = "1"
        out.attrs["computed_from"] = f"{tke_sg_name} / tke_res"
        return out

    raise ValueError("opt must be one of: 'res', 'tot', 'ratio'")


def calc_wind(t, ds):
    ds = ds.sel(time=t)
    utm = ds.u_TM
    vtm = ds.v_TM
    wtm = ds.w_TM
    return np.sqrt(utm**2 + vtm**2 + wtm**2)


def find_destagger_dimension(*arg):
    zm, xm = arg[0]
    for zm1, xm1 in arg[1:]:
        if xm1 < xm:
            xm = xm1
        if zm1 < zm:
            zm = zm1
    return zm, xm


def destagger(ia, zm, xm):
    ia = ia.values
    if ia.shape[0] != zm:
        ia = (ia[1:] + ia[:-1]) / 2
    if ia.shape[1] != xm:
        ia = (ia[:, 1:] + ia[:, :-1]) / 2
    return ia
