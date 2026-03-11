#sys.path.append(os.path.abspath(".."))
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from cm1utils.tools.data import get_sorted_file_list


def plot_topography(
    topo_filepath="/home/b/b381871/basiclab/data/nc_files/topo_100m.nc",
    overlay_water=True,
    lu_filepath="/home/b/b381871/cm1/RUN_CM1/INPUT/mylanduse_100m.nc",
    xcoord=0,
    ycoord=0,
    cmap="gist_earth",
    scatterLabel="Beromünster Tower",
):
    # Load topography data
    ds = xr.open_dataset(topo_filepath)#.isel(time=0)
    # Load land-use data
    if overlay_water:
        lu_ds = xr.open_dataset(lu_filepath)
        # lu = lu_ds.landuse.isel(band=0)
        lu = lu_ds.landuse.isel(y=slice(None, None, -1))  # flip the y-axis

        # Find water bodies (LU index 16)
        water_mask = lu == 16

    # Coordinates in km for plotting
    x_km = ds.xh / 1000
    y_km = ds.yh / 1000

    # Set up figure and axes
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    axhandles = []

    # Plot topography
    contour = ax.contourf(
        x_km,
        y_km,
        ds.zs,
        np.arange(300, 1000, 20),
        cmap=cmap,
    )
    fig.colorbar(contour, ax=ax, label="Elevation (m)")

    # Overlay water bodies
    if overlay_water:
        ax.contourf(
            x_km,
            y_km,
            water_mask,
            levels=[0.5, 1.5],
            colors="blue",
            alpha=1.0,
        )
        water_patch = Patch(color="blue", label="Water Bodies (LU Index 16)")
        axhandles.append(water_patch)

    # Mark the tower
    ax.scatter(xcoord, ycoord, marker="D", color="red", s=50, label=scatterLabel)

    # Axis ticks and labels
    ax.set_xticks(np.arange(-15, 16, 5))
    ax.set_yticks(np.arange(-15, 16, 5))
    ax.grid(visible=True, linestyle="--", alpha=0.5)
    ax.set_xlabel("Easting relative to domain center (km)")
    ax.set_ylabel("Northing relative to domain center (km)")

    # Legend setup
    tower_marker = Line2D(
        [],
        [],
        marker="D",
        color="red",
        linestyle="None",
        markersize=7,
        label=scatterLabel,
    )
    axhandles.append(tower_marker)
    ax.legend(handles=axhandles, loc="upper right")

    # Clean up
    ds.close()
    if overlay_water:
        lu_ds.close()

    plt.show()
    return fig

def get_virtual_tower_sampling_data(
    
    variables: list,
    data_dir,
    xcoord=0.0,
    ycoord=0.0,
    interpolate_nans: bool = False,
    filePrefix="cm1out_",
):
    """
    Returns dataset with the sampling data interpolated to the specified heights for all variables in the list.
    If interpolate_nans=True, missing time steps will be filled with NaNs, inferred from expected 10-minute intervals.
    The returned object is an xarray.Dataset with all variables included.
    """
    import re
    import numpy as np
    import pandas as pd

    sampling_heights = [12.5, 44.6, 71.5, 131.6, 212.5]
    output_interval_minutes = 10
    total_duration_minutes = 48 * 60
    n_expected_steps = total_duration_minutes // output_interval_minutes + 1

    # Get available output files
    available_indices = set()
    file_pattern = re.compile(filePrefix + r"(\d{6})\.nc")
    for fname in os.listdir(data_dir):
        match = file_pattern.match(fname)
        if match:
            index = int(match.group(1))
            available_indices.add(index)

    # Dictionary to store lists of DataArrays for each variable
    sampled_data_dict = {var: [] for var in variables}

    # Define a base time for fallback if ds.time is not available
    base_time = np.datetime64("2021-07-21T17:00:00")

    for i in range(1, n_expected_steps + 1):  # i from 1 to 217
        file_index = str(i).zfill(6)
        file_name = f"{filePrefix}{file_index}.nc"
        file_path = os.path.join(data_dir, file_name)

        if i in available_indices:
            try:
                ds = openSingleTimeStep(file_path)

                # Determine time coordinate for this time step
                if "time" in ds.coords:
                    time_val = ds.time.values
                    if isinstance(time_val, (np.ndarray, list)):
                        time_val = time_val[0]
                    try:
                        time_val = np.datetime64(time_val)
                    except Exception:
                        # fallback to get_absolute_time if conversion fails
                        time_val = get_absolute_time(i)
                else:
                    # Fallback: generate time based on index
                    time_val = get_absolute_time(i)

                for var in variables:
                    if var not in ds:
                        # Variable missing from file
                        nan_array = np.full((1, len(sampling_heights)), np.nan)
                        dummy = xr.DataArray(
                            nan_array,
                            dims=["time", "z"],
                            coords={"time": [time_val], "z": sampling_heights},
                            name=var,
                        )
                        sampled_data_dict[var].append(dummy)
                        continue
                    var_dims = ds[var].dims
                    interp_kwargs = {}
                    for dim in var_dims:
                        if dim in ["x", "xh"]:
                            interp_kwargs[dim] = xcoord * 1000
                        elif dim in ["y", "yh"]:
                            interp_kwargs[dim] = ycoord * 1000
                        elif dim in ["z", "zh"]:
                            interp_kwargs[dim] = sampling_heights
                    # Check that all required interpolation axes are covered
                    required_dims = ["x", "y", "z"]
                    if not all(
                        any(d in var_dims for d in [dim, f"{dim}h"])
                        for dim in required_dims
                    ):
                        raise ValueError(
                            f"Unexpected dimensions {var_dims} for variable {var}"
                        )
                    data_np = ds[var].interp(**interp_kwargs).values
                    da_interp = xr.DataArray(
                        data_np[np.newaxis, :],
                        dims=["time", "z"],
                        coords={"time": [time_val], "z": sampling_heights},
                        name=var,
                    )
                    sampled_data_dict[var].append(da_interp)
            except OSError as e:
                print(f"Skipping {file_path} due to error: {e}")
                # If file cannot be opened, fill all variables at this time with NaN
                # Use fallback time for missing file
                time_val = get_absolute_time(i)
                for var in variables:
                    nan_array = np.full((1, len(sampling_heights)), np.nan)
                    dummy = xr.DataArray(
                        nan_array,
                        dims=["time", "z"],
                        coords={"time": [time_val], "z": sampling_heights},
                        name=var,
                    )
                    sampled_data_dict[var].append(dummy)
            finally:
                if "ds" in locals():
                    ds.close()
        else:
            # File missing, fill all variables with NaN for this timestep
            time_val = get_absolute_time(i)
            for var in variables:
                nan_array = np.full((1, len(sampling_heights)), np.nan)
                dummy = xr.DataArray(
                    nan_array,
                    dims=["time", "z"],
                    coords={"time": [time_val], "z": sampling_heights},
                    name=var,
                )
                sampled_data_dict[var].append(dummy)

    # For each variable, concat along time, then combine into a Dataset
    var_dataarrays = {}
    for var in variables:
        cleaned_da_list = []
        for idx, da in enumerate(sampled_data_dict[var]):
            da = da.assign_coords(time=da.coords["time"])
            # Remove extraneous dimensions if present
            for dim in ["x", "y", "xh", "yh", "z"]:
                if dim in da.dims and dim != "z":
                    da = da.drop(dim)
            # Keep only dims "time" and "z"
            dims_to_keep = ["time"]
            if "z" in da.dims:
                dims_to_keep.append("z")
            da = da.transpose(*dims_to_keep)
            # Remove all coordinate metadata
            da = da.reset_coords(drop=True)
            cleaned_da_list.append(da)
        var_dataarrays[var] = xr.concat(cleaned_da_list, dim="time", coords="minimal")
        # Optionally interpolate missing time steps for each variable
        if interpolate_nans:
            var_dataarrays[var] = var_dataarrays[var].interpolate_na(
                dim="time", method="linear", fill_value="extrapolate"
            )
    # Combine into a single Dataset and drop all global coordinates
    final_ds = xr.Dataset(var_dataarrays)
    for var in list(final_ds.data_vars):
        if var not in var_dataarrays:
            final_ds = final_ds.drop_vars(var)
    final_ds = final_ds.reset_coords(drop=True)
    return final_ds

def get_horizontal_cross_section_at_height(
    
    data_dir,
    isel_zh: int = 1,
    variables: list = None,
    filePrefix: str = "cm1out_",
):
    """
    Returns horizontal cross-section at a given height level for specified variables as a single xarray.Dataset.
    """
    import xarray as xr
    import numpy as np
    import os

    if variables is None:
        variables = ["pt1_TM", "pt2_TM", "pt3_TM"]

    file_list = get_sorted_file_list(data_dir, prefix=filePrefix)

    data_arrays = {var: [] for var in variables}
    time_coords = []

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        try:
            ds = openSingleTimeStep(file_path)
            time_val = get_absolute_time(int(file_name[-9:-3]))

            for var in variables:
                da = ds[var].isel(zh=isel_zh)
                # Expand zh dimension to length 1 to retain coordinate
                da = da.expand_dims("zh")
                data_arrays[var].append(da)

            time_coords.append(time_val)

        except OSError as e:
            print(f"Skipping {file_path} due to error: {e}")

        finally:
            if "ds" in locals():
                ds.close()

    # Check if any data was collected before concatenation
    if not any(len(lst) > 0 for lst in data_arrays.values()):
        raise ValueError("No valid data collected. Check filenames or variable names.")
    # Concatenate along time dimension for each variable
    concatenated_vars = {}
    for var in variables:
        concatenated_vars[var] = xr.concat(data_arrays[var], dim="time")

    # Build a Dataset from variables
    final_ds = xr.Dataset(concatenated_vars)
    final_ds = final_ds.assign_coords(time=("time", time_coords))

    return final_ds

def get_vertical_cross_section_at_y(
    
    data_dir,
    isel_yh: int = 1,
    variables: list = None,
    filePrefix: str = "cm1out_",
):
    """
    Returns vertical cross-section at a fixed y-level (isel_yh) for specified variables as a single xarray.Dataset.
    """
    import xarray as xr
    import numpy as np
    import os

    if variables is None:
        variables = ["pt1_TM", "pt2_TM", "pt3_TM"]

    file_list = get_sorted_file_list(data_dir, prefix=filePrefix)

    data_arrays = {var: [] for var in variables}
    time_coords = []

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        try:
            ds = openSingleTimeStep(file_path)
            #time_val = get_absolute_time(int(file_name[-9:-3]))
            time_val = ds.time

            for var in variables:
                da = ds[var].isel(yh=isel_yh)
                da = da.expand_dims("yh")
                data_arrays[var].append(da)

            time_coords.append(time_val)

        except OSError as e:
            print(f"Skipping {file_path} due to error: {e}")

        finally:
            if "ds" in locals():
                ds.close()

    if not any(len(lst) > 0 for lst in data_arrays.values()):
        raise ValueError("No valid data collected. Check filenames or variable names.")

    concatenated_vars = {}
    for var in variables:
        concatenated_vars[var] = xr.concat(data_arrays[var], dim="time")

    final_ds = xr.Dataset(concatenated_vars)
    final_ds = final_ds.assign_coords(time=("time", time_coords))

    return final_ds

def get_vertical_profile(
    
    var,
    data_dir:str,
    xcoord=0.0,
    ycoord=0.0,
    ztop=212.5,
    saveToNetCDF: str = None,
):
    """
    Returns dataset with the sampling data interpolated to the specified heights.
    """
    # sampling_heights = [12.5, 44.6, 71.5, 131.6, 212.5]

    file_list = sorted(
        [f for f in os.listdir(data_dir) if f.startswith("cm1out_0")]
    )

    sampled_data_list = []  # Store interpolated data for each time step
    # selected_files = [f for f in file_list if int(f[-9:-3]) % 12 == 1]  # Adjust 1 if needed. % 12 means every two hours
    selected_files = file_list
    print(selected_files)
    for file_name in selected_files:  # 6 = every hour

        file_path = os.path.join(data_dir, file_name)
        try:
            ds = openSingleTimeStep(file_path)

            # Interpolate to tower coordinates and sampling heights
            ds_interp = (
                ds[var]
                .sel(zh=slice(ztop))
                .interp(xh=xcoord * 1000, yh=ycoord * 1000)
            )
            values = ds_interp.values
            z_coord = ds_interp.zh.values
            time_coord = get_absolute_time(int(file_name[-9:-3]))
            ds_interp = xr.DataArray(
                values[np.newaxis, :],
                dims=["time", "z"],
                coords={"time": time_coord, "z": z_coord},
                name=var
            )
            sampled_data_list.append(ds_interp)
        except OSError as e:
            print(f"Skipping {file_path} due to error: {e}")

        if "ds" in locals():
            ds.close()

    # Combine all interpolated time steps into a single dataset
    final_ds = xr.concat(sampled_data_list, dim="time")
    if "zh" in final_ds.dims:
        final_ds = final_ds.drop_dims("zh")
    if "zh" in final_ds.coords:
        final_ds = final_ds.drop_vars("zh")

    if saveToNetCDF is not None:
        final_ds.to_netcdf(saveToNetCDF)

    return final_ds

def get_surface_average(
    data_dir,
    variables: list = None,
    filePrefix: str = "cm1out_",
):
    """
    Returns surface average of variables.
    """
    import xarray as xr
    import numpy as np
    import os

    file_list = get_sorted_file_list(data_dir, prefix=filePrefix)

    data_arrays = {var: [] for var in variables}
    time_coords = []

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        try:
            ds = openSingleTimeStep(file_path)
            time_val = get_absolute_time(int(file_name[-9:-3]))

            for var in variables:
                da = ds[var].mean(dim=("xh","yh"))
                data_arrays[var].append(da)

            time_coords.append(time_val)

        except OSError as e:
            print(f"Skipping {file_path} due to error: {e}")

        finally:
            if "ds" in locals():
                ds.close()

    # Check if any data was collected before concatenation
    if not any(len(lst) > 0 for lst in data_arrays.values()):
        raise ValueError("No valid data collected. Check filenames or variable names.")
    # Concatenate along time dimension for each variable
    concatenated_vars = {}
    for var in variables:
        concatenated_vars[var] = xr.concat(data_arrays[var], dim="time")

    # Build a Dataset from variables
    final_ds = xr.Dataset(concatenated_vars)
    final_ds = final_ds.assign_coords(time=("time", time_coords))

    return final_ds

def openSingleTimeStep( path):
    return xr.open_dataset(path).isel(time=0)

def get_absolute_time( index, base_time="2021-07-21T17:00:00", interval_minutes=10):
    """
    Converts a file index (1-based) to an absolute timestamp assuming regular intervals.

    Parameters:
        index (int): File index (1-based, so cm1out_000001.nc = index 1)
        base_time (str): Start of simulation in ISO format
        interval_minutes (int): Interval between outputs in minutes

    Returns:
        np.datetime64: Absolute timestamp for the given index
    """
    base_time = np.datetime64(base_time).astype("datetime64[ns]")
    return base_time + np.timedelta64((index - 1) * interval_minutes, "m")
