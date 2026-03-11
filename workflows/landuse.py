import sys
import os
#sys.path.append(os.path.abspath(".."))
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cm1utils.tools.d import landuse_dict

""" Tools for plotting landuse indices from mylanduse.nc 

Landuse indices and descriptions are in basictools.d
"""

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import scipy.stats

class Landuse:
    def __init__(self):
        self.lu = None

    def downsample_landuse(self,
        factor: int = 4,
    ) -> xr.DataArray:
        """
        Downsamples a land use DataArray using mode aggregation over blocks.

        Parameters:
            landuse_da : xr.DataArray
                The input land use data (2D, with dimensions y and x).
            factor : int
                Downsampling factor (e.g., 4 = 100 m if original is 25 m).
            save_path : str, optional
                If given, the downsampled data will be saved to this NetCDF path.

        Returns:
            xr.DataArray
                The downsampled land use data.
        """
        # Convert to numpy and get shape
        data = self.lu.values
        ny, nx = data.shape

        # Ensure dimensions are divisible by factor
        ny_trim = ny - (ny % factor)
        nx_trim = nx - (nx % factor)
        data_trimmed = data[:ny_trim, :nx_trim]

        # Reshape and compute mode
        blocks = data_trimmed.reshape(
            ny_trim // factor, factor, nx_trim // factor, factor
        )
        blocks = blocks.transpose(0, 2, 1, 3).reshape(
            ny_trim // factor, nx_trim // factor, -1
        )
        data_downsampled = scipy.stats.mode(blocks, axis=2)[0].squeeze()

        # Compute new coordinates
        x_vals = self.lu.x.values[:nx_trim].reshape(-1, factor)
        y_vals = self.lu.y.values[:ny_trim].reshape(-1, factor)
        new_x = x_vals.mean(axis=1)
        new_y = y_vals.mean(axis=1)

        # Create downsampled DataArray
        self.lu = xr.DataArray(
            data_downsampled,
            coords={"y": new_y, "x": new_x},
            dims=("y", "x"),
            name=self.lu.name,
            attrs=self.lu.attrs,
        )

    def save_to_netcdf(self,save_path = "/home/b/b381871/basiclab/data/nc_files/mylanduse_new.nc"):
        self.lu.to_netcdf(save_path)

    def open_landuse_dataset(self,filepath="/home/b/b381871/cm1/RUN_CM1/INPUT/mylanduse_25m.nc"):
        # Open the dataset and extract land-use
        ds = xr.open_dataset(filepath)
        self.lu = ds.landuse.isel(band=0)

    def plot_landuse_map(self):
        # Extract colormap and normalization
        categories = list(landuse_dict.keys())
        colors = [landuse_dict[k][1] for k in categories]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(len(categories) + 1) - 0.5, len(categories))

        # Create figure and axis with constrained layout
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

        # Plot the land use data
        im = ax.pcolormesh(
            self.lu.x, self.lu.y, self.lu, cmap=cmap, norm=norm, shading="auto", rasterized=True
        )

        # Get only the used categories
        unique_indices = np.unique(self.lu.values)

        # Create a custom legend for present categories
        legend_patches = [
            Patch(color=landuse_dict[i][1], label=f"{i}: {landuse_dict[i][0]}")
            for i in unique_indices
            if i in landuse_dict
        ]
        ax.legend(
            handles=legend_patches,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
        )

        ax.set_aspect("equal")

        # Axis labels and title
        ax.set_xlabel("Swiss X")
        ax.set_ylabel("Swiss Y")
        ax.set_title("Land Use Indices Map")
        plt.show()
        return fig
    
