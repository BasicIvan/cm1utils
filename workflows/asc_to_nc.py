"""
Convert Swiss Topography to netCDF in CM1-compatible format.
The topography is obtained, e.g., at:
https://www.swisstopo.admin.ch/en/
Example usage is at bottom

created by: Ivan Basic on 08.11.2024.

"""

import sys
import os
#sys.path.append(os.path.abspath(".."))
from cm1utils.tools import r, d, p
import rasterio as ra
import xarray as xr
import numpy as np

class swisstopoToNc:
    def __init__(self) -> None:
        self.da = None
        self.topo = None
        self.original_topo = None
        self.topoYs = 0.0
        self.topoXs = 0.0

    def loadDataFromAsc(self, ascFileName: str):
        """
        Saves Xarray dataset from the .asc file from the provided path (IOfileName).
        """
        if not ascFileName.endswith(".asc"):
            raise ValueError(
                "Invalid file name. The file must have a '.asc' extension."
            )

        with ra.open(ascFileName) as src:
            data = src.read(1)  # Read the first band (there might be only one band)

            # Create coordinate arrays for x and y using the transform
            transform = src.transform
            x_coords = (
                np.arange(data.shape[1]) * transform[0] + transform[2]
            )  # X coordinates
            y_coords = (
                np.arange(data.shape[0]) * transform[4] + transform[5]
            )  # Y coordinates

            # Create xarray DataArray with the loaded data
            self.da = xr.DataArray(
                data, dims=["y", "x"], coords={"x": x_coords, "y": y_coords}
            )

    def checkSteepness(self, plots: bool = False, returns: bool = False):
        """
        Checks steepnes of topography and prints if it recommends smoothing.
        plots: set to True if steepness contours are required
        """
        # Step 1: Compute the gradient in x and y directions
        dx = self.topoXs[1] - self.topoXs[0]  # Grid spacing in x direction
        dy = self.topoYs[1] - self.topoYs[0]  # Grid spacing in y direction

        # Calculate the gradient in x and y directions
        grad_y, grad_x = np.gradient(self.topo, dy, dx)

        # Step 2: Calculate the magnitude of the gradient (steepness)
        steepnessMap = np.sqrt(grad_x**2 + grad_y**2)

        # check maximum steepness and its location
        maxSteepness = np.max(steepnessMap)

        # Print the results
        print(f"Maximum steepness: {maxSteepness}")
        if maxSteepness < 0.55:
            print("CM1 should work fine with this terrain.")
        else:
            print("CM1 will most likely stop, due to the CFL condition.")
            maxSteepnessCoords = np.unravel_index(
                np.argmax(steepnessMap), steepnessMap.shape
            )
            print(
                f"Smoothing is recommended, especially at coordinates: {maxSteepnessCoords}"
            )
        if plots:
            # Visualize the steepness
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 8))
            plt.contourf(
                self.topoXs, self.topoYs, steepnessMap, levels=100, cmap="viridis"
            )
            plt.colorbar(label="Steepness (m/m)")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("Steepness of the Topography")

            plt.show()

        if returns:
            return self.topoXs, self.topoYs, steepnessMap

    def loadTopography(self, coordinates: list, domain: list = [15000, 15000]):
        """
        coordinates (list): swiss x,y / lon,lat
        domain (list): size for domain in [X,Y] direction. default is 15x15 km
        """
        if not self.da.all():
            raise RuntimeError(
                "Data not loaded. Please load the data before proceeding."
            )
        # Check if the input is indeed a list
        if not isinstance(coordinates, list) or not isinstance(domain, list):
            raise TypeError("Expected coordinates and domain to be of type 'list'.")
        # Unpack the list
        if len(coordinates) != 2 or len(domain) != 2:
            raise ValueError("Expected a list with exactly two elements.")

        sX, sY = coordinates
        dX, dY = [x / 2 for x in domain]
        idx = np.where((self.da.x > sX - dX) & (self.da.x < sX + dX))[0]
        idy = np.where((self.da.y > sY - dY) & (self.da.y < sY + dY))[0]

        self.topo = self.da[idy, idx]
        self.topoXs = self.topo.x.values
        self.topoYs = self.topo.y.values
        # saves it as original_topo...
        self.original_topo = self.topo
    
    def reset(self):
        """
        Resets the topo data to the original state prior to modifying, so the whole loading doesn't have to be done again.
        """
        self.topo = self.original_topo
        self.topoXs = self.original_topo.x.values
        self.topoYs = self.original_topo.y.values

    def smoothTopography(
        self,
        sigma: float,
        selectively: bool = False,
        sigmaAtBoundary: float = 4.0,
        boundaryDepth: int = 100,
    ):
        """
        Smooths the topography with gaussian filter.
        optional: selectively smooth (spatially varying gaussian filter)
        """

        from scipy.ndimage import gaussian_filter

        if not selectively:
            smoothed_topo = gaussian_filter(self.topo.data, sigma)
            # Re-wrap as DataArray with same coords and dims
            self.topo = xr.DataArray(
                smoothed_topo,
                coords=self.topo.coords,
                dims=self.topo.dims,
            )
        else:
            # selectively smoothing:
            # Original topography data
            topo_data = self.topo.data
            # ny, nx = topo_data.shape

            # Create a mask for boundary regions
            mask = np.zeros_like(topo_data)

            # Set boundary regions to 1 (indicating these areas will be smoothed)
            mask[:boundaryDepth, :] = 1  # Top boundary
            mask[-boundaryDepth:, :] = 1  # Bottom boundary
            mask[:, :boundaryDepth] = 1  # Left boundary
            mask[:, -boundaryDepth:] = 1  # Right boundary

            # Apply Gaussian filter to the entire terrain but weighted by the mask
            smoothed_topo = topo_data.copy()
            boundary_smoothed = gaussian_filter(
                topo_data, sigma=sigmaAtBoundary
            )  # Aggressively smoothed version
            inner_smoothed = gaussian_filter(
                topo_data, sigma=sigma
            )  # Lightly smoothed version

            # Combine boundary-smoothed and inner-smoothed data based on the mask
            smoothed_topo[mask == 1] = boundary_smoothed[
                mask == 1
            ]  # Apply aggressive smoothing to boundaries
            smoothed_topo[mask == 0] = inner_smoothed[
                mask == 0
            ]  # Apply light smoothing to inner areas

            # old:
            #self.topo = smoothed_topo

            # Re-wrap as DataArray with same coords and dims
            self.topo = xr.DataArray(
                smoothed_topo,
                coords=self.topo.coords,
                dims=self.topo.dims,
            )
        print("Topography smoothed.")

    def downsampleTopography(self, factor: int = 4):
        """
        Downsamples the current topography by averaging over (factor x factor) blocks.
        Updates self.topo, self.topoXs, and self.topoYs accordingly.
        """
        if not self.topo.all():
            raise RuntimeError("Topography not loaded. Please load topography first.")

        topo_data = self.topo.values
        ny, nx = topo_data.shape

        # Ensure dimensions are divisible by factor
        ny_trim = ny - (ny % factor)
        nx_trim = nx - (nx % factor)
        trimmed = topo_data[:ny_trim, :nx_trim]

        # Reshape and take mean over blocks
        blocks = trimmed.reshape(ny_trim // factor, factor, nx_trim // factor, factor)
        blocks = blocks.transpose(0, 2, 1, 3).reshape(
            ny_trim // factor, nx_trim // factor, -1
        )
        topo_downsampled = blocks.mean(axis=2)

        # Compute new coordinates
        x_vals = self.topo.x.values[:nx_trim].reshape(-1, factor)
        y_vals = self.topo.y.values[:ny_trim].reshape(-1, factor)
        new_x = x_vals.mean(axis=1)
        new_y = y_vals.mean(axis=1)

        # Update internal state
        self.topo = xr.DataArray(
            topo_downsampled, coords={"y": new_y, "x": new_x}, dims=("y", "x")
        )
        self.topoXs = new_x
        self.topoYs = new_y

        print(f"Topography downsampled by factor {factor} (e.g., 25 m → {25*factor} m)")
        print(
            f"New topography shape: {self.topo.shape[0]} (y), {self.topo.shape[1]} (x)"
        )

    def createNetCDF(self, ncFileName: str = "myzs.nc"):
        """
        Creates a netCDF file with "myzs" variable for coordinates.
        Rotates it automatically for CM1.
        This routine is not very flexible and it is a result of trial and error.
        It appears as CM1, at least in the current setup, works only with nc file created like this.
        """
        if not ncFileName.endswith(".nc"):
            raise ValueError("Invalid file name. The file must have a '.nc' extension.")
        if not self.topo.all():
            raise RuntimeError(
                "Topography not loaded. Load the topography before proceeding."
            )

        # Create the xarray Dataset with dx as a scalar, and the rotated x, y, and myzs data
        ds = xr.Dataset(
            {
                "dx": (["one"], [np.float64(25)]),  # dx as a scalar variable
                "x": (["nx"], self.topoXs),  # x as a variable
                "y": (["ny"], self.topoYs),  # y as a variable
                "myzs": (
                    ["nx", "ny"],
                    self.topo.data,
                ),
            }
        )

        # Write the dataset to a NetCDF file
        ds.to_netcdf(
            f"/home/b/b381871/basiclab/data/nc_files/{ncFileName}",
            format="NETCDF4_CLASSIC",
            encoding={
                "myzs": {
                    "dtype": "float64",
                    "_FillValue": None,
                },
                "dx": {"dtype": "float64", "_FillValue": None},
                "x": {"dtype": "float64", "_FillValue": None},
                "y": {"dtype": "float64", "_FillValue": None},
            },
        )

    def blendTopoBordersToMedian(self, blend_width: int = 15,opt="median"):
        """
        Smoothly blends the outermost `blend_width` rows and columns of the topography
        into the border median using a cosine ramp.
        
        This prevents sharp discontinuities at the domain edges.

        blend_width: refers to the number of grid cells
        """
        if self.topo is None:
            raise RuntimeError("Topography not loaded. Cannot blend borders.")

        if blend_width <= 0 or blend_width > min(self.topo.shape) // 2:
            raise ValueError("Invalid blend width. Must be positive and smaller than half the domain size.")
        if opt=="mean":
            median = self.topoBorderMean
        else:
            median = self.topoBorderMedian
        topo_blend = self.topo.copy().values

        # Cosine ramp weights from 1 (original) to 0 (median)
        ramp = 0.5 * (1 + np.cos(np.linspace(np.pi, 0, blend_width)))

        # --- Top and Bottom rows ---
        for i in range(blend_width):
            weight = ramp[i]
            topo_blend[i, :] = weight * topo_blend[i, :] + (1 - weight) * median  # Top
            topo_blend[-(i + 1), :] = weight * topo_blend[-(i + 1), :] + (1 - weight) * median  # Bottom

        # --- Left and Right columns ---
        for j in range(blend_width):
            weight = ramp[j]
            topo_blend[:, j] = weight * topo_blend[:, j] + (1 - weight) * median  # Left
            topo_blend[:, -(j + 1)] = weight * topo_blend[:, -(j + 1)] + (1 - weight) * median  # Right

        # Update self.topo
        self.topo = xr.DataArray(
            topo_blend, coords=self.topo.coords, dims=self.topo.dims
        )

        print(f"Blended all borders to median {median:.2f} m over {blend_width} grid cells")

    def plotTopography(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 8))
        cf = ax.contourf(self.topo.x, self.topo.y, self.topo, np.arange(300,1000,20), cmap='terrain')
        fig.gca().set_aspect('equal')
        fig.colorbar(cf,label='Elevation (m)')
        ax.grid(visible=True, linestyle='--', alpha=0.5)

        # Label axes
        plt.xlabel('W-E (Swiss coordinates)')
        plt.ylabel('S-N (Swiss Coordinates)')
        return fig
            
    @property
    def topoBorderMedian(self):
        """
        Computes the median value of all border cells (top, bottom, left, right).
        Corners are not double-counted.
        Returns a single float.
        """
        if self.topo is None:
            raise RuntimeError("Topography not loaded. Cannot compute border median.")

        top = self.topo.isel(y=0).values
        bottom = self.topo.isel(y=-1).values
        left = self.topo.isel(x=0).isel(y=slice(1, -1)).values
        right = self.topo.isel(x=-1).isel(y=slice(1, -1)).values

        border_values = np.concatenate([top, bottom, left, right])
        median_border = np.median(border_values)

        #print(f"Median of all borders: {median_border:.2f} m")
        return median_border
    
    @property
    def topoBorderMean(self):
        """
        Computes the mean value of all border cells (top, bottom, left, right).
        Corners are not double-counted.
        Returns a single float.
        """
        if self.topo is None:
            raise RuntimeError("Topography not loaded. Cannot compute border mean.")

        top = self.topo.isel(y=0).values
        bottom = self.topo.isel(y=-1).values
        left = self.topo.isel(x=0).isel(y=slice(1, -1)).values
        right = self.topo.isel(x=-1).isel(y=slice(1, -1)).values

        border_values = np.concatenate([top, bottom, left, right])
        mean_border = np.mean(border_values)

        print(f"Mean of all borders: {mean_border:.2f} m")
        return mean_border

# example usage for Beromünster tower (655839,226774) - old Swiss coordinates
#snc = swisstopoToNc()
#snc.loadDataFromAsc("dhm25_grid_raster.asc")
#snc.loadTopography(coordinates=[655839, 226774], domain=[30000, 30000])