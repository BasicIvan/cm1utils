import sys
import os
import xarray as xr
import numpy as np
#sys.path.append(os.path.abspath(".."))

class RecirculationZones():
    """
    Applycation of the stagnation and ventilation zones from Allwine and Whiteman (1993)
    doi.org/10.1016/1352-2310(94)90048-5 (Ch. 2: Methodology)
    """

    def __init__(self):
        self.d = None  # velocity grid
        self.T = 600.  # averaging interval [s]

    def get_velocity_grid(
        self,
        data_dir,
        it1: int,
        it2: int,
        numberOfVerticalPoints: int = 2,
        filePrefix: str = "cm1out_",
    ):
        """
        Loads files within the specified time index range, extracts 2D wind fields,
        and stores the data into self.d as xarray.Dataset.

        Parameters:
            data_dir (str): Directory with NetCDF files.
            it1 (int): Start index (1-based, inclusive).
            it2 (int): End index (1-based, inclusive).
            numberOfVerticalPoints (int): Number of vertical points to average.
            filePrefix (str): File prefix, default 'cm1out_'.
        """
        variables = ["u_TM", "v_TM"]

        file_list = self.get_sorted_file_list(data_dir, prefix=filePrefix)
        file_list = file_list[it1-1:it2]  # Convert to 0-based index

        data_arrays = {var: [] for var in variables}
        time_coords = []

        for file_name in file_list:
            file_path = os.path.join(data_dir, file_name)
            try:
                ds = xr.open_dataset(file_path).isel(time=0)
                file_index = int(file_name[-9:-3])
                time_val = self.get_absolute_time(file_index)

                for var in variables:
                    if numberOfVerticalPoints > 1:
                        da = ds[var].isel(zh=slice(0, numberOfVerticalPoints))
                        da = da.mean(dim="zh")
                        #da = da.expand_dims("zh")
                    elif numberOfVerticalPoints == 1:
                        da = ds[var].isel(zh=0)#.expand_dims("zh")
                    data_arrays[var].append(da)

                time_coords.append(time_val)

            except OSError as e:
                print(f"Skipping {file_path} due to error: {e}")

            finally:
                if "ds" in locals():
                    ds.close()

        if not any(len(lst) > 0 for lst in data_arrays.values()):
            raise ValueError("No valid data collected. Check filenames or variable names.")

        concatenated_vars = {var: xr.concat(data_arrays[var], dim="time") for var in variables}
        final_ds = xr.Dataset(concatenated_vars)
        final_ds = final_ds.assign_coords(time=("time", time_coords))

        self.d = final_ds

    @property
    def recirculationFactor(self):
        S = self.windRun
        L = self.resultantTransportDistance
        return 1 - L/S

    @property
    def vectorMagnitude(self):
        u = self.d.u_TM
        v = self.d.v_TM
        return np.sqrt(u**2+v**2)
    
    @property
    def directionAngle(self):
        return np.pi + np.arcsin(self.d.u_TM/self.V)
    
    @property
    def windRun(self):
        return self.T * self.vectorMagnitude.sum(dim="time")
    
    @property
    def nsTransportDistance(self):
        return self.T * self.d.v_TM.sum(dim="time")
    
    @property
    def ewTransportDistance(self):
        return self.T * self.d.u_TM.sum(dim="time")
    
    @property
    def resultantTransportDistance(self):
        Y = self.ewTransportDistance
        X = self.nsTransportDistance
        return np.sqrt(Y**2 + X**2)
    


    def get_sorted_file_list(self, data_dir: str, prefix: str = "cm1out_"):
        file_list = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.startswith(prefix) and f.endswith(".nc")
            ]
        )
        return file_list

    def get_absolute_time(self, index, base_time="2021-07-21T18:00:00", interval_minutes=10):
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