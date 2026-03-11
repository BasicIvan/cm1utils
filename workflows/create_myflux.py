"""
create_myflux.py
@ created by Ivan Basic on 21.11.2024.
Uses Meteoskat data from Empa to create myflux.nc file for CM1 use.

Creates ptflux(nt, nx, ny, npt)
dim(fluxComponent) = npt (namelist.input)
"""

import scipy.ndimage
from cm1utils.tools import r, d, p
import xarray as xr
import pandas as pd
import numpy as np


class Myflux:

    def __init__(
        self,
        dir_path: str = "/work/bb1096/b381871/ghg_fluxes/",
        bDate: tuple = ("2021", "07", "21", "17"),
        eDate: tuple = ("2021", "07", "23", "17"),
    ) -> None:
        # dir_path = "/work/bb1096/b381871/ghg_fluxes/"
        CO2_path = "MeteotestEKAT_2015_CO2.nc"
        CH4_path = "MeteotestEKAT_2015_CH4.nc"
        # Biogenic hourly emissions:
        vprm_path = "VPRM_CoCO2_BRM_20210101-20211231.nc"
        # Time profile files for CO2 and CH4
        fCO2 = "MeteotestEKAT_2015_CO2_timeprofile.csv"
        fCH4 = "MeteotestEKAT_2015_CH4_timeprofile.csv"

        # Load anthropogenic data
        self.fluxData_CO2 = xr.open_dataset(dir_path + CO2_path)
        self.fluxData_CH4 = xr.open_dataset(dir_path + CH4_path)
        self.timeFactorData_CO2 = pd.read_csv(dir_path + fCO2, delimiter=",")
        self.timeFactorData_CH4 = pd.read_csv(dir_path + fCH4, delimiter=",")

        # Biogenic:
        self.vprm = xr.open_dataset(dir_path + vprm_path)

        # Select date:
        self.bDate = bDate
        self.eDate = eDate

        self.sectors = ("traffic", "residential", "services", "industry", "agriculture")
        

        self.findIndices()
        self.convertToDatetime()

        # Trimming indices for lon and lat (assumes symmetrical trimming)
        self.lonTrim = 15
        self.latTrim = 21

    def findIndices(self):
        # Select desired dates and times: date and time form: 'YYYY-MM-DD hh:mm:ss' (LT)
        byear, bmonth, bday, bhour = self.bDate
        eyear, emonth, eday, ehour = self.eDate
        # Find beginning and ending index (ib, ie) based on the selected date range
        self.ib = self.timeFactorData_CO2[
            self.timeFactorData_CO2["dtm"] == f"{byear}-{bmonth}-{bday} {bhour}:00:00"
        ].index.item()
        self.ie = self.timeFactorData_CO2[
            self.timeFactorData_CO2["dtm"] == f"{eyear}-{emonth}-{eday} {ehour}:00:00"
        ].index.item()
        dtm = self.vprm.time.values
        self.ibBio = np.where(
            dtm
            == np.datetime64(
                byear
                + "-"
                + bmonth
                + "-"
                + bday
                + "T"
                + bhour
                + ":00"
                + ":00.000000000"
            )
        )[0][0]
        self.ieBio = np.where(
            dtm
            == np.datetime64(
                eyear
                + "-"
                + emonth
                + "-"
                + eday
                + "T"
                + ehour
                + ":00"
                + ":00.000000000"
            )
        )[0][0]

    def convertToDatetime(self):
        # Convert the time columns into datetime for proper indexing
        self.timeFactorData_CO2["dtm"] = pd.to_datetime(self.timeFactorData_CO2["dtm"])
        self.timeFactorData_CH4["dtm"] = pd.to_datetime(self.timeFactorData_CH4["dtm"])

    @property
    def resp(self):
        """
        Returns resp(x,y) with selected date range, lat/lon trimmed.
        Also converts from umol m-2 s-1 to kg m-2 s-1
        """
        return (
            self.vprm.RESP.values[
                self.ibBio : self.ieBio,
                self.latTrim : -self.latTrim - 1,
                self.lonTrim : -self.lonTrim,
            ]
            * 44e-09 * 2.0 # respiration increased by factor 2.0
        )

    @property
    def gpp(self):
        return (
            self.vprm.GPP.values[
                self.ibBio : self.ieBio,
                self.latTrim : -self.latTrim - 1,
                self.lonTrim : -self.lonTrim,
            ]
            * 44e-09
        )

    def spatialCO2(self, sector):
        return self.fluxData_CO2[f"CO2_{sector}"].values[
            self.latTrim : -self.latTrim - 1,
            self.lonTrim : -self.lonTrim,
        ]

    def timeFactorsCO2(self, sector):
        return (
            self.timeFactorData_CO2[f"CO2_{sector}"]
            .iloc[self.ib : self.ie]
            .values[:, np.newaxis, np.newaxis]
        )

    @property
    def anthropogenicFLuxesCO2(self):
        sumArray = None
        for sector in self.sectors:
            # Access CO2 time factors and spatial flux data

            # Sum the results for CO2
            if sumArray is None:
                sumArray = self.timeFactorsCO2(sector) * self.spatialCO2(
                    sector
                )  # Initialize with first sector
            else:
                sumArray += self.timeFactorsCO2(sector) * self.spatialCO2(sector)

        return sumArray

    def spatialCH4(self, sector):
        return self.fluxData_CH4[f"CH4_{sector}"].values[
            self.latTrim : -self.latTrim - 1,
            self.lonTrim : -self.lonTrim,
        ]

    def timeFactorsCH4(self, sector):
        return (
            self.timeFactorData_CH4[f"CH4_{sector}"]
            .iloc[self.ib : self.ie]
            .values[:, np.newaxis, np.newaxis]
        )

    @property
    def anthropogenicFLuxesCH4(self):
        sumArray = None
        for sector in self.sectors:
            # Access CH4 time factors and spatial flux data

            # Sum the results for CH4
            if sumArray is None:
                sumArray = self.timeFactorsCH4(sector) * self.spatialCH4(
                    sector
                )  # Initialize with first sector
            else:
                sumArray += self.timeFactorsCH4(sector) * self.spatialCH4(sector)
        return sumArray

    def interpolateToNewGrid(self, data: list, shape: list = [1200, 1200]) -> list:
        """
        Interpolate a list of input datasets to a new grid of the given shape.

        Parameters:
        - data: list of ndarray
            List of input arrays to be resampled. Each array should be 2D (y, x) or 3D (time, y, x).
        - shape: list
            Desired shape for the grid [newY, newX].

        Returns:
        - list of ndarray
            List of resampled data arrays with the new shape.
        """
        newY, newX = shape
        interpolated_data = []

        for dataset in data:

            # Determine zoom factors based on the dataset's dimensions
            if dataset.ndim == 3:  # For 3D data: (time, y, x)
                zoom_factors = (
                    1,  # No zoom on the time axis
                    newY / dataset.shape[1],  # y-axis scaling
                    newX / dataset.shape[2],  # x-axis scaling
                )
            elif dataset.ndim == 2:  # For 2D data: (y, x)
                zoom_factors = (
                    newY / dataset.shape[0],  # y-axis scaling
                    newX / dataset.shape[1],  # x-axis scaling
                )
            else:
                raise ValueError("Input data must be 2D or 3D.")

            # Perform interpolation
            interpolated_dataset = scipy.ndimage.zoom(
                dataset, zoom=zoom_factors, order=1
            )
            interpolated_data.append(interpolated_dataset)

        return interpolated_data

    def createNetCDF(self, ncName: str = "myflux.nc", shape: list = [1200, 1200]):
        if not ncName.endswith(".nc"):
            raise ValueError(
                f"The provided name '{ncName}' is invalid. It must end with 'nc'."
            )
        # Stack CO2 and CH4 into the new "npt" dimension
        stacked_flux = np.stack(
            self.interpolateToNewGrid(
                data=[
                    self.gpp,
                    self.resp,
                    self.anthropogenicFLuxesCO2,
                    self.anthropogenicFLuxesCH4,
                ],
            shape=shape),
            axis=-1,
        ) # Shape (time, ny, nx, npt)
        stacked_flux = np.transpose(stacked_flux, (1, 2, 0, 3))  # Swap to yxtp. this will work with current CM1
        print(
            f"""The dimensions after transposing:
              {stacked_flux.shape}
              """
        )

        time_values = np.arange(
            1, stacked_flux.shape[0] + 1
        )  # Regular numpy array for time
        x_values = np.arange(stacked_flux.shape[1])
        y_values = np.arange(stacked_flux.shape[2])
        npt_values = np.arange(1, stacked_flux.shape[3] + 1)

        # Create the NetCDF file for the summed data
        # Create the NetCDF dataset for the flux data
        ds = xr.Dataset(
            {
                #"time":(["nt"], time_values),
                #"x": (["nx"], x_values),
                #"y": (["ny"], y_values),
                #"n": (["npt"], npt_values),
                "flux": (
                    ["ny", "nx", "nt","npt"],  # Define the dimensions of flux
                    stacked_flux,  # The actual flux data array
                )
            },
            #coords={
            #    "time": time_values,  # Time starts at 1
            #    "y": y_values,  # Spatial y-coordinates
            #    "x": x_values,  # Spatial x-coordinates
            #    "npt": npt_values,  # Flux components (starting at 1)
            #},
        )

        # Add metadata for the flux variable
        #ds["flux"].attrs[
        #    "description"
        #] = "Surface flux data for CO2 and CH4 components."
        #ds["flux"].attrs["units"] = "kg m-2 s-1"

        # Add global metadata
        #ds.attrs["description"] = (
        #    "NetCDF file containing surface flux data for CO2 and CH4. "
        #    "The 'npt' dimension encodes different flux components: "
        #    "[1]: GPP, [2]: Respiration, [3]: Anthropogenic CO2, [4]: Anthropogenic CH4."
        #)
        #ds.attrs["author"] = "Ivan Basic"
        #ds.attrs["creation_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        # Write the dataset to a NetCDF file
        ds.to_netcdf(
            f"/home/b/b381871/basiclab/data/nc_files/{ncName}",
            format="NETCDF4_CLASSIC",
            encoding={
                "flux": {
                    "dtype": "float64",
                    "_FillValue": None,
                },
                #"time": {"dtype": "float64", "_FillValue": None},
                #"y": {"dtype": "float64", "_FillValue": None},
                #"x": {"dtype": "float64", "_FillValue": None},
                #"n": {"dtype": "float64", "_FillValue": None},
            },
        )


# example usage:
#fl = Myflux()
#fl.createNetCDF(ncName="myflux36h.nc")