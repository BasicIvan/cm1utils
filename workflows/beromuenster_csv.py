from pandas import read_csv
import pandas as pd
from cm1utils.tools import r, d, p
import numpy as np

csv_files = {
    "12m": {
        "path": "/home/b/b381871/basiclab/data/Beromuenster_measurements_new/Beromuenster_UniBern_2021_10min_12m.csv",
        "label": "12 m",
        "color": "Black",
    },
    "45m": {
        "path": "/home/b/b381871/basiclab/data/Beromuenster_measurements_new/Beromuenster_UniBern_2021_10min_45m.csv",
        "label": "45 m",
        "color": "tab:blue",
    },
    "72m": {
        "path": "/home/b/b381871/basiclab/data/Beromuenster_measurements_new/Beromuenster_UniBern_2021_10min_72m.csv",
        "label": "72 m",
        "color": "tab:green",
    },
    "132m": {
        "path": "/home/b/b381871/basiclab/data/Beromuenster_measurements_new/Beromuenster_UniBern_2021_10min_132m.csv",
        "label": "132 m",
        "color": "tab:red",
    },
    "212m": {
        "path": "/home/b/b381871/basiclab/data/Beromuenster_measurements_new/Beromuenster_UniBern_2021_10min_212m.csv",
        "label": "212 m",
        "color": "tab:brown",
    },
}


class manipulate_CSV:
    def __init__(self,interpolate:str="No") -> None:
        self.extracted_data = None
        self.interpolate = interpolate
        pass

    def extract_datetime_interval(self, csv_data, start_date: str, end_date: str):
        """Convert the date and time column to a datetime object"""
        csv_data["Date"] = pd.to_datetime(csv_data["Date"])
        # Use boolean indexing to filter the DataFrame based on the date and time range
        self.extracted_data = csv_data.loc[
            (csv_data["Date"] >= start_date) & (csv_data["Date"] <= end_date)
        ]
        if self.extracted_data.empty:
            print(f"No data found between {start_date} and {end_date}")
        # return extracted_data

    def select_column(self, col: str):
        """Select the column and interpolate."""
        if self.interpolate == "Yes":
            return self.extracted_data[col].interpolate()
        else:
            return self.extracted_data[col]

    def inspect_columns(self):
        """Print available columns in the extracted data"""
        if self.extracted_data is not None:
            print("Available columns:", self.extracted_data.columns.tolist())
        else:
            print("No data loaded yet. Please run extract_datetime_interval first.")

    def get_diurnal_cycle(self, col):
        """Calculates the diurnal cycle. The input data must be multiple of 144."""
        data = self.select_column(col).values[
            :-1
        ]  # because the shape is invalid: it takes +1 element from the csv file
        data = data.reshape(-1, 144)  # shapes it to: (days, times)
        # Calculate the mean along the days
        mean_diurnal_cycle = np.mean(data, axis=0)

        return mean_diurnal_cycle


def get_observations(col="CO2_dry", start_date = "2021-07-21 17:00:00", end_date = "2021-07-23 17:00:00",interpolate="No"):

    bm_dict = {
        "CO2_dry": {},
        # "CH4_dry":{},
        # "CO": {},
        "winddirection": {},
        "windspeed": {},
        "relativehumidity": {},
        "Date": {},
        "temperature": {},
    }
    csvm = manipulate_CSV(interpolate)
    csvm_dc = manipulate_CSV(interpolate)
    d = {}
    dc = {}

    for hgs in csv_files:
        # Load data from path
        data = read_csv(csv_files[hgs]["path"], sep=";")

        # for diurnal cycle
        start_date_dc = "2021-07-01 00:00:00"
        end_date_dc = "2021-08-01 00:00:00"

        # store data in the class to access it later (extracted_data)
        csvm.extract_datetime_interval(data, start_date, end_date)
        csvm_dc.extract_datetime_interval(data, start_date_dc, end_date_dc)

        csvm.extracted_data = csvm.extracted_data.set_index("Date")

        d[hgs] = csvm.select_column(col)
        dc[hgs] = csvm_dc.get_diurnal_cycle(col)  # if diurnal cycle is needed...

    return d, dc


def pt_diurnal_cycle(height):
    # Number of points in one day = 24h * 6 points/hour = 144
    diurnal_points = 24 * 6

    # Shift by -6h = 6 * 6 = 36 indices
    shift = 18 * 6

    d, dc = get_observations()
    # Get the raw diurnal cycle (assumed to be of length 144)
    y_obs_dc = dc[height]  # from your diurnal cycle dict

    # Convert to numpy array if not already
    y_obs_dc = np.array(y_obs_dc)

    # Apply -6h shift (rotate left)
    y_obs_dc_shifted = np.roll(y_obs_dc, -shift)

    # Now repeat to cover 36h (216 points)
    repeats = 216 // len(y_obs_dc_shifted) + 1  # repeat enough times
    y_obs_dc_repeated = np.tile(y_obs_dc_shifted, repeats)[:216]

    # Optional: center like y_obs
    y_obs_dc_repeated -= 410
    x_dc = np.linspace(0, 36, len(y_obs_dc_repeated))

    return x_dc, y_obs_dc_repeated
