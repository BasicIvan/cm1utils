"""
DATA RELATED FUNCTIONS (with some dictionaries for help) - IVAN BASIC
"""
import numpy as np
import json
import xarray as xr


def drop_time(ds) -> xr.DataArray:
    if "time" in ds.dims:
        return ds.isel(time=0)
    else:
        return ds

def get_sorted_file_list(data_dir: str, prefix: str = "cm1out_"):
        import os
        file_list = sorted(
            [
                f
                for f in os.listdir(data_dir)
                if f.startswith(prefix) and f.endswith(".nc")
            ]
        )
        return file_list

def open_data(path: str, variables: list = None, missing_value: float = -999999.875):
    if variables is not None:
        ds = xr.open_dataset(path)[variables]
    else:
        ds = xr.open_dataset(path)

    ds = ds.where(ds != missing_value)
    return ds

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy array to list
        return json.JSONEncoder.default(self, obj)


class jsonFile:
    def __init__(self):
        self.my_dict = None

    def write(self, in_dict, filename="my_dict.json"):
        # Saving dict to a JSON file
        with open(filename, "w") as file:
            json.dump(in_dict, file, cls=NumpyEncoder)

    def load(self, filename: str):
        # Loading dict from the JSON file and saves it
        with open(filename, "r") as file:
            self.my_dict = json.load(file)

    def printNice(self, sort=False, indent=4):
        def round_values(d):
            if isinstance(d, dict):
                return {k: round_values(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [round_values(v) for v in d]
            elif isinstance(d, float):
                return round(d, 2)
            else:
                return d

        rounded_dict = round_values(self.my_dict)
        if sort:
            # Sort dictionary items by value in descending order
            out_dict = dict(
                sorted(rounded_dict.items(), key=lambda item: item[1], reverse=True)
            )
        else:
            out_dict = rounded_dict

        print(json.dumps(out_dict, indent=indent))


dataSpec = {
    "ref": {
        "path": "ref1.nc",
        "label": "REF",
        "alpha": 1.0,
        "linestyle": "--",
        "marker": "o",
        "color": "black",
    },
    "U15NE": {
        "path": "U15NE.nc",
        "label": "U15NE",
        "alpha": 1.0,
        "linestyle": ":",
        "marker": "o",
        "color": "grey",
    },
    "U10NE": {
        "path": "U10NE1.nc",
        "label": "U10NE",
        "alpha": 1.0,
        "linestyle": "-",
        "marker": "o",
        "color": "tab:blue",
    },
    "U2.5NE": {
        "path": "U2p5NE1.nc",
        "label": "U2.5NE",
        "alpha": 1.0,
        "linestyle": "--",
        "marker": "x",
        "color": "tab:blue",
    },
    "U5N": {
        "path": "U5N1.nc",
        "label": "U5N",
        "alpha": 1.0,
        "linestyle": "-",
        "marker": ">",
        "color": "tab:green",
    },
    "U5E": {
        "path": "U5E1.nc",
        "label": "U5E",
        "alpha": 1.0,
        "linestyle": "--",
        "marker": "<",
        "color": "tab:green",
    },
    "h700": {
        "path": "h7001.nc",
        "label": "h700",
        "alpha": 1.0,
        "linestyle": "-",
        "marker": "o",
        "color": "tab:orange",
    },
    "h175": {
        "path": "h1751.nc",
        "label": "h175",
        "alpha": 1.0,
        "linestyle": "--",
        "marker": "d",
        "color": "tab:orange",
    },
    "h115": {
        "path": "h115.nc",
        "label": "h115",
        "alpha": 1.0,
        "linestyle": ":",
        "marker": "d",
        "color": "grey",
    },
    # "nbl_bero_constpt": {
    #    "path": "nbl_bero_constpt.nc",
    #    "label": "nbl",
    #    "alpha": 1.0,
    #    "linestyle": "-",
    #    "marker": "o",
    #    "color": "black",
    # },
}


class bm_profiles:
    def __init__(self) -> None:
        self.sfcqv = 11.30959354  # g/kg
        self.sfcpres = 934.5  # [hPa]
        self.sfctheta = 302.4  # [K]

        self.heights = np.array(
            [
                7200.8865948,
                6954.93516781,
                6716.39158764,
                6484.96679823,
                6260.39309524,
                6042.42174869,
                5830.82094151,
                5625.37397573,
                5425.8777066,
                5232.14117221,
                5043.98439163,
                4861.23730949,
                4683.73886829,
                4511.33619293,
                4343.88387434,
                4181.24334142,
                4023.2823117,
                3869.87431305,
                3720.89826986,
                3576.2381479,
                3435.78265336,
                3299.42498204,
                3167.06261566,
                3038.59716262,
                2913.93424146,
                2792.9834057,
                2675.65810946,
                2561.87571388,
                2451.55753528,
                2344.62893664,
                2241.01946527,
                2140.66304071,
                2043.49819856,
                1949.46839812,
                1858.5224045,
                1770.6147596,
                1685.70636164,
                1603.76518004,
                1524.76714281,
                1448.69724868,
                1375.55097851,
                1305.33611493,
                1238.07513356,
                1173.80841787,
                1112.59870194,
                1054.53741733,
                999.75413867,
                948.43138632,
                900.8294465,
                857.33205425,
                818.54305415,
                786.25635396,
                762.09365055,
            ]
        )

        self.thetas = np.array(
            [
                323.14397992,
                322.65690228,
                321.90497186,
                320.95723215,
                320.14777311,
                319.42437917,
                318.79505061,
                318.10860298,
                317.53000053,
                316.96693341,
                316.44798435,
                315.9633052,
                315.55930361,
                315.04187382,
                314.38905377,
                313.57065556,
                313.02647086,
                312.32599818,
                311.48492452,
                310.58919015,
                309.79260087,
                309.25760366,
                308.7892318,
                308.41573847,
                308.01390603,
                307.58989623,
                307.19670695,
                306.75795103,
                306.25249952,
                305.60918881,
                305.15054877,
                304.50710499,
                303.92214101,
                303.45075941,
                303.15449276,
                302.96687667,
                302.67466118,
                302.41211899,
                302.26634006,
                302.19558542,
                302.16274035,
                302.14244369,
                302.1278978,
                302.12024773,
                302.11911282,
                302.12061241,
                302.1236674,
                302.12820455,
                302.13455328,
                302.14360255,
                302.15773932,
                302.18478477,
                302.2641852,
            ]
        )  # K

        self.qvs = np.array(
            [
                0.2456963,
                0.24721996,
                0.25031295,
                0.27571252,
                0.38346204,
                0.45489604,
                0.49337058,
                0.55663427,
                0.63411324,
                0.6989371,
                0.73559532,
                0.7985217,
                0.90737128,
                0.94304774,
                1.00140081,
                1.10079682,
                1.19559033,
                1.20289576,
                1.24169784,
                1.44506155,
                1.69962987,
                1.81025131,
                1.91066228,
                2.01419632,
                2.22037213,
                2.52021623,
                2.58092205,
                2.55573715,
                2.66390381,
                3.17094253,
                3.56809757,
                4.25848966,
                5.10837282,
                5.79645401,
                6.38066975,
                6.72813112,
                7.40395098,
                8.22698534,
                8.66231246,
                8.73480467,
                8.67512784,
                8.58119244,
                8.48184418,
                8.44993901,
                8.45434565,
                8.46611144,
                8.48176529,
                8.50096496,
                8.52444221,
                8.55441234,
                8.59701057,
                8.67268724,
                8.8875075,
            ]
        )  # g/kg

        self.rhs = np.array(
            [
                14.84733052,
                12.95517376,
                11.67648745,
                11.68551642,
                14.72891513,
                15.84688793,
                15.58701021,
                16.10147796,
                16.76765748,
                16.96620064,
                16.42632112,
                16.44452345,
                17.22062733,
                16.68555609,
                16.72612756,
                17.60356856,
                18.05345743,
                17.38019779,
                17.37349789,
                19.70332641,
                22.50632611,
                22.96646226,
                23.19136871,
                23.31388534,
                24.613568,
                26.85781903,
                26.45781674,
                25.33243798,
                25.68964015,
                30.0607293,
                32.94332596,
                38.79982707,
                45.84671672,
                50.98473026,
                54.51569165,
                55.57343089,
                59.59766339,
                64.52252346,
                65.84808687,
                64.18588952,
                61.58839045,
                58.91248249,
                56.38368208,
                54.45473197,
                52.88360842,
                51.48068218,
                50.21938513,
                49.09237073,
                48.09988582,
                47.25184189,
                46.57971567,
                46.20053781,
                46.58328083,
            ]
        )
