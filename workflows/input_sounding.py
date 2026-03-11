import sys
import os

#sys.path.append(os.path.abspath(".."))
from cm1utils.tools import r, d, p
import metpy.calc as mpcalc
from metpy.units import units
import numpy as np


class InputSounding:

    def __init__(
        self,
        dz: float = 5.0,
        ztop: float = 6100.0,
        z_s: float = 40.0,
        z_i: float = 650.0,
        rh: float = 0.4,
        sfcpres=1000.0,
        sfctheta=239.0,
        profilesFromIcon=False,
    ):

        self.dz = dz  # grid spacing
        self.ztop = ztop  # [m] domain height
        self.z_s = z_s  # [m] surface layer height
        self.z_i = z_i  # [m] height at which atmosphere becomes stable
        self.rh = rh  # relative humidity
        self.sfcpres = sfcpres
        self.sfctheta = sfctheta
        self.iconData = None
        if profilesFromIcon:
            self.iconData = get_bm_ds(self.zarray)

    @property
    def zarray(self):
        return np.arange(self.dz, self.ztop + 5 * self.dz, self.dz)

    @property
    def sfc_T(self):
        return mpcalc.temperature_from_potential_temperature(
            self.sfcpres * units.hPa, self.sfctheta * units.kelvin
        )

    @property
    def sfcqv(self):
        return (
            mpcalc.mixing_ratio_from_relative_humidity(
                self.sfcpres * units.hPa, self.sfc_T, self.rh
            ).magnitude
            * 1000
        ).round(3)

    def tempfromtheta(self, theta, p, qv):
        return (
            theta * (p / self.sfcpres) ** (287.0 / 1005.0) * (1.0 + 0.61 * qv / 1000.0)
        )

    def esatpres(self, temp):
        # return 6.11*np.exp(5423.0*(1.0/273.15 - 1.0/temp))
        return 6.113 * (
            np.exp(17.2694 * (temp - 273.15) / (temp - 35.86))
        )  # Magnus formula instead of the one used before

    def ppres(self, lapse_rate=None):
        """
        Compute pressure at height(s) zarray using either:
        - Isothermal barometric formula (if lapse_rate is None)
        - Lapse-rate-adjusted barometric formula (if lapse_rate is provided)

        Parameters:
            zarray     : numpy array of altitudes in meters
            sfcpres    : surface pressure in Pascals
            sfc_T      : surface temperature in Kelvin (default: 288)
            lapse_rate : lapse rate in K/m (default: None for isothermal)

        Returns:
            p : numpy array of pressures in Pascals
        """
        R = 287.0  # J/(kg·K), specific gas constant for dry air
        g = 9.81  # m/s²

        if lapse_rate is None:
            # Isothermal case
            H = R * self.sfc_T / g
            p = self.sfcpres * np.exp(-self.zarray / H)
        else:
            # With lapse rate
            T = self.sfc_T - lapse_rate * self.zarray
            exponent = g / (R * lapse_rate)
            p = self.sfcpres * (T / self.sfc_T) ** exponent

        return p

    def pprestemp(self, theta, p, qv, zarray=zarray, sfcqv=sfcqv):
        grR = 9.81 / 287.0
        temp = self.tempfromtheta(theta, p, qv)
        p = np.zeros(zarray.shape)
        # p[0] = sfcpres - grR*sfcpres*dz/(sfctheta*(1.0+0.61*sfcqv/1000.0))
        p[0] = self.sfcpres - grR * self.sfcpres * self.dz / (
            self.tempfromtheta(self.sfctheta, self.sfcpres, self.sfcqv)
        )
        p[1] = p[0] - grR * p[0] * self.dz / temp[0]

        for k in range(2, len(self.zarray)):
            # p[k] = p[k-2] - 2.0*dz*grR*p[k-1]/temp[k-1]
            p[k] = p[k - 2] - 2.0 * self.dz * grR * p[k - 1] / temp[k]
        return p

    def rsatratio(self, theta, p):
        esat = self.esatpres(self.tempfromtheta(theta))
        return 622.0 * esat / (p - esat)

    def gammas(self, case="const_lapse_rate"):

        if case == "const_lapse_rate":
            gamma_const = 0.003  # 	[K/m] lapse rate (for constant lapse rate case)
            return gamma_const, case

        elif case == "mixed_layer":

            gamma_s = -0.005  # 	surface lapse rate [K/m]
            # gamma_s = 0.
            gamma_m = 0.0  # 	mixing/residual lapse rate [K/m]
            gamma_ez = 0.0035  # 	entrainment zone / free atmosphere lapse rate [K/m]
            # gamma_fa =  0.004   #free atmosphere lapse rate [K/m]

            nk1 = self.z_s / self.dz
            nk2 = (self.z_i - self.z_s) / self.dz
            nk3 = (self.zarray[-1] - self.z_i) / self.dz

            g_bot = np.full(int(nk1), gamma_s)
            g_mid = np.full(int(nk2), gamma_m)
            g_top = np.full(int(nk3), gamma_ez)

            return np.hstack((g_bot, g_mid, g_top)), case

        # elif case == "custom":
        #    return ds.theta

    def thetatemp(self, *param):  # , case='const_lapse_rate'):
        theta = np.zeros(self.zarray.shape)
        param = param[0]

        gamma = param[0]
        case = param[1]

        if case == "const_lapse_rate":
            for k, z in enumerate(self.zarray):
                theta[k] = self.sfctheta + gamma * z

        elif case == "mixed_layer" or case == "near_sunset":

            theta[0] = self.sfctheta
            for k, z in enumerate(self.zarray, start=1):
                if k >= self.zarray.shape[0]:
                    break
                theta[k] = theta[k - 1] + gamma[k - 1] * self.dz

        return theta

    def qvmoist(self, *param, case="const_lapse_rate"):
        qv = np.zeros(self.zarray.shape)
        if case == "const_lapse_rate":
            for k, z in enumerate(self.zarray):
                qv[k] = self.sfcqv + param[0] * z

        elif case == "const_rh":
            qv = param[0] * param[1]  # rh*rsat

        # elif case == "custom":
        #    qv = ds.qv
        return qv

    def wind(self, *param, case="const_wind"):
        u = np.zeros(self.zarray.shape)
        if case == "const_wind":
            u[:] = param[0]
        elif case == "schmidli2013":
            for k, z in enumerate(self.zarray):
                u[k] = min(param[2], max(0.0, (z - param[0]) * param[2] / param[1]))
        elif case == "low_level_wind":
            for k, z in enumerate(self.zarray):
                if z < param[0]:
                    u[k] = param[2]
                elif z >= param[0] and z < (param[1] + param[0]):
                    u[k] = param[2] * ((param[0] + param[1]) - z) / param[1]
                elif z >= (param[1] + param[0]):
                    u[k] = 0.0
        return u

    def generateTheSounding(
        self,
        uWind: float,
        vWind: float,
        gammaCase: str = "mixed_layer",
        humidityCase: str = "const_rh",
        windCase: str = "const_wind",
    ):
        if self.iconData:
            theta = self.iconData.theta
            qv = self.iconData.qv
        else:
            theta = self.thetatemp(*[self.gammas(gammaCase)])
            rsat = self.rsatratio(theta, p)
            qv = self.qvmoist(*[self.rh, rsat], case=humidityCase)

        p = self.pprestemp(theta, p, qv)

        u = self.wind(uWind, case=windCase)
        v = self.wind(vWind, case=windCase)

        return theta, qv, u, v

    def writeToFile(
        self,
        filePath: str,
    ):
        theta, qv, u, v = self.generateTheSounding
        with open(filePath, "w") as f:
            # Write the header line (surface pressure, potential temperature, specific humidity)
            f.write(
                f"  {self.sfcpres:10.4f} {self.sfctheta:12.5f} {self.sfcqv:12.5f}\n"
            )

            # Write the vertical profiles
            for k, z in enumerate(self.zarray):
                # Format each value with the appropriate number of decimal places
                z_str = f"{z:12.4f}"  # Height (z)
                theta_str = f"{theta[k]:12.5f}"  # Potential temperature
                qv_str = f"{qv[k]:12.7E}"  # Specific humidity
                u_str = f"{u[k]:12.7E}"  # U wind component (scientific notation)
                v_str = f"{v[k]:12.7E}"  # V wind component (scientific notation)
                # p_str = f"{p[k]:12.7E}"        # Pressure (scientific notation)

                # Write the formatted line to the file
                f.write(f"{z_str} {theta_str} {qv_str} {u_str} {v_str}\n")  # , {p_str}
        f.close


class bm_profiles:
    def __init__(self) -> None:
        self.sfcqv = 11.30959354  # g/kg
        self.sfcpres = 934.5  # [hPa]
        # self.sfctheta = 302.4  # [K]
        self.sfctheta = 300.0  # new value

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


def get_bm_ds(zarray):
    bm = bm_profiles()

    heights = bm.heights
    thetas = bm.thetas
    qvs = bm.qvs
    rhs = bm.rhs

    heights = heights - heights[-1] + 20

    heights = np.append(heights, 0)
    thetas = np.append(thetas, bm.sfctheta)
    qvs = np.append(qvs, bm.sfcqv)
    rhs = np.append(rhs, rhs[-1])

    heights = np.flip(heights)
    thetas = np.flip(thetas)
    qvs = np.flip(qvs)
    rhs = np.flip(rhs)

    import xarray as xr

    ds = xr.Dataset(
        data_vars=dict(
            theta=(["height"], thetas),
            rh=(["height"], rhs),
            qv=(["height"], qvs),
        ),
        coords=dict(
            height=(["height"], heights),
        ),
        attrs=dict(description="Profiles at Beromünster from ICON"),
    )

    ds = ds.interp(height=zarray)

    return ds
