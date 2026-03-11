"""Time series of volume average up to a certain height"""

import numpy as np
from cm1utils.tools.r import find_nearest_index
import cm1utils.tools.constants as C


class VolumeAverage:
    """Calculate volume average"""

    def __init__(
        self,
        ds,
        ztop: float,
    ):
        """Initialize the class with dataset and the top height"""
        self.ztop = ztop
        self.ds = ds[["pt1_TM", "zhval", "th_TM", "hfx", "prs_TM", "qv_TM", "rho"]].sel(
            zh=slice(None, ztop)
        )

        self.dz = ds.zh.values[1] - ds.zh.values[0]
        self.gcv = self.grid_cell_volumes

    @property
    def zhvals(self):
        """Returns zhvals (type: ndarray)"""
        return self.ds.zhval.isel(time=0).values

    @property
    def grid_cell_volumes(self):
        """3d ndarray of grid cell volumes."""
        # grid spacings
        dz = np.diff(self.zhvals, axis=0)
        # dx = np.diff(self.ds.sel(zh=self.ztop).xh.values, axis=0)
        # dy = np.diff(self.ds.sel(zh=self.ztop).yh.values, axis=0)
        self.dx = self.ds.xh.values[1] - self.ds.xh.values[0]
        self.dy = self.ds.yh.values[1] - self.ds.yh.values[0]

        # replicate the last vertical layer on the z-axis to the first position
        dz = self.replicate_last_layer(dz)

        return dz * self.dx * self.dy

    def replicate_last_layer(self, in_array3D):
        """Replicates the last vertical layer of an 3D ndarray and adds it on the first axis"""
        # Replicate the last vertical layer along the z-axis
        last_layer = in_array3D[-1, :, :]
        # Add a new dimension along the first axis
        last_layer = last_layer[np.newaxis, :, :]
        return np.vstack((in_array3D, last_layer))

    @property
    def gcv_filtered(self):
        return self.filter_below_ztop(self.gcv)

    @property
    def total_volume(self):
        "Calculates the total volume below the defined height ztop"
        total_volume = np.sum(self.gcv_filtered)
        return total_volume

    def filter_below_ztop(self, in_arr):
        """Filters out values below ztop using boolean mask."""
        below_set_zs = self.zhvals < self.ztop  # boolean mask
        return in_arr[below_set_zs]
    
    def filter_the_height(self, in_arr, height, tolerance=2.49):
        """
        Filter values within a tolerance range from an array.

        Parameters:
            in_arr (numpy.ndarray): Input array.
            height (float): Target height around which to filter values.
            tolerance (float): Tolerance range (+/-) around the target height.

        Returns:
            numpy.ndarray: Filtered array containing values within the tolerance range of the target height.
        """
        lower_bound = height - tolerance
        upper_bound = height + tolerance

        # Boolean mask to filter values within the tolerance range of zhvals
        mask = np.logical_and(self.zhvals > lower_bound, self.zhvals < upper_bound)

        return in_arr[mask]

    def get_weighted_mass(
        self, time: str, var: str = None, diff: float = 0.0, in_arr=None,
    ):
        if (var and in_arr) is not None:
            mass1 = self.ds[var].sel(time=time).values - diff
            mass2 = in_arr
            mass = mass1 * mass2
        elif var:
            mass = self.ds[var].sel(time=time).values - diff
        else:
            mass = in_arr - diff

        wMass = mass * self.gcv
        wMass = self.filter_below_ztop(wMass)
        return wMass

    def calc_volume_average(
        self,
        time: str = None,
        tbi=None,
        tei=None,
        tstep=6,
        var: str = None,
        diff: float = 0.0,
        in_arr=None,
        divide_by_volume=True,
    ):
        """Calculates the volume average of the value enclosed by ztop as a time series.
        tbi = time begin index
        tei = time end index
        tstep = time step
        diff = difference with respect to a certain value? mainly added for Qreq calculation
        """
        if time:
            va = np.sum(
                self.get_weighted_mass(time=time, var=var, diff=diff, in_arr=in_arr)
            )

        else:
            va = []
            for t in self.ds.time[tbi:tei:tstep]:
                total_mass = np.sum(
                    self.get_weighted_mass(t, var=var, diff=diff, in_arr=in_arr)
                )
                va.append(total_mass)
            va = np.array(va)

        if divide_by_volume:
            return va / self.total_volume
        else:
            return va

    def find_closest_value(self,arr, target, tolerance=5.):
        """Find the first value in a 1D numpy array that is close to a target value within a specified tolerance.

        Parameters:
            arr (numpy.ndarray): 1D numpy array to search through.
            target (float): Target value.
            tolerance (float): Tolerance range within which a value is considered close to the target.

        Returns:
            float: The first value found in the array that is close to the target within the specified tolerance.
            None: If no such value is found within the tolerance."""

        # Calculate absolute differences between target and array elements
        abs_diff = np.abs(arr - target)

        # Find indices where absolute differences are within tolerance
        close_indices = np.where(abs_diff <= tolerance)[0]

        # If there are no values within the tolerance, return None
        if len(close_indices) == 0:
            return None

        # Return the first and the last value within the tolerance range
        #return arr[close_indices[0]], arr[close_indices[-1]]
        # Return the index of the first value within the tolerance range
        return close_indices[0],close_indices[-1]
        #return close_indices

    def calculate_area_average(self, var:str, time:str="12:30:00", z: float = None):
        # theta at the height of the crest aa = var(zh = ztop)
        aa = []
        zhvals = self.ds["zhval"].isel(time=0, yh=0)
        # finds where to start and where to end to not waste compute time
        xb,xe = self.find_closest_value(zhvals.isel(zh=0).values, z)
        for x in self.ds.xh[xb:xe].values:
            zhIndex = find_nearest_index(zhvals.sel(xh=x).values, z)
            if zhIndex is not None:
                aa.append(
                    self.ds[var]
                    .sel(time=time, xh=x)
                    .isel(zh=zhIndex)
                    .mean(dim="yh")
                )
        aa = np.array(aa)
        return np.mean(aa)
    
    def vertical_profile_area_average(self, var:str, time:str="12:30:00"):
        vp = []
        for z in self.ds.zh.values:
            aa = self.calculate_area_average(var=var,time=time,z=z)
            if aa is not None:
                vp.append(aa)
                print(str(z)+"...")
            
        return np.array(vp)
    
    def calculate_area_average_bool(self, var:str, time:str="12:30:00", z: float = None):
        # theta at the height of the crest aa = var(zh = ztop)
        mass = self.ds[var].sel(time=time).values
        wMass = mass * self.gcv
        wMass = self.filter_the_height(wMass,z)
        total_volume = np.sum(self.filter_the_height(self.gcv,z))
        total_mass = np.sum(wMass)
        aa = total_mass / total_volume
        return aa
    
    def vertical_profile_area_average_bool(self, var:str, time:str="12:30:00"):
        vp = []
        for z in self.ds.zh.values:
            aa = self.calculate_area_average_bool(var=var,time=time,z=z)
            vp.append(aa)
            print(str(z)+"...")
        return np.array(vp)
    
    def calculate_domain_average_2d(self,var:str,time:str=None):
        
        if time:
            return self.ds[var].sel(time=time).mean(dim=("yh","xh")).values
        else:
            return self.ds[var].mean(dim=("yh","xh")).values
            

    def rho(self, time):
        # https://www.omnicalculator.com/physics/air-density#
        r = C.RD / C.CP_DRY
        # r = 0.286
        w = self.ds["qv_TM"].sel(time=time).values
        prs = self.ds["prs_TM"].sel(time=time).values
        th = self.ds["th_TM"].sel(time=time).values
        rho_dry = self.ds["rho"].sel(time=time).values
        temp = th / ((934.5 / prs) ** r)

        # saturation water vapor pressure p1 at given temperature
        es = 0.6078 * 2.71828 ** (17.27 * (temp - 273.15) / ((temp - 273.15) + 237.03))
        # saturation mixing ratio
        ws = 0.622 * es / (prs - es)
        rh = w / ws
        pv = es * rh
        #pdry = prs - pv
        #rho = pdry / (Rd * temp) + pv / (Rv * temp)
        rho = rho_dry + pv / (C.RV * temp)
        return rho
    
    def rho_ne(self, time):
        # https://www.omnicalculator.com/physics/air-density#
        r = C.RD / C.CP_DRY
        # r = 0.286
        w = self.ds["qv_TM"].sel(time=time).values
        prs = self.ds["prs_TM"].sel(time=time).values
        th = self.ds["th_TM"].sel(time=time).values
        rho_dry = self.ds["rho"].sel(time=time).values
        #temp = th / ((934.5 / prs) ** r)
        temp = ne.evaluate("th / ((934.5 / prs) ** r)")

        # saturation water vapor pressure p1 at given temperature
        #es = 0.6078 * 2.71828 ** (17.27 * (temp - 273.15) / ((temp - 273.15) + 237.03))
        es = ne.evaluate("0.6078 * 2.71828 ** (17.27 * (temp - 273.15) / ((temp - 273.15) + 237.03))")
        # saturation mixing ratio
        #ws = 0.622 * es / (prs - es)
        ws = ne.evaluate("0.622 * es / (prs - es)")
        rh = w / ws
        pv = es * rh
        #pdry = prs - pv
        #rho = pdry / (Rd * temp) + pv / (Rv * temp)
        #rho = rho_dry + pv / (Rv * temp)
        rho = ne.evaluate("rho_dry + pv / (Rv * temp)")
        return rho

    @property
    def Qreq(self):
        """Same as calc_total_mass, but with two values as var, at one time step)"""
        tm = "12:30:00"
        #dz = 5.
        thetaE = self.calculate_area_average("th_TM",time="12:30:00",z=self.ztop)
        Qreq = -self.calc_volume_average(
            time=tm,
            var="th_TM",
            diff=thetaE,
            in_arr=self.rho(time=tm),
            divide_by_volume=False,
        )
        Qreq = Qreq * C.CP #/ dz
        return Qreq

    def calc_inverse_breakup_parameter(self,tb = "12:30:00"):
        """Calculate the inverse breakup parameter as a time series (Following Leukauf (2016))
        Qreq  = the energy required to remove the inversion, evaluated at sunrise
        Qprov = the energy provided by the surface sensivle heat flux until the time of the breakup
        Return: Qprov / Qreq
        """
        # hfx = sensible heat flux (time, yh, xh)
        # Qreq: weighted mass of theta * rho
        time_step = 600.0 # seconds
        dxdy = 100.
        Qprov = self.ds["hfx"].sel(time=slice(tb, None)).values * time_step * dxdy
        Qprov = np.sum(Qprov, (1, 2))
        Qprov = np.cumsum(Qprov)

        return Qprov / self.Qreq
