from cm1utils.tools import r, d, p
import numpy as np

class BulkValue:
    """Calculate bulk value"""

    def __init__(self, ds, xh_slice, time_slice=None, ztop: float = 350.0):
        """Initialize the class with dataset and height bounds"""
        self.ds = ds.sel(xh=xh_slice)  # xh_slice  # float or slice
        self.ztop = ztop
        if time_slice:
            self.ds = self.ds.sel(time=time_slice)

    def z_select(self, z0, z1):
        """Returns the dataset with selected heights"""
        return self.ds.sel(zh=[z0, z1], method="nearest")

    def z_id_select(self, ds, zh_index: int):
        """Returns the height at zh_index (first or last)"""
        return ds.isel(zh=zh_index).values

    def ds_horizontal_mean(self, ds, variables_of_interest: list):
        """Calculate the mean along 'yh' and 'xh' (if any) dimensions for variables of interest"""
        hmean = ds[variables_of_interest].mean(dim="yh")
        if "xh" in hmean.dims:
            hmean = hmean.mean(dim="xh")
        return hmean

    def set_vars(self, ds, list_of_vars):
        for var_name in list_of_vars:
            setattr(self, var_name + "0", self.z_id_select(ds[var_name], 0))
            setattr(self, var_name + "1", self.z_id_select(ds[var_name], -1))

    def brn(self, z0: float, z1: float):
        """Calculate bulk Richardson number (BRN) for the given layer"""
        list_of_variables = ["th_TM", "u_TM", "v_TM", "w_TM", "qv_TM"]
        hmean = self.ds_horizontal_mean(self.z_select(z0, z1), list_of_variables)
        self.set_vars(hmean, list_of_variables)

        def virtual_temp(theta, qv):
            return theta * (1 + 0.61 * qv)

        # set the variables
        g = 9.81
        theta = (self.th_TM0 + self.th_TM1) / 2.0
        qv = (self.qv_TM0 + self.qv_TM1) / 2.0
        Tv = virtual_temp(theta, qv)
        dTv = virtual_temp(self.th_TM1, self.qv_TM1) - virtual_temp(
            self.th_TM0, self.qv_TM0
        )
        dz = z1 - z0
        dU = self.u_TM1 - self.u_TM0
        dV = self.v_TM1 - self.v_TM0
        dW = self.w_TM1 - self.w_TM0

        # calculate BRN
        nom = g / Tv * dTv * dz
        denom = dU**2 + dV**2 + dW**2

        # Avoid division by zero by setting small values to a minimum threshold
        min_denom_threshold = 0.01
        denom = np.maximum(denom, min_denom_threshold)

        brn = nom / denom

        # Avoid huge values by setting the maximum threshold value
        max_brn_threshold = 20.0
        min_brn_threshold = -20.0
        brn = np.minimum(np.maximum(brn, min_brn_threshold), max_brn_threshold)
        return brn

    def brn_surface(self):
        """Returns the surface bulk richardson number from the model output"""
        return self.ds_horizontal_mean(self.x_select, "br")

    def dimensionless_valley_height(
        self, z0: float, z1: float, time_average: str = "no"
    ):
        """Calculate dimensionless valley heights as in Sheridan (2019): NH/U"""
        # list_of_variables = ["th_TM", "u_TM", "v_TM"]
        list_of_variables = ["u_TM", "v_TM"]
        hmean = self.ds_horizontal_mean(self.z_select(z0, z1), list_of_variables)
        self.set_vars(hmean, list_of_variables)
        # Tv = (self.th_TM0 + self.th_TM1) / 2.0
        # dT = self.th_TM1 - self.th_TM0
        # dz = z1 - z0
        # g = 9.81

        # U = (
        #    np.sqrt((self.u_TM0 + self.u_TM1) ** 2 + (self.v_TM0 + self.v_TM1) ** 2)
        # ) / 2
        U = (self.u_TM0 + self.u_TM1) / 2
        H = self.ztop
        # N = np.sqrt(abs(g / Tv * dT / dz))
        N = np.sqrt(
            abs(
                self.ds_horizontal_mean(self.ds.sel(zf=slice(z0, z1)), "nm").mean(
                    dim="zf"
                )
            )
        )
        # N = np.sqrt(abs(self.ds.nm.sel(zf=slice(z0,z1)).mean(dim=("zf","xh","yh"))))
        result = abs(N * H / U)

        if time_average == "yes":
            return np.average(result)
        else:
            return np.array(result)


# Example usage...
# bv = BulkValue(ds, xh_slice=0.)
# brn = bv.brn(z1=0., z2=100.)
