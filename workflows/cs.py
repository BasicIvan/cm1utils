from cm1utils.tools import r, d, p
import numpy as np

class CrossSection:
    def __init__(self, ds):
        self.ds = ds

    def td(self, time):
        """Selects a time step in the dataset"""
        if type(time) == str:
            return self.ds.sel(time=time)
        elif type(time) == int:
            return self.ds.isel(time=time)
        else:
            return self.ds

    def calc_tke(self, time, opt: str = "res"):
        """Calculates resolved or total tke of the dataset for the given time"""
        d = self.td(time)

        uu = d.u_u
        vv = d.v_v
        ww = d.w_w
        utm = d.u_TM
        vtm = d.v_TM
        wtm = d.w_TM
        tke_sg = d.tke_sg
        tke_res = 0.5 * (uu + vv + ww) - 0.5 * (utm**2 + vtm**2 + wtm**2)

        if opt == "res":
            return tke_res
        elif opt == "tot":
            return tke_res + tke_sg
        
    def calc_turbPtW(self,time):
        d = self.td(time)
    
        totalTransport = d.pt1_w.values*1e6  # Total transport (wc) (z, y, x)
        w_mean = d.w_TM.values  # Vertical wind speed (z, y, x)
        c_mean = d.pt1_TM.values*1e6  # Tracer concentration (z, y, x)

        meanTransport = w_mean * c_mean  # Mean transport (y, x)

        # Compute turbulent transport
        turbulentTransport = totalTransport - meanTransport  # Shape: (z, y, x)

        return turbulentTransport

    def calc_advPtW(self, time, horizontalOrVertical = "vertical"):
        d = self.td(time)
    
        w = d.w_TM.values  # Vertical wind speed (z, y, x)
        u = d.u_TM.values
        v = d.v_TM.values
        c = d.pt1_TM.values*1e6  # Tracer concentration [ppm] (z, y, x)

        # vartical w
        dc = np.diff(c, axis=0)  # difference in concentration (z-1, y, x)
        w_mid = 0.5 * (w[:-1] + w[1:])  # Interpolate wind to midpoints
        dz = np.diff(d.zh)[:, np.newaxis, np.newaxis]  # Vertical grid spacing
        wAdvectiveTransport = w_mid * (dc / dz) # Advective transport (z-1, y, x)

        # horizontal u,v
        dcx = np.diff(c, axis=2)
        u_mid = 0.5 * (u[:,:,:-1] + u[:,:,1:])
        dx = np.diff(d.xh)[np.newaxis, np.newaxis, :]

        dcy = np.diff(c, axis=1)
        v_mid = 0.5 * (v[:,:-1,:] + v[:,1:,:])
        dy = np.diff(d.yh)[np.newaxis, :, np.newaxis]
        
        # Now we need to put both on the common grid: cell center, i.e., (z, y-1, x-1)
        # First compute both transports at their staggered locations
        ux = u_mid * (dcx / dx)  # (z, y, x-1)
        vy = v_mid * (dcy / dy)  # (z, y-1, x)

        # Now average them to (z, y-1, x-1)
        ux_center = 0.5 * (ux[:, :-1, :] + ux[:, 1:, :])  # (z, y-1, x-1)
        vy_center = 0.5 * (vy[:, :, :-1] + vy[:, :, 1:])  # (z, y-1, x-1)

        hAdvectiveTransport = ux_center + vy_center  # (z, y-1, x-1)
        print(f"{wAdvectiveTransport.shape}; {hAdvectiveTransport.shape}")

        # Pad horizontal transport to match vertical transport shape (if needed)
        #hAdvectiveTransport = np.pad(hAdvectiveTransport, ((0, 0), (0, 0), (0, 1)), mode='edge')

        if horizontalOrVertical == "vertical":
            return wAdvectiveTransport
        else:
            return hAdvectiveTransport

    def extract_aux_vars(self, time, ymean=True):
        """Extract variables and calculate means. Sets them as class attributes."""
        variables = ["th_TM", "u_TM", "v_TM", "w_TM"]
        for var_name in variables:
            # var = self.td(time)[var_name]
            # setattr(self, var_name, var.mean(dim="yh").values)
            if ymean:
                setattr(self, var_name, self.get_ymean(var_name, time))
            else:
                setattr(self, var_name, self.td(time)[var_name])

    def get_ymean(self, var: str, time: str):
        """ymean of the variable at the selected time step. Returns ndarray."""
        d = self.td(time)

        if var == "tke_res":
            vr = self.calc_tke(time, "res")
        elif var == "tke_tot":
            vr = self.calc_tke(time, opt="tot")
        elif var == "turb_pt_w":
            vr = self.calc_turbPtW(time)
            return np.mean(vr,axis=1)
        elif var == "adv_pt_w":
            vr = self.calc_advPtW(time)
            return np.mean(vr,axis=1)
        else:
            vr = d[var]
        vr = vr.mean(dim="yh")
        if var.startswith("pt1"):
            return vr.values * 1e6
        else:
            return vr.values
