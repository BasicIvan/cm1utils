"""
First is class used in the first paper - a bit confusing and too complicated.

Below is a function 
"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from cm1utils.tools import r, d, p

inletHeights = {
    "1": {"color": "black", "label": "12.5 m", "height": 12.5},
    "2": {"color": "tab:blue", "label": "44.6 m", "height": 44.6},
    "3": {"color": "tab:green", "label": "71.5 m", "height": 71.5},
    "4": {"color": "tab:red", "label": "131.6 m", "height": 131.6},
    "5": {"color": "tab:orange", "label": "212.5 m", "height": 212.5},
}
xpos = {
    "Wslope": {"xh": -2500.0, "title": "W Slope"},
    "Center": {"xh": 0.0, "title": "Valley Center"},
    "Eslope": {"xh": 2500.0, "title": "E Slope"},
    "Ridge": {"xh": 5000.0, "title": "Ridge"},
}

xpos_plat = {
    "Wslope": {"xh": -2500.0, "title": "W Slope"},
    "Center": {"xh": 0.0, "title": "Valley Center"},
    "Eslope": {"xh": 2500.0, "title": "E Slope"},
    "Ridge": {"xh": 6000.0, "title": "Ridge"},
}

def plot_ref_vtower_panels(step: int):
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 9), sharex=True)

    # EXPERIMENT LOOP
    for dsVals in d.dataSpec.values():
        ds = d.open_data(dsVals["path"], ["pt1_TM"])

        # Select every n-th time step
        ds = ds.isel(time=slice(None, None, step))
        # select xh at the three locations
        xh_values = [entry["xh"] for entry in xpos.values()]
        ds = ds.sel(xh=xh_values).mean(dim="yh")
        # interpolate
        interpHeights = [entry["height"] for entry in inletHeights.values()]
        ds = r.xrinterpolate(ds, interpHeights)

        # SUBPLOT LOOP
        for i, xposVals in enumerate(xpos.values()):
            axs = axes[i]
            x = np.linspace(0, 16, len(ds.time))
            if i == 2:
                p.set_xaxis_CEST(axs)

            # Add a text annotation in the upper left corner of the subplot
            p.add_annotation(axs, xposVals["title"], [0.02, 0.92])
            # INLET HEIGHTS (PLOTS) LOOP
            for hVals in inletHeights.values():
                ys = ds.pt1_TM.sel(xh=xposVals["xh"], zh=hVals["height"]).values * 1e6
                plotargs = {
                    "color": hVals["color"],
                    "label": hVals["label"],
                    "alpha": dsVals["alpha"],
                    "linestyle": dsVals["linestyle"],
                }
                axs.plot(
                    x,
                    ys,
                    **plotargs,
                    zorder=i,
                )
                if dsVals["marker"]:
                    axs.scatter(
                        x,
                        ys,
                        **plotargs,
                        marker=dsVals["marker"],
                        zorder=i,
                    ),

    handles = p.get_legend_handles(d.dataSpec)
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    # Adjust layout to prevent clipping of the legend
    plt.subplots_adjust(right=0.8)
    plt.subplots_adjust(hspace=0.12)

    plt.savefig(
        "Figures/virtual_towers_ref", facecolor="white", edgecolor="black", dpi=150
    )
    plt.show()


class VirtualTowers:

    def __init__(
        self,
        ds,
        var: str,
        nth: int,
        interp: str,
        tbi=None,
        tei=None,
        plateau_option=False,
    ):
        """Initialize the class with:
        - ds: dataset
        - var: which varialble(s)
        - nth: n-th time step
        - interp: interpolate ("yes"), or "no"
        - tbi: index of begining time
        - tei: index of ending time
        """

        self.ds = ds
        self.var = var
        self.interp = interp
        if plateau_option == True:
            self.xhs = [entry["xh"] for entry in xpos_plat.values()]
        else:
            self.xhs = [entry["xh"] for entry in xpos.values()]
        self.hgs = [entry["height"] for entry in inletHeights.values()]
        self.time_slice = slice(tbi, tei, nth)  # time slice index
        self.data = self.get_data

    @property
    def ds_select(self):
        """Selects the data: xhs, variable (pt1_TM most of the time)"""
        return self.ds[self.var].sel(xh=self.xhs).isel(time=self.time_slice)

    @property
    def ds_horizontal_mean(self):
        """Calculate the mean along 'yh'"""
        return self.ds_select.mean(dim="yh")

    @property
    def interpolated_heights(self):
        """Interpolate data to the prescribed heights given in inletHeights"""
        interpolated_data = r.xrinterpolate(
            self.ds_horizontal_mean, self.hgs, method="linear"
        )
        sorted_interpolated_data = interpolated_data.sortby(
            "zh"
        )  # Sort along the 'zh' dimension
        return sorted_interpolated_data

    @property
    def nearest_heights(self):
        """Use heights that are nearest to the inletHeights"""
        return self.ds_horizontal_mean.sel(zh=self.hgs, method="nearest")

    @property
    def get_data(self):
        """Gets the data as virtual towers for the experiment given in the instance"""
        if self.interp == "yes":
            return self.interpolated_heights
        else:
            return self.nearest_heights

    def tower_height(self, xpos: float, h: float) -> np.ndarray:
        """
        Get data for a tower at position xpos, at height h.
        Returns:
        1D np.ndarray, 1D np.ndarray (x,ys)
        """
        sorted_data = self.data.sortby(["xh", "zh"])
        ys = sorted_data.sel(xh=xpos, zh=h, method="nearest").values
        x = np.linspace(0, 16, len(ys))
        if self.var.startswith("pt"):
            ys = ys * 1e6
        return x, ys

    def get_peak(self):
        """Gets peaks (in the morning, hence time_slice = ) and their indices.
        Returns: a dictionary:
        key (from xpos): [max_value, max_index]
        """
        out_dict = {}
        for xkey, x in xpos.items():
            arr = self.tower_height(
                x["xh"],
                h=inletHeights["1"]["height"],
            )[1]
            max_value = np.max(arr)
            max_index = np.argmax(arr)
            out_dict[xkey] = [max_value, max_index]
        return out_dict


# Example usage...
# vt = VirtualTowers(ds,"pt1_TM",1,"no")
# data_1D = vt.tower_height(x=xpos, h=inletHeights["1"]["height"])
#   where x is the xh position, and h the inlet height
