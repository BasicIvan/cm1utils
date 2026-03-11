"""
PLOTTING TOOLS - IVAN BASIC
"""
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
import numpy as np
import matplotlib.colors as mcolors
from pathlib import Path
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import datetime as dt

label_size = 12
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": label_size,
    "xtick.labelsize": label_size - 2,
    "ytick.labelsize": label_size - 2,
})

mpl.rcParams["timezone"] = "UTC"


def format_time_axis_with_midnight_date(
    ax,
    hour_interval: int = 4,
    date_format: str = "%d-%b",
    hour_format: str = "%H",
    add_minor_ticks: bool = True,
):
    # Major ticks every N hours
    ax.xaxis.set_major_locator(
        mdates.HourLocator(interval=hour_interval)
    )

    # --- Major formatter: hours only ---
    def major_formatter(x, pos=None):
        dtt = mdates.num2date(x, tz=dt.timezone.utc)
        if dtt.hour == 0:
            return dtt.strftime(f"{hour_format}\n{date_format}")
        return dtt.strftime(hour_format)

    ax.xaxis.set_major_formatter(FuncFormatter(major_formatter))

    if add_minor_ticks:
        # Minor ticks every hour (guarantees midnight exists)
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))

        # --- Minor formatter: only show date at midnight ---
        def minor_formatter(x, pos=None):
            dtt = mdates.num2date(x, tz=dt.timezone.utc)
            if dtt.hour == 0:
                return dtt.strftime(f"\n {date_format}")
            return ""

        ax.xaxis.set_minor_formatter(FuncFormatter(minor_formatter))

        ax.tick_params(axis="x", which="minor", length=3)
        ax.tick_params(axis="x", which="minor", labelsize=label_size)


# Function to plot individual time series
def plot_virtual_tower(ds, ax, hours_of_simulation=36,labels=False):
    height_colors = ["black", "tab:blue", "tab:green", "tab:red", "tab:orange"]
    height_labels = ["12m", "45m", "72m", "132m", "212m"]
    x = np.linspace(0, hours_of_simulation, len(ds.time))  # Common time axis
    for color, height,label in zip(height_colors, ds.z,height_labels):
        y = ds.sel(z=height).values
        if labels:
            ax.plot(ds.time, y, color=color, label=label, alpha=.8)
        else:
            ax.plot(ds.time, y, color=color, alpha=.8)

def save_figure(
    fig: Figure,
    name: str,
    path: str = "/home/b/b381871/basiclab/figures",
    format: str = ".pdf",
    dpi: int = 300,
):
    """
    Save figure to path/name.format
    Works whether path has trailing slash or not.
    Creates directories automatically.
    """

    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)

    filepath = outdir / f"{name}{format}"

    fig.savefig(
        filepath,
        facecolor="white",
        edgecolor="black",
        bbox_inches="tight",
        dpi=dpi,
    )

    return filepath

from pathlib import Path

def set_xaxis_space(
    axs: Axes,
    major_tick_positions: list = [-5, 5],
    minor_tick_positions: list = list(range(-4, 5, 1)),
):
    """Set the x-axis tick labels"""
    axs.xaxis.set_ticks_position("bottom")
    # minor
    axs.xaxis.set_minor_locator(plt.FixedLocator(minor_tick_positions))
    # major
    axs.xaxis.set_major_locator(plt.FixedLocator(major_tick_positions))


def set_xaxis_time(
    xaxis,
    axs: Axes,
    opt="LT",
    custom_major_tick_labels: list = None,
    majInt: int = 3,
    minInt: int = 1,
):
    """Set the x-axis tick labels."""
    axs.xaxis.set_ticks_position("bottom")
    major_tick_positions = list(range(0, int(xaxis[-1]), majInt))
    minor_tick_positions = list(range(0, int(xaxis[-1]), minInt))
    if opt == "CEST":
        major_tick_labels = ["19", "22", "01", "04", "07", "10"]
        axs.set_xlabel("Time (CEST)")
    elif opt == "LT":
        if custom_major_tick_labels:
            major_tick_labels = custom_major_tick_labels
        else:
            major_tick_labels = ["18", "21", "00", "03", "06", "09"]
        axs.set_xlabel("Time (LT)")
    elif opt == "csv":
        major_tick_labels = ["00", "03", "06", "09", "12", "15", "18", "21"]
        axs.set_xlabel("Time (LT)")
    elif opt == "csv_dates":
        major_tick_labels = [
            "07-18",
            "07-19",
            "07-20",
            "07-21",
            "07-22",
            "07-23",
            "07-24",
        ]
        axs.tick_params(axis="x", labelrotation=45)
        major_tick_positions = list(range(0, int(xaxis[-1]), 24))
        minor_tick_positions = list(range(0, int(xaxis[-1]), 3))

    
    # minor
    axs.xaxis.set_minor_locator(plt.FixedLocator(minor_tick_positions))
    # major
    axs.xaxis.set_major_locator(plt.FixedLocator(major_tick_positions))
    axs.xaxis.set_major_formatter(plt.FixedFormatter(major_tick_labels))

    # usage:
    # define xaxis = np.linspace(0, 16, len(ds.time)) in original code before calling this

def get_linestyle(var: str):
    if var == "tke_tot":
        return "-", "TOT"
    elif var == "tke_res":
        return "--", "RES"
    elif var == "tke_sg":
        return ":", "SGS"
    else:
        return "-", ""

def get_legend_handles(
    in_dict: dict,
    specific_design: dict = None,
    with_markers=False,
    with_linestyles=True,
    keys_to_skip: list = [],
):
    """Goes through in_dict and creates legend entries.
    If specific design dict is given, goes through that instead.
    set with_markers=True, if you wish legend with scatter markers.
    if there are keys you want to skip, list them in: keys_to_skip"""
    dictio = in_dict
    if specific_design:
        dictio = specific_design

    # Create an empty list to store legend handles
    handles = []
    # Loop through the dictionary to plot and create legend entries
    for key in dictio:
        if key in keys_to_skip:
            continue
        d = dictio[key]

        if with_markers:
            marker = d.get("marker", None)
        else:
            marker = None
        if with_linestyles:
            linestyle = d.get("linestyle", None)
        else:
            linestyle = None
        alpha = d.get("alpha", 1.0)
        label = d.get("label", in_dict[key]["label"])
        color = d.get("color", "black")

        # Create Line2D objects for legend entry
        marker_line = mlines.Line2D(
            [],
            [],
            color=color,
            marker=marker,
            alpha=alpha,
            linestyle=linestyle,
            label=label,
        )
        # scatter = plt.scatter(
        #    [], [], color=color, marker=marker, label=label, alpha=alpha
        # )

        # Append Line2D objects to handles list
        handles.append(marker_line)
    return handles


def add_annotation_abc(fig, coords: list = [0.02, 0.98]):
    """Adds alphabetical annotation to figure subplots with given coordinates"""
    abc = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)", "(m)", "(n)", "(o)","(p)"]

    for i, axs in enumerate(fig.axes):
        axs.text(
            coords[0],
            coords[1],
            abc[i],
            transform=axs.transAxes,
            fontsize=label_size,
            fontweight="bold",
            verticalalignment="top",
        )

def add_annotation(axs: Axes, annotation: str, coords: list):
    """Adds annotation to axs with given coordinates"""
    axs.text(
        coords[0],
        coords[1],
        annotation,
        transform=axs.transAxes,
        fontsize=label_size,
        fontweight="bold",
        verticalalignment="top",
    )

def get_rxrz(ds, zrange=None, xrange=None, unit="km"):
    xhs = ds.xh.values
    zhs = ds.zh.values
    if unit == "km":
        xhs = xhs / 1000.0
        # zhs = zhs / 1000.0

    rx = [xhs[0], xhs[-1]]
    rz = [zhs[0], zhs[-1]]

    if xrange:
        rx = [xrange[0], xrange[-1]]
    if zrange:
        rz = [zrange[0], zrange[-1]]

    return rx, rz

def make_mesh_grid(ds, zrange=None, xrange=None, unit="km"):
    """Returns X,Z meshgrid"""
    # if vr is 2D
    # vr = vr.values
    # nx,nz = vr.shape
    # x = np.linspace(0, 16, nx)
    # z = np.linspace(0,ztop,nz)+zstart
    # X, Z = np.meshgrid(x, z)

    rx, rz = get_rxrz(ds, zrange, xrange, unit)
    xhs = ds.xh.values
    zhs = ds.zh.values

    # meshgrid
    import numpy as np

    x = np.linspace(rx[0], rx[1], len(xhs))
    z = np.arange(len(zhs))
    return np.meshgrid(x, z)

def plot_cs_orography(axs, ds, yh=0):
    "Plots the orography, if any"
    if "zs" in ds.variables:
        import numpy as np

        xhs = ds.xh.values / 1000.0
        rx = (xhs[0], xhs[-1])
        x = np.linspace(rx[0], rx[1], len(xhs))
        zskm = ds.zs.sel(yh=yh, method="nearest") / 1000.0
        oro = zskm
        axs.fill_between(x, oro, y2=0, color="lightgray", zorder=0)
        axs.plot(x, oro, linewidth=1, color="black", zorder=4)

def load_ncl_colormap(file_path):
    """
    Load a colormap from an NCL .rgb file and return a Matplotlib ListedColormap.

    Parameters:
    - file_path (str): Path to the .rgb file.

    Returns:
    - cmap (Matplotlib ListedColormap): The loaded colormap.
    """
    # Read the RGB values from the file
    rgb_values = []
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comments or empty lines
            if line.strip() and not line.startswith('#'):
                # Each line contains R, G, B values, usually in the range 0-255
                rgb = list(map(int, line.strip().split()[:3]))  # Take only R, G, B
                rgb_values.append(rgb)
    
    # Convert to a numpy array and normalize to 0-1
    rgb_array = np.array(rgb_values) / 255.0
    
    # Create a ListedColormap from the RGB values
    cmap = mcolors.ListedColormap(rgb_array)
    
    return cmap

landuse_dict = {
    0: ("No data", "#000000"),
    1: ("Urban and Built-Up Land", "#FF0000"),
    2: ("Dryland Cropland and Pasture", "#FFFF00"),
    3: ("Irrigated Cropland and Pasture", "#FFA500"),
    4: ("Mixed Dryland/Irrigated Cropland and Pasture", "#800080"),
    5: ("Cropland/Grassland Mosaic", "#32CD32"),
    6: ("Cropland/Woodland Mosaic", "#228B22"),
    7: ("Grassland", "#7FFF00"),
    8: ("Shrubland", "#8B4513"),
    9: ("Mixed Shrubland/Grassland", "#D2B48C"),
    10: ("Savanna", "#FFD700"),
    11: ("Deciduous Broadleaf Forest", "#006400"),
    12: ("Deciduous Needleleaf Forest", "#8B0000"),
    13: ("Evergreen Broadleaf Forest", "#228B22"),
    14: ("Evergreen Needleleaf Forest", "#556B2F"),
    15: ("Mixed Forest", "#66CDAA"),
    16: ("Water Bodies", "#4682B4"),
    17: ("Herbaceous Wetland", "#40E0D0"),
    18: ("Wooded Wetland", "#00CED1"),
    19: ("Barren or Sparsely Vegetated", "#D3D3D3"),
    20: ("Herbaceous Tundra", "#A9A9A9"),
    21: ("Wooded Tundra", "#778899"),
    22: ("Mixed Tundra", "#708090"),
    23: ("Bare Ground Tundra", "#2F4F4F"),
    24: ("Snow or Ice", "#FFFFFF"),
    25: ("Playa", "#F0E68C"),
    26: ("Lava", "#B22222"),
    27: ("White Sand", "#FFFACD"),
    28: ("Unassigned", "#000000"),
    29: ("Unassigned", "#000000"),
    30: ("Unassigned", "#000000"),
    31: ("Low Intensity Residential", "#FF4500"),
    32: ("High Intensity Residential", "#FF6347"),
    33: ("Industrial or Commercial", "#800000"),
}