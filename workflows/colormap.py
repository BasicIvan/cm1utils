from matplotlib.pyplot import cycler
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.cm

def get_cycle(cmap, N=None, use_index="auto"):
    if isinstance(cmap, str):
        if use_index == "auto":
            if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']:
                use_index=True
            else:
                use_index=False
        cmap = matplotlib.cm.get_cmap(cmap)
    if not N:
        N = cmap.N
    if use_index=="auto":
        if cmap.N > 100:
            use_index=False
        elif isinstance(cmap, LinearSegmentedColormap):
            use_index=False
        elif isinstance(cmap, ListedColormap):
            use_index=True
    if use_index:
        ind = np.arange(int(N)) % cmap.N
        return cycler("color",cmap(ind))
    else:
        colors = cmap(np.linspace(1/N,1,N))
        return cycler("color",colors)
    
"""
_____________________________
****************************
USAGE FOR THE CONTINOUS CASE
****************************

import matplotlib.pyplot as plt
N = 6
plt.rcParams["axes.prop_cycle"] = get_cycle("viridis", N)

fig, ax = plt.subplots()
for i in range(N):
    ax.plot([0,1], [i, 2*i])

plt.show()
_____________________________
*****************************
USAGE FOR THE "DISCRETE" CASE
*****************************

import matplotlib.pyplot as plt

plt.rcParams["axes.prop_cycle"] = get_cycle("tab20c")

fig, ax = plt.subplots()
for i in range(15):
    ax.plot([0,1], [i, 2*i])

plt.show()
"""

