"""
creates a input_grid_z file for stretch_z = 4 switch in input.namelist
for CM1 v21.0
"""
import sys
import os
sys.path.append(os.path.abspath(".."))
from cm1utils.tools import r, d, p
import numpy as np
    
def get_layer(ls,le,dz_bot,dz_top):
    # ls = layer start, le = layer end
    # dz_bot = stretching at bottom, dz_top = -II- top
    
    if ls > le: raise Exception("layer start is higher than layer end!")
    
    ld = le - ls
    
    if dz_bot == dz_top: case='const'
    elif dz_bot < dz_top: case='stretch'
    
    if case=='const': 
        # returns grid spacing with constant dz
        dz = dz_bot
        
        #if ld%dz != 0: raise Exception("layer depth (le - ls) does not divide exacly by dz!")
            
        nk = ld/dz; nk = int(nk)
        
        dzs = np.full(nk,dz)
    
    elif case=='stretch':
        # returns grid spacing with increasing dz
        dz_ave = 0.5*(dz_bot+dz_top)
        nk = ld/dz_ave; nk = int(nk)
        
        dzs = np.geomspace(dz_bot,dz_top,nk)
        if dzs[1]/dzs[0] > 1.1: 
            raise Exception("stretching too high! Increase layer depth, or reduce difference between dz_top and dz_bot!")
            
        # the following line makes sure that the ending number is close to the requested end height...    
        while ld>np.sum(dzs[:-1]):
            nk+=1
            dzs = np.geomspace(dz_bot,dz_top,nk)    
    else: 
        raise Exception("dz_bot is higher than dz_top!")
        
    zarr = np.zeros_like(dzs)
    
    for k, z in enumerate(zarr):
        if k==0: 
            zarr[k] = ls
        else:
            zarr[k] = zarr[k-1] + dzs[k-1]
    ls_new = zarr[-1]+dzs[-1]
    
    return zarr,ls_new
    # since stretching "disturbs" the layer boundaries, 
    # we have to have a new layer start, 
    # which is in the radius of dz_top around le.
        
"""
---- layers -----
modify the layers here.
just add or delete levels as needed
ls = layer start, le = layer end
dz_bot = grid spacing at bottom, dz_top = -II- top
Note: everything is in meters!
"""

# ------1------
ls      = 0
le      = 600
dz_bot  = 5
dz_top  = 5

l1,ls_new = get_layer(ls,le,dz_bot,dz_top)
# -------------
# ------2------
le      = 2000
dz_bot  = 5
dz_top  = 50

l2,ls_new = get_layer(ls_new,le,dz_bot,dz_top)
# -------------
# ------3------
le      = 6150
dz_bot  = 50
dz_top  = 50

l3,ls_new = get_layer(ls_new,le,dz_bot,dz_top)
# -------------

# merge the layers together (don't forget to add or remove layers form np.hstack)
zarray = np.hstack((l1,l2,l3))

# write to file (you have to specify your file path)
with open('/home/b/b381871/cm1/RUN_CM1/INPUT/input_grid_z_5to50', 'w') as f:
    for k, z in enumerate(zarray):
        f.write("%.2f\n" % z)
    f.write(" ")
f.close()
lenz = len(zarray)-1
zarrayLast = zarray[-1]
print("namelist.input: don't forget to set nz = %d" %lenz)
print("mind that: ztop = {}".format(zarrayLast))