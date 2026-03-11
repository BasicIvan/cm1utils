#!/usr/bin/env python

def calc_grid(str_bot, str_top, ztop, dz_bot, dz_top):
    dz_ave = 0.5*(dz_bot+dz_top)
    print('bot:  ', str_bot)
    print('top:  ', str_top)
    print('ztop: ', ztop)
    print('dz_bot', dz_bot)
    print('dz_top', dz_top)
    print('dz_ave', dz_ave)
    print()

    nk1 = str_bot / dz_bot
    nk3 = (ztop - str_top) / dz_top
    nk2 = (str_top - str_bot)/ dz_ave
    nz =  nk1+nk2+nk3
    dz = ztop/nz
    print('nk1: ', nk1)
    print('nk2: ', nk2)
    print('nk3: ', nk3)
    print('nz:  ', nz)
    print('dzm: ', dz)
    print()
    print()
    return

#print('DX=DZ=40m')
#calc_grid(40, 2800, 5000, 8, 40)
#print('DX=DZ=20m')
calc_grid(100, 1000, 3000, 4, 20 ) # example for LES
#print('DX=DZ=10m')
#calc_grid(40, 2800, 5000, 2, 10)
#print('DX=DZ=5m')
#calc_grid(40, 2800, 5000, 1, 5)

calc_grid(100, 1000, 3000, 4, 20 ) # example for LES
