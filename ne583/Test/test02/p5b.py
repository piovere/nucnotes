import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit
def favg(mu, eta, bottom, left, source, dx, dy):
    s = source
    den = totxs + 2 * mu / dx + 2 * eta / dy
    num = 2 * mu / dx * left + 2 * eta / dy * bottom + s
    
    return num / den

@jit
def fnext(favg, fprev):
    rv = 2 * favg - fprev
    return rv

nx = 12000
ny = 12000
nang = 2

totxs = 1.0

width = 6 / totxs
height = 6 / totxs

dx = width / nx
dy = height / nx

mus = np.array([0.3500212, 0.8688903])
wt = 0.3333333
etas = np.array([0.3500212, 0.8688903])

left = np.zeros(nx)
right = np.zeros(nx)
top = np.zeros(nx)
bottom = np.zeros(nx)
average = np.zeros(nx)

tottop = np.zeros(nx)

source = np.zeros((ny, nx))
for ix in range(ny):
    for iy in range(nx):
        if ix*dx<1/totxs and iy*dy<1/totxs:
            source[iy, ix] = 36.0 / nx / ny

scalar = np.zeros((ny, nx))

for ieta in range(nang):
    eta = etas[ieta]

    for imu in range(nang):
        mu = mus[imu]
        
        for iy in range(ny):

            for ix in range(nx):
                s = source[iy, ix]
                # Calc average flux for cell (iy, ix) for this mu, eta
                b = bottom[ix]
                l = left[ix]
                average[ix] = favg(mu, eta, b, l, s, dx, dy)
                # Calc right flux for cell (iy, ix) for this mu, eta
                right[ix] = fnext(average[ix], l)
                # Set left flux for cell (ix+1, iy) for this mu, eta
                if ix+1<nx and iy+1<ny:
                    left[ix+1] = right[ix]
            
            for ix in range(nx):
                # Set top flux for each (iy, ix)
                top[ix] = fnext(average[ix], bottom[ix])
            
            if iy+1<ny:
                bottom = np.copy(top)
        
        tottop += wt * eta * wt * mu * top

        #print(top)

plt.plot(np.linspace(0, width, nx), tottop)
for mu in mus:
    plt.axvline(x=mu*6, color='r', alpha=0.3, linestyle=':')
plt.savefig('test2problem5.png', dpi=300)
