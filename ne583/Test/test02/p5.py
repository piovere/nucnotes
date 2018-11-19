import numpy as np
import matplotlib.pyplot as plt


def favg(mu, eta, ix, iy):
    s = source[iy, ix]
    den = totxs + 2 * mu / dx + 2 * eta / dy
    num = 2 * mu / dx * left[iy, ix] + 2 * eta / dy * bottom[iy, ix] + s
    if num / den < 0:
        if left[iy, ix] < 0.0:
            #print(left[iy, ix])
            pass
    return num / den

def ftop(favg, fbot):
    rv = 2 * favg - fbot
    if rv < 0:
        print(rv)
    return rv

def fright(favg, fleft):
    rv = 2 * favg - fleft
    if rv < 0:
        print(rv)
    return rv

nx = 600
ny = 600
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
                # Calc average flux for cell (iy, ix) for this mu, eta
                average[iy, ix] = favg(mu, eta, ix, iy)
                # Calc right flux for cell (iy, ix) for this mu, eta
                right[iy, ix] = fright(average[iy, ix], left[iy, ix])
                # Set left flux for cell (ix+1, iy) for this mu, eta
                if ix+1<nx and iy+1<ny:
                    left[iy, ix+1] = right[iy, ix]
                scalar[iy, ix] = average[iy, ix] * wt
            
            for ix in range(nx):
                # Set top flux for each (iy, ix)
                top[iy, ix] = ftop(average[iy, ix], bottom[iy, ix])
            
            if iy+1<ny:
                bottom[iy+1, :] = top[iy, :]
        
        tottop += wt * eta * wt * mu * top

        #print(top)

plt.plot(tottop[-1, :])
plt.show()
