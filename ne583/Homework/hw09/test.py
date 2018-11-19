import numpy as np
import matplotlib.pyplot as plt


nx = 18
ny = 18
nang = 2

totxs = 1.0

width = 6 / totxs
height = 6 / totxs

dx = width / nx
dy = height / nx

mus = np.array([0.3500212, 0.8688903])
wt = 0.3333333
etas = np.copy(mus)

left = np.zeros((nx, ny))
right = np.zeros((nx, ny))
top = np.zeros((nx, ny))
bottom = np.zeros((nx, ny))
average = np.zeros((nx, ny))

tottop = np.zeros((nx, ny))

source = np.zeros((nx, ny))
for ix in range(nx):
    for iy in range(ny):
        if ix*dx<1/totxs and iy*dy<1/totxs:
            source[ix, iy] = 1.0
print(source)

scalar = np.zeros((nx, ny))

for ieta in range(nang):
    eta = etas[ieta]

    for imu in range(nang):
        mu = mus[imu]
        
        for iy in range(ny):

            for ix in range(nx):
                # Calc average flux for cell (ix, iy) for this mu, eta
                average[ix, iy] = favg(mu, eta, ix, iy)
                # Calc right flux for cell (ix, iy) for this mu, eta
                right[ix, iy] = fright(average[ix, iy], left[ix, iy])
                # Set left flux for cell (ix+1, iy) for this mu, eta
                if ix+1<nx and iy+1<ny:
                    left[ix+1, iy]
                scalar[ix, iy] = average[ix, iy] * wt
            
            for ix in range(nx):
                # Set top flux for each (ix, iy)
                top[ix, iy] = ftop(average[ix, iy], bottom[ix, iy])


def favg(mu, eta, ix, iy):
    s = source[ix, iy]
    den = totxs + 2 * mu / dx + 2 * eta / dy
    num = 2 * mu / dx * left[ix, iy] + 2 * eta / dy * bottom[ix, iy] + s
    return num / den

def ftop(favg, fbot):
    return 2 * favg - fbot

def fright(favg, fleft):
    return 2 * favg - fleft
    