import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt

NX = 1000
NY = 1000
TOTXS = 1.0
WIDTH = 6 / TOTXS

DX = WIDTH / NX
DY = WIDTH / NY

def step(phi0, mu, s=0, nx=NX, totxs=TOTXS, width=WIDTH):
    dx = width / nx
    phi1 = (s + mu * phi0 / dx) / (mu / dx + totxs)
    fluxave = phi1
    return phi1, fluxave


def diamond_difference(phi0, mu, s=0, nx=NX, totxs=TOTXS, width=WIDTH):
    dx = width / NX
    mux = mu / dx
    htotxs = totxs / 2
    
    phi1 = phi0 * (mux - htotxs) / (htotxs + mux)
    
    fluxave = (phi1 + phi0) / 2
    
    return phi1, fluxave


def conv(scalar, scalarOld):
    if np.all(scalar == 0):
        return 1
    else:
        return np.max(np.abs(scalar - scalarOld) / scalar)


INNER = 0
EPS = 0.000001
C = 10000

mu, wt = leggauss(NANG)

WTTOT = np.abs(mu) @ wt
INFLUX = 2.0 / WTTOT

PSI_AVG = np.zeros((NANG, NX))
PSI_LEFT = np.zeros((NX, NY))

SCALAR = np.zeros(NX)

INNER = 0
while (C > EPS) and (INNER < 10000):
    INNER += 1
    SCALAROLD = np.copy(SCALAR)
    SCALAR = np.zeros(NX)
    
    LEFT = 0
    RIGHT = 0
    
    for ia in range(NANG):
        phi0 = 0.0
        muabs = abs(mu[ia])
        
        if mu[ia] > 0:
            phi0 = INFLUX
            
        for ix0 in range(NX):
            if mu[ia] < 0:
                ix = NX - 1 - ix0
            else:
                ix = ix0

            phi1, fluxave = step(phi0, mu[ia])
            phi0 = phi1

            SCALAR[ix] += wt[ia] * fluxave

        if mu[ia] > 0:
            RIGHT += wt[ia] * phi0 * mu[ia]
        
    C = conv(SCALAR, SCALAROLD)

print(RIGHT)
