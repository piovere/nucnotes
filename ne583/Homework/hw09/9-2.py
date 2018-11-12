import numpy as np
import numpy.linalg as la
from numpy.polynomial.legendre import leggauss


nx = 100
nang = 2
width = 50
nmfp = 5
mfp = width / nmfp
sigma = 1 / mfp

# Get points for Gauss-Legendre quadrature
mus, wts = leggauss(nang)

# Define our auxiliary equations
def step(psi_in, mu, s=0, nx=nx, width=width, sigma=sigma):
    dx = width / nx

    print(f"{width}\t{nx}\t{dx}")

    mux = mu / dx
    
    num = s + mux * psi_in
    den = mux + sigma
    
    psi_out = num / den
    
    psi_avg = psi_out
    
    return psi_avg, psi_out

def diamond_difference(psi_in, mu, s=0, nx=nx, width=width, sigma=sigma):
    dx = width / nx
    mux = mu / dx
    hsig = sigma / 2
    
    psi_out = s + (mux - hsig) * psi_in
    psi_out /= mux + sigma / 2
    
    psi_avg = (psi_in + psi_out) / 2
    
    return psi_avg, psi_out

def weighted_diamond_difference(psi_in, mu, s=0, nx=nx, width=width, 
                                sigma=sigma, alpha=0.8):
    dx = width / nx
    mux = mu / dx
    
    psi_out = s + (mux - (1 - alpha) * sigma) * psi_in
    psi_out /= (mux) + (alpha * sigma)
    
    psi_avg = (1 - alpha) * psi_in + alpha * psi_out
    
    return psi_avg, psi_out

def convergence(old_scalar, new_scalar):
    if np.all(new_scalar == 0):
        return 1.0
    else:
        tmp = np.abs(new_scalar - old_scalar) / new_scalar
        return np.max(tmp)

# Set intial fluxes for all angles and positions to zero
left = np.zeros((nang, nx))
right = np.zeros((nang, nx))
avg = np.zeros((nang, nx))

# Make placeholders for old and new scalar flux
scalar = np.zeros(nx)
old_scalar = np.ones(nx)

# Incoming current for all angles is 1.0
left[:, 0] = np.ones_like(left[:, 0])

# Decide which auxiliary equation to use
aux_function = step

while convergence(old_scalar, scalar) > 0.000001:
    # Make a copy of the current scalar for convergence
    old_scalar = np.copy(scalar)

    # Loop over all angles
    for ia in range(mus.shape[0]):
        mu = mus[ia]
        wt = wts[ia]

        # Set indices by direction/sign of mu
        ixs = [i for i in range(nx)]
        if mu < 0:
            # Reverse the ixs array
            ixs = ixs[::-1]
            phi0 = right[ia, ixs[0]]
        else:
            phi0 = left[ia, ixs[0]]

        for ix in ixs:
            if mu > 0:
                fluxavg, phi1 = aux_function(phi0, mu)

                avg[ia, ix] = fluxavg
                right[ia, ix] = phi1

print("I did a thing")
