import numpy as np
from numpy.polynomial.legendre import leggauss
import matplotlib.pyplot as plt
from itertools import product
import tabulate


NX = 100.0
nang = 14
width = 5.0
nmfp = 5.0
mfp = width / nmfp
sigma = 1 / mfp

# Define our auxiliary equations
def step(psi_in, mu, s=0, nx=NX, width=width, sigma=sigma):
    dx = width / nx

    mux = mu / dx

    num = s + mux * psi_in
    den = mux + sigma

    psi_out = num / den

    psi_avg = psi_out

    return psi_avg, psi_out


def diamond_difference(psi_in, mu, s=0, nx=NX, width=width, sigma=sigma):
    dx = width / nx
    mux = mu / dx
    hsig = sigma / 2

    psi_out = s + (mux - hsig) * psi_in
    psi_out /= mux + sigma / 2

    psi_avg = (psi_in + psi_out) / 2

    return psi_avg, psi_out


def weighted_diamond_difference(psi_in, mu, s=0, nx=NX, width=width,
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


def calc_right_leakage(nang, aux_function, nx=NX, make_plot=True,\
                       verbose=False):
    # Make placeholders for old and new scalar flux
    scalar = np.zeros(nx)
    old_scalar = np.ones(nx)

    # Make placeholder for left and right leakage
    left_leakage = 0
    right_leakage = 0

    # Keep track of how many loops we have done
    loop_counter = 0
    max_loops = 10000

    # Get points for Gauss-Legendre quadrature
    mus, wts = leggauss(nang)
    mus = -1 * mus  # It loads the negative ones first
    wts /= 2

    while convergence(old_scalar, scalar) > 0.000001 and \
            loop_counter < max_loops:
        loop_counter += 1
        if loop_counter % 1000 == 0:
            print(f"After {loop_counter} iterations, "
                  f"right leakage is {right_leakage:0.5f}")

        # Make a copy of the old scalar flux
        old_scalar = np.copy(scalar)
        scalar = np.zeros(nx)

        left_leakage = 1.0 / (wts.T @ np.abs(mus)) / 2.0
        right_leakage = 0

        # Loop over all the angles
        for mu, wt in zip(mus, wts):
            left_leakage = 1.0 / (wts.T @ np.abs(mus)) / 2.0
            if mu > 0:
                psi_in = left_leakage

                # Loop over all the positions
                for ix in range(nx):
                    psi_avg, psi_out = aux_function(psi_in, mu)

                    # Add to the scalar flux
                    scalar[ix] += wt * psi_avg

                    psi_in = psi_out

                # Add to outgoing leakage
                right_leakage += mu * wt * psi_in

            else:  # mu is negative
                psi_in = right_leakage

                # Reverse the index
                ixs = [ix for ix in range(nx)][::-1]

                # Loop over all the positions
                for ix in ixs:
                    psi_avg, psi_out = aux_function(psi_in, mu)

                    scalar[ix] += wt * psi_avg

                    psi_in = psi_out

                # mu is negative so adding a negative...
                left_leakage += mu * wt * psi_in

    if verbose:
        print(f"Converged after {loop_counter} iterations")
        print(f"Right leakage should be {0.0017556017855412775}")
        print(f"Right leakage is {right_leakage}")
        print(f"Left leakage is {left_leakage}")

    if make_plot:
        plt.plot(np.linspace(0, width, nx), scalar)
        plt.xlabel('x')
        plt.ylabel('Scalar flux')
        plt.yscale('log')
        plt.title(f'$S_{{{nang}}}$, {aux_function.__name__}')
        plt.show()

    return right_leakage


if __name__ == "__main__":
    angles_list = [4, 8, 12]
    functions_list = [step, diamond_difference, weighted_diamond_difference]

    res = {
        'Quadrature': [],
        'Aux Function': [],
        'Right Leakage': [],
        'Analytic': [],
        'Error %': []
    }

    for nangles, fxn in product(angles_list, functions_list):
        rl = calc_right_leakage(nangles, fxn, make_plot=False, nx=100, verbose=False)
        res['Quadrature'].append(nangles)
        res['Aux Function'].append(fxn.__name__)
        res['Right Leakage'].append(rl)
        analytic = 0.0017556017855412775
        res['Analytic'].append(analytic)
        res['Error %'].append(100 * abs(rl - analytic) / analytic)

    print(tabulate.tabulate(res, headers='keys'))
