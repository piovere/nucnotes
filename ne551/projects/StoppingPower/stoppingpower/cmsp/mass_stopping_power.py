import numpy as np
import scipy.constants as const
from pyne import nucname
from pyne import data


def I(material):
    """ Looks up the Adjusted Average Excitation Energy if Z<=13. Otherwise calculates it
        Output units are in MeV
    """
    # Lookup table for Z<=13. Key is the "Z" of the material. Values are in eV
    lookup_table = {
        1: 18.9,
        2: 42.0,
        3: 38.0,
        4: 60.0, 
        6: 78.0,
        7: 85.0,
        8: 89.0,
        10: 131.0,
        13: 163.0
    }
    
    I_list = []
    
    for mat, frac in material.mult_by_mass().items():
        Z = nucname.znum(mat)
        # Check to see if Z is in our table
        I = lookup_table.get(Z)

        # If I is not in the table, calculate it
        # Use Anderson Equation 2.33
        if I is None:
            I = 9.73 * Z + 58.8 * Z ** -0.19
            
        I_list.append(I * frac)
    
    I_a = sum(I_list)
    
    # Convert I from eV to MeV
    I_a = I_a * 10**-6.0
    
    return I_a


def beta_squared(T, m):
    """ Gives value of beta^2 for a given Mass (MeV/c^2) and Kinetic Energy (MeV)
    """
    numerator = T * (T + 2 * m)
    denominator = (T + m) ** 2
    return numerator / denominator


def gamma_squared(T, m):
    """ Gives value of gamma^2 for a given mass (MeV/c^2) and Kinetic Energy (MeV)
    """
    return 1.0 / (1 - beta_squared(T, m)) ** 0.5


def Z_eff(material):
    """ Returns the effective Z/M == Z/A of a material
    """
    return sum(
        [
            1.0 * nucname.znum(m) / data.atomic_mass(m) * f \
            for m, f in material.mult_by_mass().items()
        ]
    )


def S_c(incident, target, T, M_b):
    """ Returns the stopping power in MeV/cm
        T in MeV
        density in g/cm^3
        Output in MeV/cm

        ToDo: Harden against T <= 0
    """
    # Them constants though
    pi = np.pi
    r_e = const.value('classical electron radius') * 100.0 # Convert 1m = 100cm
    m_e = const.value('electron mass energy equivalent in MeV')
    N_A = const.value('Avogadro constant')

    # Currently the incident and target are specified in Z number. incident is assumed to be fully ionized
    z = incident
    Z = target
    Z_target = Z_eff(target)
    I_target = I(target)

    # M_b is specified in AMU
    M_b = M_b * const.value('atomic mass constant energy equivalent in MeV')

    def T_above_zero(T):
        first = 4 * (z ** 2) * pi * (r_e ** 2) * m_e
        second = N_A * Z_target # TODO: Take M_m from a Pyne material
        third = 1.0 / beta_squared(T, M_b)
        logpart = (2 * m_e * beta_squared(T, M_b) * gamma_squared(T, M_b)) / (I_target)
        fourth = np.log(logpart) - beta_squared(T, M_b) + beta_squared(T, M_b)
    
        return first * second * third * fourth

    return np.piecewise(T, [T<=0.0, T>0], [0.0, T_above_zero])
