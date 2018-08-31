from tabulate import tabulate


M_235 = 235.043930131
rho = 10.5
A = 6.022*10**23
wt = [0.01, 0.02, 0.03, 0.04, 0.05]
M_238 = 238.02891
M_O = 15.9994
sigma_f_235 = 577.0 * 10 ** -24
sigma_a_238 = 2.73 * 10 ** -24
sigma_a_U = 7.6 * 10 ** -24


def M_U(w):
    return M_235 * M_238 / (w * M_238 + (1 - w) * M_235)


def N_235(w):
    temp = rho
    temp *= w * M_U(w) * A
    temp /= M_235
    temp /= M_U(w) + 2 * M_O
    return temp


def N_238(w):
    temp = rho
    temp *= (1 - w) * M_U(w) * A
    temp /= M_238
    temp /= M_U(w) + 2 * M_O
    return temp


def Sigma_f(w):
    return sigma_f_235 * N_235(w)


def Sigma_a(w):
    return sigma_a_238 * N_238(w)


def Sigma_a_UO2(w):
    return rho * A / M_U(w) * sigma_a_U


print tabulate(zip(wt, map(N_235, wt), map(Sigma_f, wt)), headers=['w', 'N', 'Sigma'])


lastpart = [Sigma_a(w) / Sigma_a_UO2(w) for w in wt]

print tabulate(zip(wt, lastpart), headers=['w', 'Sigma'])
