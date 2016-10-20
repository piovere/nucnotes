import numpy as np
import threading
import csv


def solution(N):
    # define our interval length
    deltaX = 4.0 / (N-1)
    deltaXsquared = deltaX ** 2
    
    # define our matrix
    diag = -2.0 / deltaXsquared - 1.0
    offdiag = 1.0 / deltaXsquared
    
    # instantiate matrix and fill diagonal values
    A = np.eye(N-2) * diag
    A = A + offdiag * np.diagflat(np.resize(np.array([1.0]), (N-3)), 1)
    A = A + offdiag * np.diagflat(np.resize(np.array([1.0]), (N-3)), -1)
    
    # define our output vector, b
    b = np.zeros(N-2)
    b[0] = -2.0 / deltaXsquared
    b[N-3] = -54.61647 / deltaXsquared
    
    # compute solution
    f = np.linalg.solve(A, b)
    
    return f


def analytic(N):
    """ The analytic solution to the equation """
    x = np.linspace(0, 4, num=N)
    x = x[1:-1]
    return np.exp(x) + np.exp(-x)


errors = []

def error(n):
    errors.append((n, np.linalg.norm(np.abs(analytic(n)-solution(n)))))

threads = []

e_min = 10**-3
e=1.0

n_max = 10**3
n = 5

# while e>e_min and n<n_max:
for i in range(n, n_max):
    # n += 1
    t = threading.Thread(target=error, args=(i, ))
    threads.append(t)
    t.start()

np.savetxt('errors.csv', errors, delimiter=',', newline='\n')
