
# coding: utf-8

# # Problem 3

# In[1]:


from scipy.integrate import trapz
from scipy.special import legendre
from scipy.optimize.zeros import newton
from numpy.polynomial.legendre import leggauss
import numpy as np
import matplotlib.pyplot as plt


# Plot the 14th Legendre polynomial to eyeball 
# the starting guesses for the zeros

# In[2]:


x = np.linspace(-1, 1, 10000)


# In[3]:


l = legendre(14)


# In[4]:


plt.plot(x, l(x))
plt.xlim(0, 1)
plt.axhline(y=0, alpha=0.3, color='black')


# This has seven positive and seven negative zeros

# In[5]:


guesses = [
    0.1,
    0.33,
    0.52,
    0.7,
    0.8,
    0.9,
    1.0
]


# `newton` uses the Newton-Raphson method to find zeros of a function

# In[6]:


zeros = np.array([newton(l, g) for g in guesses])
print(zeros)


# Now I can construct the matrix of integrals for $x^n$

# Calculate the numerical integral of $x^n$ for even $n$'s

# In[7]:


integrals = np.array([0.5*trapz(x**n, x) for n in range(15)[::2]])
print(integrals)


# Create a matrix where each column $j$ and row $i$ is $\mu_j^{2i}$

# In[8]:


functions = np.array([zeros**n for n in range(15)[::2]])


# In[9]:


np.set_printoptions(precision=1)
print(functions)


# In[10]:


np.set_printoptions(precision=8)


# Get the official values to compare with

# In[11]:


mus, wts = leggauss(14)


# Compare the official $\mu$ values (stored in the variable `mus`) to my 
# calculated values (stored in `zeros`)

# In[12]:


mus[7:]


# In[13]:


zeros


# Compare the official weights (stored in `wts`) to my calculated values 
# (stored in `weights`)

# In[14]:


wts[7:]


# In[15]:


weights = np.linalg.inv(functions.T @ functions) @ functions.T @ integrals
weights


# Calculate the fractional error between my calculated weights and the 
# official ones

# In[16]:


np.abs(weights - wts[7:]) / wts[7:]


# Pretty close! Within ~$10^{-4}$%
