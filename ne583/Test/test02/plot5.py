import numpy as np
import matplotlib.pyplot as plt


f = 'no5res.txt'
data = np.loadtxt(f)

width = 6.0
x = np.linspace(0, width, data.shape[0])

plt.plot(x, data)
plt.title("Top leakage")
plt.savefig('no5result.png', dpi=1000)
plt.clf()
