import numpy as np
from nn2 import Layer, NN


x = np.random.rand(20).reshape(-1, 4)
y = np.round(np.random.rand(5))

l = Layer(4, 1)

print(l.transfer(x))

net = NN(x.shape[1], 1, 5)

print(net.predict(x[0])[0])