import numpy as np


num_outputs = 1
num_hidden = 10

weights = (np.random.rand(num_outputs * num_hidden) - 0.5) / 50
weights = weights.reshape(num_outputs, num_hidden)

while True:
    # Shuffle xs and yy
    # For each x and y
    for x, y in zip(x, y):
        # For each output class
        o_i = 0
        # For each hidden neuron