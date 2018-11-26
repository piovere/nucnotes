import numpy as np
import numpy.linalg as la
from random import random


def make_network(inputs, outputs, hidden_layers, neurons_per_layer, 
                 max_error=None, max_epochs=None, max_improvement=None):
    # Initialize weight matrix
    raise NotImplementedError


def print_error(network, x_train, y_train):
    raise NotImplementedError


def predict(network, x_predict):
    raise NotImplementedError


def confusion():
    raise NotImplementedError


class Layer():
    def __init__(self, inputs, outputs=None, activation='sigmoid'):
        if outputs is None:
            # This is a hidden layer
            outputs = inputs
        self.neurons = [Neuron(inputs) for _ in range(outputs)]
        # Need to impliment the different activation functions
        self.activation = activation
    
    def activate(self, x):
        # Return the activations from the layer
        answers = []
        for neuron in self.neurons:
            answers.append(neuron.activate(x))
        return np.array(answers)
    
    def fit(self, x, y):
        raise NotImplementedError

    def backprop(self, yp, yt):
        # Returns errors from previous layer
        error = yt - yp
        layer_delta = error * self.derivative(error)
        return layer_delta
    
    def derivative(self, x):
        return x * (1 - x)

    def __repr__(self):
        return f"Layer with {len(self.neurons)} neurons"


class Neuron():
    def __init__(self, num_inputs):
        # Create a weight for each input + bias
        self.weights_ = np.random.rand(num_inputs)
    
    def transfer(self, x):
        return 1 / (1 + np.exp(-x))
    
    def activate(self, xs):
        i = xs @ self.weights_
        return self.transfer(i)
    
    def transfer_derivative(self, output):
        return output * (1 - output)


class NNClassifier():
    def __init__(self, num_layers, neurons_per_layer):
        self.input = None
        self.layers = [Layer(neurons_per_layer) for _ in range(num_layers)]
        self.output = Layer(inputs=neurons_per_layer, outputs=1)
        self.neurons_per_layer = neurons_per_layer
    
    def predict(self, x):
        # Predict input layer
        tmp = self.input.activate(x)
        # Predict hidden layers
        for hl in self.layers:
            tmp = hl.activate(tmp)
        # Predict output layer
        tmp = self.output.activate(tmp)
        return tmp
    
    def fit(self, x, y):
        self.input = Layer(x.shape[1], self.neurons_per_layer)
        return self
