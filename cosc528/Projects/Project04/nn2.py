import numpy as np


class Layer():
    def __init__(self, inputs, outputs, activation='sigmoid'):
        self.inputs = inputs
        self.outputs = outputs
        self.weights = np.random.random((inputs, outputs))
        self.activation = activation
        self.out = None
        self.prop = None
        self.error = None
        self.delta = None
    
    def transfer(self, x):
        self.prop = x @ self.weights
        return self.prop
    
    def sigmoid(self, x):
        return 1 / (1 - np.exp(-x))

    def self.predict(self, x):
        self.out = self.sigmoid(self.prop(x))
        return self.out

    def derivative(self, x):
        return x * (1 - x)
    
    def backprop(self, delta):
        self.error = delta @ self.weights.T


class OutputLayer(Layer):
    def __init__(self, hidden_width):
        super().__init__(hidden_width, 1)

    def backprop(self, yp):
        error = yp - self.out
        self.error = error


class NN():
    def __init__(self, inputs, num_hidden, hidden_width, outputs=1):
        self.num_inputs = inputs
        self.input = Layer(inputs, hidden_width)
        self.hiddens = [Layer(hidden_width, hidden_width) for _ \
                        in range(num_hidden)]
        self.output = OutputLayer(hidden_width)
        self.layers = self.input
        self.layers.extend(self.hiddens)
        self.layers.append(self.output)
    
    def predict(self, x):
        tmp = self.input.transfer(x)
        for h in self.hiddens:
            tmp = h.transfer(tmp)
        tmp = self.output.transfer(tmp)
        return tmp
    
    def backprop(self, x, y, lr=0.5):
        yp = self.predict(x)

        self.layers[-1].error = 
        
        for i in range(1,len(self.layers)-1):


