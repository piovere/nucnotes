import numpy as np


class Neuron():
    def __init__(self, ninputs):
        self.weights = 0.02 * np.random.rand(ninputs+1) - 0.01
        self.delta = None
        self.error = None
    
    def predict(self, x):
        p = self.weights[-1]
        for i in range(len(x)):
            p += x[i] * self.weights[i]
        self.out = self.sigmoid(p)
        return self.out
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class Layer():
    def __init__(self, ninputs, noutputs):
        self.neurons = [Neuron(ninputs) for i in range(noutputs)]
    
    def softmax(self, probs):
        shiftx = probs - np.max(probs)
        eprobs = np.exp(shiftx)
        return eprobs / np.sum(eprobs)

    def predict(self, x):
        res = [n.predict(x) for n in self.neurons]
        return res


class NN():
    def __init__(self, ninputs, nhidden, hiddenwidth, noutputs):
        self.layers = [Layer(ninputs, hiddenwidth)]
        for i in range(nhidden):
            self.layers.append(Layer(hiddenwidth, hiddenwidth))
        self.layers.append(Layer(hiddenwidth, 1))
    
    def predict(self, x):
        tmp = x
        lnum = 0
        for l in self.layers:
            tmp = l.predict(tmp)
            lnum += 1
        return tmp

    def derivative(self, x):
        return x * (1 - x)

    def backprop(self, xt, yt):
        yp = self.predict(xt)
        for i in range(len(self.layers))[::-1]:
            # Going through layers backwards
            l = self.layers[i]
            errors = list()
            if i != len(self.layers)-1:
                # This is not the output layer
                for j in range(len(l.neurons)):
                    error = 0.0
                    for n in self.layers[i+1].neurons:
                        error += n.weights[j] * n.delta
                    errors.append(error)
            else:
                # This is the output layer
                for j in range(len(l.neurons)):
                    n = l.neurons[j]
                    errors.append(yt - yp)
            for j in range(len(l.neurons)):
                n = l.neurons[j]
                n.delta = errors[j] * self.derivative(n.out)

    def train(self, xt, yt, lr=0.1):
        self.backprop(xt, yt)
        for i in range(len(self.layers)):
            inpts = xt
            if i != 0:
                inpts = [n.out for n in self.layers[i - 1].neurons]
            for neuron in self.layers[i].neurons:
                for j in range(len(inpts)):
                    neuron.weights[j] += lr * neuron.delta * inpts[j]
                neuron.weights[-1] += lr * neuron.delta * 1.0
