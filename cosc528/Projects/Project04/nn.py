import numpy as np


def make_network(inputs, outputs, hidden_layers, neurons_per_layer, 
                 max_error=None, max_epochs=None, max_improvement=None):
    raise NotImplementedError


def print_error(network, x_train, y_train):
    raise NotImplementedError


def predict(network, x_predict):
    raise NotImplementedError
