import math
from random import seed, random

class Nuclide(object):
    def __init__(self, name, daughter, halflife, activity=None, amount=None):
        self._halflife = halflife
        self._daughter = daughter
        self._name = name

        if activity is None:
            self._amount = amount
            self._activity = amount * math.log(2) / self._halflife
        elif amount is None:
            self._activity = activity
            self.amount = activity * self._halflife / math.log(2)

    
    def repr(self):
        return self._name
    

    def set_activity(self, activity):
        self.set_lambda()


    def set_lambda(self):
        self._lambda = math.log(2) / self._halflife


    def decay(self, time):
        self.set_lambda()
        frac = math.exp(self._lambda) * time
        daughter = (1 - frac) * self._amount
        self._amount = frac * self._amount
        return daughter


sample = {
    94241: Nuclide(94241, 95241, 4.509581e8, activity=3.714e12)
}

def timestep(sample, time):
    for nuclide in sample.items():
        daughter = nuclide._daughter
        daughter_amount = sample.get(daughter, Nuclide(daughter, ))