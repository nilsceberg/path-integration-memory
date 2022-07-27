from .. import rate
from .. import basic
from ...network import IdentityLayer, Network, RecurrentForwardNetwork, Layer, InputLayer, FunctionLayer
from ..constants import *

import numpy as np
from scipy.special import expit


#def cpu4_output(noise):
#    def f(inputs):
#        tb1, tn1, tn2 = inputs
#        np.dot(W_TN)
#    return f


class PlasticWeightLayer(Layer):
    def __init__(self, noise: float, gain: float, fade: float = 0.125, sigmoid=False, initial_weights = np.ones(N_CPU4) * 0.5):
        self.gain = gain
        self.fade = fade
        self.noise = noise
        self.sigmoid = sigmoid

        self.weights = initial_weights
        self._output = np.zeros(N_CPU4)
        super().__init__(initial = self._output)

    def step(self, network: Network, dt: float):
        cpu4 = network.output("CPU4")
        self._output = cpu4 * self.weights #+ np.random.normal(0, self.noise)
        dwdt = (cpu4 - self.fade) * self.gain
        self.weights += dwdt * dt
        self.weights = np.clip(self.weights, 0, 1)
        #print(self.weights)

    def internal(self):
        return self.weights.tolist()

    def output(self, network: Network):
        #return self._output
        if self.sigmoid:
            return rate.noisy_sigmoid(self._output, cpu4_slope_tuned, cpu4_bias_tuned, self.noise)
        else:
            return self._output

# To get the shifted readout to work, we removed the sigmoid from the pontines
def pontine_output(noise):
    def f(inputs):
        cpu4, = inputs
        inputs = np.dot(rate.W_CPU4_pontine, cpu4)
        return inputs
        #return rate.noisy_sigmoid(inputs, pontine_slope_tuned, pontine_bias_tuned, noise)
    return f

def cpu1a_pontine_output(noise, slope, bias):
    def f(inputs):
        """The memory and direction used together to get population code for
        heading."""
        reference, memory, pontine = inputs

        inputs = 0.5 * np.dot(rate.W_CPU4_CPU1a, memory)
        inputs -= 0.5 * np.dot(rate.W_pontine_CPU1a, pontine)

        inputs -= reference

        #return np.clip(inputs, 0, 1000)
        return rate.noisy_sigmoid(inputs, slope, bias, noise)
    return f

def cpu1b_pontine_output(noise, slope, bias):
    def f(inputs):
        """The memory and direction used together to get population code for
        heading."""
        reference, memory, pontine = inputs

        inputs = 0.5 * np.dot(rate.W_CPU4_CPU1b, memory)
        inputs -= 0.5 * np.dot(rate.W_pontine_CPU1b, pontine)

        inputs -= reference

        #return np.clip(inputs, 0, 1000)
        return rate.noisy_sigmoid(inputs, slope, bias, noise)
    return f

def motor_output(noise):
    """outputs a scalar where sign determines left or right turn."""
    def f(inputs):
        cpu1a, cpu1b = inputs
        motor = np.dot(rate.W_CPU1a_motor, cpu1a)
        motor += np.dot(rate.W_CPU1b_motor, cpu1b)
        output = (motor[0] - motor[1])
        return output
        return (2*expit(100*output)-1)
    return f
