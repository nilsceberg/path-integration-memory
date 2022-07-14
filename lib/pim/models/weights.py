from . import rate
from . import basic
from ..network import IdentityLayer, Network, RecurrentForwardNetwork, Layer, InputLayer, FunctionLayer
from .constants import *

import numpy as np


#def cpu4_output(noise):
#    def f(inputs):
#        tb1, tn1, tn2 = inputs
#        np.dot(W_TN)
#    return f


class PlasticWeightLayer(Layer):
    def __init__(self, noise: float, gain: float, fade: float = 0.125, sigmoid=False,initial_weights = np.ones(N_CPU4) * 0.5):
        self.gain = gain
        self.fade = fade
        self.noise = noise
        self.sigmoid = sigmoid

        self.weights = initial_weights
        self._output = np.zeros(N_CPU4)
        super().__init__(initial = self._output)

    def step(self, network: Network, dt: float):
        cpu4 = network.output("CPU4")
        self._output = cpu4 * self.weights + np.random.normal(0, self.noise)
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

# To get the shifted readout to work, we removed the sigmoids from the pontine and cpu1 outputs.
def pontine_output(noise):
    def f(inputs):
        cpu4, = inputs
        inputs = np.dot(rate.W_CPU4_pontine, cpu4)
        return inputs
        #return noisy_sigmoid(inputs, pontine_slope_tuned, pontine_bias_tuned, noise)
    return f

def cpu1a_pontine_output(inputs, noise):
    """The memory and direction used together to get population code for
    heading."""
    tb1, cpu4, pontine = inputs

    inputs = 0.5 * np.dot(rate.W_CPU4_CPU1a, cpu4)

    inputs -= 0.5 * np.dot(rate.W_pontine_CPU1a, pontine)
    inputs -= np.dot(rate.W_TB1_CPU1a, tb1)

    return np.clip(inputs, 0, 1000)
    #return noisy_sigmoid(inputs, cpu1_slope_tuned, cpu1_bias_tuned, noise)

def cpu1b_pontine_output(inputs, noise):
    """The memory and direction used together to get population code for
    heading."""
    tb1, cpu4, pontine = inputs

    inputs = 0.5 * np.dot(rate.W_CPU4_CPU1b, cpu4)

    inputs -= 0.5 * np.dot(rate.W_pontine_CPU1b, pontine)
    inputs -= np.dot(rate.W_TB1_CPU1b, tb1)

    return np.clip(inputs, 0, 1000)
    #return noisy_sigmoid(inputs, cpu1_slope_tuned, cpu1_bias_tuned, noise)

def cpu1_pontine_output(noise):
    def f(inputs):
        tb1, cpu4, pontine = inputs
        cpu1a = cpu1a_pontine_output([tb1, cpu4, pontine], noise)
        cpu1b = cpu1b_pontine_output([tb1, cpu4, pontine], noise)
        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])
    return f

def motor_output_theoretical(noise):
    def f(inputs):
        cpu4, tb1, pontine = inputs
        #tb1 = 0.5*tb1

        left_pontine = np.roll(cpu4[:8], -3)
        right_pontine = np.roll(cpu4[:8], 3)

        left = 0.5 * (np.roll(cpu4[:8], -1) - right_pontine) # type: ignore
        right = 0.5 * (np.roll(cpu4[8:], 1) - left_pontine) # type: ignore

        motor = np.sum(np.clip(right - tb1, 0, 1000)) - np.sum(np.clip(left - tb1, 0, 1000))
        return -motor + np.random.normal(0, noise)
    return f

def motor_output(noise):
    """outputs a scalar where sign determines left or right turn."""
    def f(inputs):
        cpu1, = inputs
        cpu1a = cpu1[1:-1]
        cpu1b = np.array([cpu1[-1], cpu1[0]])
        motor = np.dot(rate.W_CPU1a_motor, cpu1a)
        motor += np.dot(rate.W_CPU1b_motor, cpu1b)
        output = (motor[0] - motor[1])
        return -output
    return f


def build_phase_shift_network(params) -> Network:
    # TODO: allow noisy weights
    noise = params.get("noise", 0.1)
    mem_gain = params.get("mem_gain", 0.0025)
    mem_fade = params.get("mem_fade", 0.1)
    pfn_weight_factor = params.get("pfn_weight_factor", 1)

    return RecurrentForwardNetwork({
        "flow": InputLayer(initial = np.zeros(2)),
        "heading": InputLayer(),
        "TL2": FunctionLayer(
            inputs = ["heading"],
            function = rate.tl2_output(noise),
            initial = np.zeros(N_TL2),
        ),
        "CL1": FunctionLayer(
            inputs = ["TL2"],
            function = rate.cl1_output(noise),
            initial = np.zeros(N_CL1),
        ),
        #"CL1": IdentityLayer("heading"),
        "TB1": FunctionLayer(
            inputs = ["CL1", "TB1"],
            #function = basic.tb1_output, #(noise),
            function = rate.tb1_output(noise), #(noise),
            initial = np.zeros(N_TB1),
        ),
        "TN1": FunctionLayer(
            inputs = ["flow"],
            function = rate.tn1_output(noise),
            initial = np.zeros(N_TN1),
        ),
        "TN2": FunctionLayer(
            inputs = ["flow"],
            function = rate.tn2_output(noise),
            initial = np.zeros(N_TN2),
        ),
        "CPU4.old": rate.CPU4PontineLayer(
            "TB1", "TN1", "TN2",
            rate.W_TN_CPU4,
            rate.W_TB1_CPU4,
            gain = 1.0,
            slope = cpu4_slope_tuned,
            bias = cpu4_bias_tuned,
            noise = noise,
        ),
        "CPU4": rate.MemorylessCPU4Layer(
            "TB1", "TN1", "TN2",
            rate.W_TN_CPU4,
            rate.W_TB1_CPU4,
            gain = pfn_weight_factor,
            slope = cpu4_slope_tuned,
            bias = cpu4_bias_tuned,
            noise = noise,
            background_activity = 0.0,
        ),
        "memory": PlasticWeightLayer(noise, mem_gain, mem_fade),
        "Pontine": FunctionLayer(
            inputs = ["memory"],
            function = pontine_output(noise),
            initial = np.zeros(N_Pontine)
        ),
        "theory": FunctionLayer(
            inputs = ["memory", "TB1", "Pontine"],
            function = motor_output_theoretical(noise)
        ),
        "CPU1": FunctionLayer(
            inputs = ["TB1", "memory", "Pontine"],
            function = cpu1_pontine_output(noise),
            initial = np.zeros(N_CPU1),
        ),
        "motor": FunctionLayer(
            inputs = ["CPU1"],
            function = motor_output(noise),
        )
    })


def build_inverting_network(params) -> Network:
    # TODO: allow noisy weights
    noise = params.get("noise", 0.1)
    background_activity = params.get("background_activity", 0.0)
    mem_gain = params.get("mem_gain", 0.0025)
    mem_fade = params.get("mem_fade", 0.125)

    return RecurrentForwardNetwork({
        "flow": InputLayer(initial = np.zeros(2)),
        "heading": InputLayer(),
        "TL2": FunctionLayer(
            inputs = ["heading"],
            function = rate.tl2_output(noise),
            initial = np.zeros(N_TL2),
        ),
        "CL1": FunctionLayer(
            inputs = ["TL2"],
            function = rate.cl1_output(noise),
            initial = np.zeros(N_CL1),
        ),
        "TB1": FunctionLayer(
            inputs = ["CL1", "TB1"],
            function = rate.tb1_output(noise), #(noise),
            initial = np.zeros(N_TB1),
        ),
        "TN1": FunctionLayer(
            inputs = ["flow"],
            function = rate.tn1_output(noise),
            initial = np.zeros(N_TN1),
        ),
        "TN2": FunctionLayer(
            inputs = ["flow"],
            function = rate.tn2_output(noise),
            initial = np.zeros(N_TN2),
        ),
        "CPU4": rate.MemorylessCPU4Layer(
            "TB1", "TN1", "TN2",
            rate.W_TN_CPU4,
            rate.W_TB1_CPU4,
            gain = 1.0,
            slope = cpu4_slope_tuned,
            bias = cpu4_bias_tuned,
            noise = noise,
            background_activity = background_activity, # since we will lose information about the weights if this is 0
        ),
        "weighted": PlasticWeightLayer(noise, gain=mem_gain, fade=mem_fade),
        "memory": FunctionLayer(
            inputs = ["CPU4", "weighted"],
            function = lambda inputs: rate.noisy_sigmoid(inputs[1] / inputs[0], cpu4_slope_tuned, cpu4_bias_tuned, noise),
            initial = np.zeros(N_CPU4),
        ),
        "Pontine": FunctionLayer(
            inputs = ["memory"],
            function = rate.pontine_output(noise),
            initial = np.zeros(N_Pontine)
        ),
        "CPU1": FunctionLayer(
            inputs = ["TB1", "memory", "Pontine"],
            function = rate.cpu1_pontine_output(noise),
            initial = np.zeros(N_CPU1),
        ),
        "motor": FunctionLayer(
            inputs = ["CPU1"],
            function = motor_output(noise),
        )
    })

