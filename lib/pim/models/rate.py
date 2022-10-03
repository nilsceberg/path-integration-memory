import numpy as np
from pim.network import InputLayer
from scipy.special import expit

from ..network import Network, RecurrentForwardNetwork, FunctionLayer, IdentityLayer, WeightedSynapse, Layer, Output
from .constants import *


def gen_tb_tb_weights(weight=1.):
    """Weight matrix to map inhibitory connections from TB1 to other neurons"""
    W = np.zeros([N_TB1, N_TB1])
    sinusoid = -(np.cos(np.linspace(0, 2*np.pi, N_TB1, endpoint=False)) - 1)/2
    for i in range(N_TB1):
        values = np.roll(sinusoid, i)
        W[i, :] = values
    return weight * W


def noisy_sigmoid(v, slope=1.0, bias=0.5, noise=0.01):
    """Takes a vector v as input, puts through sigmoid and
    adds Gaussian noise. Results are clipped to return rate
    between 0 and 1"""
    sig = expit(v * slope - bias)
    if noise > 0:
        sig += np.random.normal(scale=noise, size=len(v))
    return np.clip(sig, 0, 1)


def noisify_weights(W, noise=0.01):
    """Takes a weight matrix and adds some noise on to non-zero values."""
    N = np.random.normal(scale=noise, size=W.shape)
    # Only noisify the connections (positive values in W). Not the zeros.
    N_nonzero = N * W
    return W + N_nonzero


class MemorylessCPU4Layer(Layer):
    def __init__(self, TB1, TN1, TN2, W_TN, W_TB1, gain, slope, bias, noise, background_activity, holonomic=False, disable_beta_on_outbound=False):
        self.TB1 = TB1
        self.TN1 = TN1
        self.TN2 = TN2

        self.W_TN = W_TN
        self.W_TB1 = W_TB1

        self.gain = gain
        self.slope = slope
        self.bias = bias

        self.noise = noise
        self.background_activity = background_activity

        self.holonomic = holonomic
        self.disable_beta_on_outbound = disable_beta_on_outbound

        super().__init__(initial = np.ones(N_CPU4) * self.background_activity)

    def output(self, network: Network) -> Output:
        tb1 = network.output(self.TB1)
        tn1 = network.output(self.TN1)
        tn2 = network.output(self.TN2)

        use_beta = (not self.disable_beta_on_outbound) or network.context["homing"]
        beta = self.background_activity if use_beta else 0.0

        # Not really holonomic!
        if self.holonomic:
            mem_update = -np.dot(self.W_TN, tn2) * (np.dot(self.W_TB1, tb1) - 0.5) + beta
            return np.clip(noisy_sigmoid(mem_update, self.slope, self.bias, self.noise) * self.gain, 0, 1)
        else:
            mem_update = np.dot(self.W_TN, tn2)
            mem_update -= np.dot(self.W_TB1, tb1)
            return np.clip(noisy_sigmoid(mem_update, self.slope, self.bias, self.noise) * self.gain + beta, 0, 1)

class CPU4Layer(Layer):
    def __init__(self, TB1, TN1, TN2, W_TN, W_TB1, gain, slope, bias, noise):
        self.TB1 = TB1
        self.TN1 = TN1
        self.TN2 = TN2

        self.W_TN = W_TN
        self.W_TB1 = W_TB1

        self.gain = gain
        self.slope = slope
        self.bias = bias

        self.noise = noise

        self.memory = np.ones(N_CPU4) * 0.5
        super().__init__(initial = self.memory)

    def step(self, network: Network, dt: float):
        """Memory neurons update.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        tb1 = network.output(self.TB1)
        tn1 = network.output(self.TN1)
        tn2 = network.output(self.TN2)

        self.memory += (np.clip(np.dot(self.W_TN, 0.5-tn1), 0, 1) *
                     self.gain * np.dot(self.W_TB1, 1.0-tb1)) * dt
        self.memory -= self.gain * 0.25 * np.dot(self.W_TN, tn2) * dt
        self.memory = np.clip(self.memory, 0.0, 1.0)

    def output(self, network: Network) -> Output:
        """The output from memory neuron, based on current calcium levels."""
        return noisy_sigmoid(self.memory, self.slope,
                             self.bias, self.noise)

class CPU4PontineLayer(CPU4Layer):
    def internal(self):
        return [self.memory, self.memory]

    def step(self, network: Network, dt: float):
        """Memory neurons update.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        tb1 = network.output(self.TB1)
        tn1 = network.output(self.TN1)
        tn2 = network.output(self.TN2)

        mem_update = np.dot(self.W_TN, tn2)
        mem_update -= np.dot(self.W_TB1, tb1)
        mem_update = np.clip(mem_update, 0, 1)
        mem_update *= self.gain
        self.memory += mem_update * dt
        self.memory -= 0.125 * self.gain * dt
        self.memory = np.clip(self.memory, 0.0, 1.0)

W_CL1_TB1 = np.tile(np.eye(N_TB1), 2)
W_TB1_TB1 = gen_tb_tb_weights()
W_TB1_CPU1a = np.tile(np.eye(N_TB1), (2, 1))[1:N_CPU1A+1, :]
W_TB1_CPU1b = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0]])
W_TB1_CPU4 = np.tile(np.eye(N_TB1), (2, 1))
W_TN_CPU4 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
]).T
W_CPU4_CPU1a = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
])
W_CPU4_CPU1b = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #9
])
W_CPU1a_motor = np.array([
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
W_CPU1b_motor = np.array([[0, 1],
                          [1, 0]])

W_pontine_CPU1a = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #2
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #15
])
W_pontine_CPU1b = np.array([
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #9
])
W_CPU4_pontine = np.eye(N_CPU4)

def tl2_output(noise):
    """Just a dot product with preferred angle and current heading""" # bad description
    def f(inputs):
        theta, = inputs
        output = np.cos(theta - tl2_prefs)
        return noisy_sigmoid(output, tl2_slope_tuned, tl2_bias_tuned, noise)
    return f

def cl1_output(noise):
    """Takes input from the TL2 neurons and gives output."""
    def f(inputs):
        tl2, = inputs
        return noisy_sigmoid(-tl2, cl1_slope_tuned, cl1_bias_tuned, noise)
    return f

def tb1_output(noise):
    """Ring attractor state on the protocerebral bridge."""
    def f(inputs):
        cl1, tb1 = inputs
        prop_cl1 = 0.667   # Proportion of input from CL1 vs TB1
        prop_tb1 = 1.0 - prop_cl1
        output = (prop_cl1 * np.dot(W_CL1_TB1, cl1) -
                    prop_tb1 * np.dot(W_TB1_TB1, tb1))
        return noisy_sigmoid(output, tb1_slope_tuned, tb1_bias_tuned, noise)
    return f

def tn1_output(noise):
    def f(inputs):
        flow, = inputs
        output = (1.0 - flow) / 2.0
        if noise > 0.0:
            output += np.random.normal(scale=noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)
    return f

def tn2_output(noise):
    def f(inputs):
        flow, = inputs
        output = flow
        if noise > 0.0:
            output += np.random.normal(scale=noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)
    return f

def cpu1a_output(noise):
    def f(inputs):
        """The memory and direction used together to get population code for
        heading."""
        tb1, cpu4 = inputs
        inputs = np.dot(W_CPU4_CPU1a, cpu4) * np.dot(W_TB1_CPU1a, 1.0-tb1)
        return noisy_sigmoid(inputs, cpu1_slope_tuned, cpu1_bias_tuned, noise)
    return f

def cpu1b_output(noise):
    def f(inputs):
        """The memory and direction used together to get population code for
        heading."""
        tb1, cpu4 = inputs
        inputs = np.dot(W_CPU4_CPU1b, cpu4) * np.dot(W_TB1_CPU1b, 1.0-tb1)
        return noisy_sigmoid(inputs, cpu1_slope_tuned, cpu1_bias_tuned, noise)
    return f

#def cpu1_output(noise):
#    def f(inputs):
#        tb1, cpu4 = inputs
#        cpu1a = cpu1a_output([tb1, cpu4], noise)
#        cpu1b = cpu1b_output([tb1, cpu4], noise)
#        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])
#    return f

def motor_output(noise):
    """outputs a scalar where sign determines left or right turn."""
    def f(inputs):
        cpu1a, cpu1b = inputs
        motor = np.dot(W_CPU1a_motor, cpu1a)
        motor += np.dot(W_CPU1b_motor, cpu1b)
        output = (motor[0] - motor[1]) * 0.25  # To kill the noise a bit!
        return output
    return f

def pontine_output(noise):
    def f(inputs):
        cpu4, = inputs
        inputs = np.dot(W_CPU4_pontine, cpu4)
        return noisy_sigmoid(inputs, pontine_slope_tuned, pontine_bias_tuned, noise)
    return f

def cpu1a_pontine_output(noise):
    def f(inputs):
        """The memory and direction used together to get population code for
        heading."""
        reference, cpu4, pontine = inputs

        inputs = 0.5 * np.dot(W_CPU4_CPU1a, cpu4)

        inputs -= 0.5 * np.dot(W_pontine_CPU1a, pontine)
        inputs -= reference

        return noisy_sigmoid(inputs, cpu1_pontine_slope_tuned, cpu1_pontine_bias_tuned, noise)
    return f

def cpu1b_pontine_output(noise):
    def f(inputs):
        """The memory and direction used together to get population code for
        heading."""
        reference, cpu4, pontine = inputs

        inputs = 0.5 * np.dot(W_CPU4_CPU1b, cpu4)

        inputs -= 0.5 * np.dot(W_pontine_CPU1b, pontine)
        inputs -= reference

        return noisy_sigmoid(inputs, cpu1_pontine_slope_tuned, cpu1_pontine_bias_tuned, noise)
    return f



def build_network(params, CPU4LayerClass = CPU4Layer) -> Network:
    noise = params.get("noise",0.1)
    cpu4_mem_gain = params.get("cpu4_mem_gain",0.005)

    # TODO: allow noisy weights
    return RecurrentForwardNetwork({
        "flow": InputLayer(initial = np.zeros(2)),
        "heading": InputLayer(),
        "TL2": FunctionLayer(
            inputs = ["heading"],
            function = tl2_output(noise),
            initial = np.zeros(N_TL2),
        ),
        "CL1": FunctionLayer(
            inputs = ["TL2"],
            function = cl1_output(noise),
            initial = np.zeros(N_CL1),
        ),
        "TB1": FunctionLayer(
            inputs = ["CL1", "TB1"],
            function = tb1_output(noise),
            initial = np.zeros(N_TB1),
        ),
        "TN1": FunctionLayer(
            inputs = ["flow"],
            function = tn1_output(noise),
            initial = np.zeros(N_TN1),
        ),
        "TN2": FunctionLayer(
            inputs = ["flow"],
            function = tn2_output(noise),
            initial = np.zeros(N_TN2),
        ),
        "memory": CPU4LayerClass(
            "TB1", "TN1", "TN2",
            W_TN_CPU4,
            W_TB1_CPU4,
            cpu4_mem_gain,
            cpu4_slope_tuned,
            cpu4_bias_tuned,
            noise,
        ),
        "CPU1a": FunctionLayer(
            inputs = ["TB1", "memory"],
            function = cpu1a_output(noise),
            initial = np.zeros(N_CPU1A)
        ),
        "CPU1b": FunctionLayer(
            inputs = ["TB1", "memory"],
            function = cpu1b_output(noise),
            initial = np.zeros(N_CPU1A)
        ),
        "motor": FunctionLayer(
            inputs = ["CPU1a", "CPU1b"],
            function = motor_output(noise),
        )
    })

def build_network_pontine(params) -> Network:
    # TODO: allow noisy weights
    noise = params.get("noise",0.1)
    cpu4_mem_gain = params.get("cpu4_mem_gain",0.0025)

    return RecurrentForwardNetwork({
        "flow": InputLayer(initial = np.zeros(2)),
        "heading": InputLayer(),
        "TL2": FunctionLayer(
            inputs = ["heading"],
            function = tl2_output(noise),
            initial = np.zeros(N_TL2),
        ),
        "CL1": FunctionLayer(
            inputs = ["TL2"],
            function = cl1_output(noise),
            initial = np.zeros(N_CL1),
        ),
        "TB1": FunctionLayer(
            inputs = ["CL1", "TB1"],
            function = tb1_output(noise),
            initial = np.zeros(N_TB1),
        ),
        "TN1": FunctionLayer(
            inputs = ["flow"],
            function = tn1_output(noise),
            initial = np.zeros(N_TN1),
        ),
        "TN2": FunctionLayer(
            inputs = ["flow"],
            function = tn2_output(noise),
            initial = np.zeros(N_TN2),
        ),
        "memory": CPU4PontineLayer(
            "TB1", "TN1", "TN2",
            W_TN_CPU4,
            W_TB1_CPU4,
            cpu4_mem_gain,
            cpu4_slope_tuned,
            cpu4_bias_tuned,
            noise,
        ),
        "Pontine": FunctionLayer(
            inputs = ["memory"],
            function = pontine_output(noise),
            initial = np.zeros(N_Pontine)
        ),
        "CPU1a": FunctionLayer(
            inputs = [WeightedSynapse("TB1", W_TB1_CPU1a), "memory", "Pontine"],
            function = cpu1a_pontine_output(noise),
            initial = np.zeros(N_CPU1A),
        ),
        "CPU1b": FunctionLayer(
            inputs = [WeightedSynapse("TB1", W_TB1_CPU1b), "memory", "Pontine"],
            function = cpu1b_pontine_output(noise),
            initial = np.zeros(N_CPU1B),
        ),
        "motor": FunctionLayer(
            inputs = ["CPU1a", "CPU1b"],
            function = motor_output(noise),
        )
    })
