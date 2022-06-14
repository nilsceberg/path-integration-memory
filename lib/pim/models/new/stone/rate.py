import numpy as np
from scipy.special import expit

from ...network import Network, RecurrentForwardNetwork, FunctionLayer, IdentityLayer, Layer, Output
from .constants import *
from .cx import CentralComplex


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

def decode_position(cpu4_reshaped, cpu4_mem_gain):
    """Decode position from sinusoid in to polar coordinates.
    Amplitude is distance, Angle is angle from nest outwards.
    Without offset angle gives the home vector.
    Input must have shape of (2, -1)"""
    signal = np.sum(cpu4_reshaped, axis=0)
    fund_freq = np.fft.fft(signal)[1]
    angle = -np.angle(np.conj(fund_freq))
    distance = np.absolute(fund_freq) / cpu4_mem_gain
    return angle, distance


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
                     self.gain * np.dot(self.W_TB1, 1.0-tb1))
        self.memory -= self.gain * 0.25 * np.dot(self.W_TN, tn2)
        self.memory = np.clip(self.memory, 0.0, 1.0)

    def output(self, network: Network) -> Output:
        """The output from memory neuron, based on current calcium levels."""
        return noisy_sigmoid(self.memory, self.slope,
                             self.bias, self.noise)

class CPU4PontinLayer(CPU4Layer):
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
        self.memory += mem_update
        self.memory -= 0.125 * self.gain
        self.memory = np.clip(self.memory, 0.0, 1.0)


class CXRate(CentralComplex):
    def build_network(self) -> Network:
        return RecurrentForwardNetwork({
            "flow": self.flow_input,
            "heading": self.heading_input,
            "TL2": FunctionLayer(
                inputs = ["heading"],
                function = self.tl2_output,
                initial = np.zeros(N_TL2),
            ),
            "CL1": FunctionLayer(
                inputs = ["TL2"],
                function = self.cl1_output,
                initial = np.zeros(N_CL1),
            ),
            "TB1": FunctionLayer(
                inputs = ["CL1", "TB1"],
                function = self.tb1_output,
                initial = self.tb1,
            ),
            "TN1": FunctionLayer(
                inputs = ["flow"],
                function = self.tn1_output,
                initial = np.zeros(N_TN1),
            ),
            "TN2": FunctionLayer(
                inputs = ["flow"],
                function = self.tn2_output,
                initial = np.zeros(N_TN2),
            ),
            "CPU4": CPU4Layer(
                "TB1", "TN1", "TN2",
                self.W_TN_CPU4,
                self.W_TB1_CPU4,
                self.cpu4_mem_gain,
                self.cpu4_slope,
                self.cpu4_bias,
                self.noise,
            ),
            "CPU1": FunctionLayer(
                inputs = ["TB1", "CPU4"],
                function = self.cpu1_output,
                initial = np.zeros(N_CPU1),
            ),
            "motor": FunctionLayer(
                inputs = ["CPU1"],
                function = self.motor_output,
            )
        })

    """Class to keep a set of parameters for a model together.
    No state is held in the class currently."""
    def __init__(self,
                 noise=0.1,
                 tl2_slope=tl2_slope_tuned,
                 tl2_bias=tl2_bias_tuned,
                 tl2_prefs=np.tile(np.linspace(0, 2*np.pi, N_TB1,
                                               endpoint=False), 2),
                 cl1_slope=cl1_slope_tuned,
                 cl1_bias=cl1_bias_tuned,
                 tb1_slope=tb1_slope_tuned,
                 tb1_bias=tb1_bias_tuned,
                 cpu4_slope=cpu4_slope_tuned,
                 cpu4_bias=cpu4_bias_tuned,
                 cpu1_slope=cpu1_slope_tuned,
                 cpu1_bias=cpu1_bias_tuned,
                 motor_slope=motor_slope_tuned,
                 motor_bias=motor_bias_tuned,
                 weight_noise=0.0,
                 ):
        super().__init__()

        # Default noise used by the model for all layers
        self.noise = noise

        # Weight matrices based on anatomy. These are not changeable!)
        self.W_CL1_TB1 = np.tile(np.eye(N_TB1), 2)
        self.W_TB1_TB1 = gen_tb_tb_weights()
        self.W_TB1_CPU1a = np.tile(np.eye(N_TB1), (2, 1))[1:N_CPU1A+1, :]
        self.W_TB1_CPU1b = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                     [1, 0, 0, 0, 0, 0, 0, 0]])
        self.W_TB1_CPU4 = np.tile(np.eye(N_TB1), (2, 1))
        self.W_TN_CPU4 = np.array([
                [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
            ]).T
        self.W_CPU4_CPU1a = np.array([
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
        self.W_CPU4_CPU1b = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], #9
            ])
        self.W_CPU1a_motor = np.array([
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
        self.W_CPU1b_motor = np.array([[0, 1],
                                       [1, 0]])

        if weight_noise > 0.0:
            self.W_CL1_TB1 = noisify_weights(self.W_CL1_TB1, weight_noise)
            self.W_TB1_TB1 = noisify_weights(self.W_TB1_TB1, weight_noise)
            self.W_TB1_CPU1a = noisify_weights(self.W_TB1_CPU1a, weight_noise)
            self.W_TB1_CPU1b = noisify_weights(self.W_TB1_CPU1b, weight_noise)
            self.W_TB1_CPU4 = noisify_weights(self.W_TB1_CPU4, weight_noise)
            self.W_CPU4_CPU1a = noisify_weights(self.W_CPU4_CPU1a,
                                                weight_noise)
            self.W_CPU4_CPU1b = noisify_weights(self.W_CPU4_CPU1b,
                                                weight_noise)
            self.W_CPU1a_motor = noisify_weights(self.W_CPU1a_motor,
                                                 weight_noise)
            self.W_CPU1b_motor = noisify_weights(self.W_CPU1b_motor,
                                                 weight_noise)
        # The cell properties (for sigmoid function)
        self.tl2_slope = tl2_slope
        self.tl2_bias = tl2_bias
        self.tl2_prefs = tl2_prefs
        self.cl1_bias = cl1_bias
        self.cl1_slope = cl1_slope
        self.tb1_slope = tb1_slope
        self.tb1_bias = tb1_bias
        self.cpu4_slope = cpu4_slope
        self.cpu4_bias = cpu4_bias
        self.cpu1_slope = cpu1_slope
        self.cpu1_bias = cpu1_bias
        self.motor_slope = motor_slope
        self.motor_bias = motor_bias
        

    def tl2_output(self, inputs):
        """Just a dot product with preferred angle and current heading""" # bad description
        theta, = inputs
        output = np.cos(-theta - self.tl2_prefs)
        return noisy_sigmoid(output, self.tl2_slope, self.tl2_bias, self.noise)

    def cl1_output(self, inputs):
        """Takes input from the TL2 neurons and gives output."""
        tl2, = inputs
        return noisy_sigmoid(-tl2, self.cl1_slope, self.cl1_bias, self.noise)

    def tb1_output(self, inputs):
        """Ring attractor state on the protocerebral bridge."""
        cl1, tb1 = inputs
        prop_cl1 = 0.667   # Proportion of input from CL1 vs TB1
        prop_tb1 = 1.0 - prop_cl1
        output = (prop_cl1 * np.dot(self.W_CL1_TB1, cl1) -
                  prop_tb1 * np.dot(self.W_TB1_TB1, tb1))
        return noisy_sigmoid(output, self.tb1_slope, self.tb1_bias, self.noise)

    def tn1_output(self, inputs):
        flow, = inputs
        output = (1.0 - flow) / 2.0
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def tn2_output(self, inputs):
        flow, = inputs
        output = flow
        if self.noise > 0.0:
            output += np.random.normal(scale=self.noise, size=flow.shape)
        return np.clip(output, 0.0, 1.0)

    def cpu1a_output(self, inputs):
        """The memory and direction used together to get population code for
        heading."""
        tb1, cpu4 = inputs
        inputs = np.dot(self.W_CPU4_CPU1a, cpu4) * np.dot(self.W_TB1_CPU1a,
                                                          1.0-tb1)
        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                             self.noise)

    def cpu1b_output(self, inputs):
        """The memory and direction used together to get population code for
        heading."""
        tb1, cpu4 = inputs
        inputs = np.dot(self.W_CPU4_CPU1b, cpu4) * np.dot(self.W_TB1_CPU1b,
                                                          1.0-tb1)

        return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                             self.noise)

    def cpu1_output(self, inputs):
        tb1, cpu4 = inputs
        cpu1a = self.cpu1a_output([tb1, cpu4])
        cpu1b = self.cpu1b_output([tb1, cpu4])
        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])

    def motor_output(self, inputs):
        """outputs a scalar where sign determines left or right turn."""
        cpu1, = inputs
        cpu1a = cpu1[1:-1]
        cpu1b = np.array([cpu1[-1], cpu1[0]])
        motor = np.dot(self.W_CPU1a_motor, cpu1a)
        motor += np.dot(self.W_CPU1b_motor, cpu1b)
        output = (motor[0] - motor[1]) * 0.25  # To kill the noise a bit!
        return -output


class CXRatePontin(CXRate):
    def build_network(self) -> Network:
        return RecurrentForwardNetwork({
            "flow": self.flow_input,
            "heading": self.heading_input,
            "TL2": FunctionLayer(
                inputs = ["heading"],
                function = self.tl2_output,
                initial = np.zeros(N_TL2),
            ),
            "CL1": FunctionLayer(
                inputs = ["TL2"],
                function = self.cl1_output,
                initial = np.zeros(N_CL1),
            ),
            "TB1": FunctionLayer(
                inputs = ["CL1", "TB1"],
                function = self.tb1_output,
                initial = self.tb1,
            ),
            "TN1": FunctionLayer(
                inputs = ["flow"],
                function = self.tn1_output,
                initial = np.zeros(N_TN1),
            ),
            "TN2": FunctionLayer(
                inputs = ["flow"],
                function = self.tn2_output,
                initial = np.zeros(N_TN2),
            ),
            "CPU4": CPU4PontinLayer(
                "TB1", "TN1", "TN2",
                self.W_TN_CPU4,
                self.W_TB1_CPU4,
                self.cpu4_mem_gain,
                self.cpu4_slope,
                self.cpu4_bias,
                self.noise,
            ),
            "Pontin": FunctionLayer(
                inputs = ["CPU4"],
                function = self.pontin_output,
                initial = np.zeros(N_Pontin)
            ),
            "CPU1": FunctionLayer(
                inputs = ["TB1", "CPU4", "Pontin"],
                function = self.cpu1_output,
                initial = np.zeros(N_CPU1),
            ),
            "motor": FunctionLayer(
                inputs = ["CPU1"],
                function = self.motor_output,
            )
        })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.cpu4_mem_gain *= 0.5
        self.cpu1_bias = -1.0
        self.cpu1_slope = 7.5

        # Pontine cells
        self.pontin_slope = 5.0
        self.pontin_bias = 2.5

        self.W_pontin_CPU1a = np.array([
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
        self.W_pontin_CPU1b = np.array([
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #8
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], #9
            ])
        self.W_CPU4_pontin = np.eye(N_CPU4)

    def pontin_output(self, inputs):
        cpu4, = inputs
        inputs = np.dot(self.W_CPU4_pontin, cpu4)
        return noisy_sigmoid(inputs, self.pontin_slope, self.pontin_bias,
                             self.noise)

    def cpu1a_output(self, inputs):
       """The memory and direction used together to get population code for
       heading."""
       tb1, cpu4, pontin = inputs

       inputs = 0.5 * np.dot(self.W_CPU4_CPU1a, cpu4)

       inputs -= 0.5 * np.dot(self.W_pontin_CPU1a, pontin)
       inputs -= np.dot(self.W_TB1_CPU1a, tb1)

       return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                            self.noise)

    def cpu1b_output(self, inputs):
       """The memory and direction used together to get population code for
       heading."""
       tb1, cpu4, pontin = inputs

       inputs = 0.5 * np.dot(self.W_CPU4_CPU1b, cpu4)

       inputs -=  0.5 * np.dot(self.W_pontin_CPU1b, pontin)
       inputs -= np.dot(self.W_TB1_CPU1b, tb1)

       return noisy_sigmoid(inputs, self.cpu1_slope, self.cpu1_bias,
                            self.noise)

    def cpu1_output(self, inputs):
        tb1, cpu4, pontin = inputs
        cpu1a = self.cpu1a_output([tb1, cpu4, pontin])
        cpu1b = self.cpu1b_output([tb1, cpu4, pontin])
        return np.hstack([cpu1b[-1], cpu1a, cpu1b[0]])
