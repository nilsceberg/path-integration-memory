from abc import abstractmethod
from cgi import print_arguments
from pim.models.new.stone.rate import CXRatePontine, MemorylessCPU4Layer, noisy_sigmoid

from ...network import Network, RecurrentForwardNetwork, Layer, FunctionLayer, Output
from .constants import *
import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

class DyeLayer(Layer):
    def __init__(self):
        self.weights = np.ones(N_CPU4) * 0.5
        super().__init__(initial = np.zeros(N_CPU4))

    def step(self, network: Network, dt: float):
        self.update_weights(network.output("CPU4"), dt)

    @abstractmethod
    def update_weights(self, inputs: Output, dt: float):
        pass

    def output(self, network: Network) -> Output:
        return network.output("CPU4") * self.weights * 5
        #return np.log10(network.output("CPU4") * self.weights) + 6 #* 30000


class SimpleDyeLayer(DyeLayer):
    def __init__(self, gain=0.0025):
        self.gain = gain # K (should be phi)
        self.fade_rate = -0.125 * self.gain # gamma
        self.backreaction_rate = self.gain - self.fade_rate # B (should be k)

        super().__init__()

    def update_weights(self, inputs: Output, dt: float):
        self.weights += -self.backreaction_rate*dt + self.gain*inputs*dt

class AdvancedDyeLayer(DyeLayer):
    def __init__(self, epsilon, length, T_half, phi, beta, c_tot):
        self.epsilon = epsilon
        self.length = length
        self.k = np.log(2) / T_half
        self.phi = phi
        self.c_tot = c_tot

        self.last_c = np.ones(16) * phi * beta / self.k

        super().__init__()

    def update_weights(self, inputs: Output, dt: float):

        def dcdt(t, c):
            return -self.k * c + self.phi * inputs


        solution = solve_ivp(dcdt, y0 = self.last_c, t_span=(0, dt*0.01))
        c = solution.y[:,-1]
        # print(c)
        self.last_c = c

        A = self.epsilon * (self.c_tot - c) * self.length

        self.weights = 10 ** -A

class CXDye(CXRatePontine):
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
            "CPU4": self.build_cpu4_layer(),
            "memory": AdvancedDyeLayer(
                epsilon = self.epsilon, 
                length = self.length, 
                T_half = self.T_half, 
                beta = self.beta,
                phi = self.phi,
                c_tot = self.c_tot
            ),
            "Pontine": FunctionLayer(
                inputs = ["memory"],
                function = self.pontine_output,
                initial = np.zeros(N_Pontine)
            ),
            "CPU1": FunctionLayer(
                inputs = ["TB1", "memory", "Pontine"],
                function = self.cpu1_output,
                initial = np.zeros(N_CPU1),
            ),
            "motor": FunctionLayer(
                inputs = ["CPU1"],
                function = self.motor_output,
            )
        })

    def __init__(self, phi=1.0, beta=1.0, T_half=1.0, epsilon=1.0, length=10e-4, c_tot=0.3, **kwargs):

        self.epsilon = epsilon
        self.length = length
        self.T_half = T_half
        self.phi = phi
        self.beta = beta
        self.c_tot = c_tot

        super().__init__(MemorylessCPU4Layer, **kwargs)

    def build_cpu4_layer(self) -> Layer:
        return self.CPU4LayerClass(
            "TB1", "TN1", "TN2",
            self.W_TN_CPU4,
            self.W_TB1_CPU4,
            self.cpu4_mem_gain * 10,
            self.cpu4_slope,
            self.cpu4_bias,
            self.noise,
            background_activity = self.beta
        )