import numpy as np

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from abc import abstractmethod

from pim.models import rate
from ..network import Network, RecurrentForwardNetwork, InputLayer, Layer, FunctionLayer, Output
from .constants import *


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
        return network.output("CPU4") * self.weights * 5 - network.output("CPU4") * 1
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

        print(epsilon)

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

def build_dye_network(params) -> Network:

    noise = params.get("noise", 0.1)

    epsilon = params.get("epsilon", 1.0)
    length = params.get("length", 10e-4)
    T_half = params.get("T_half", 1.0)
    beta = params.get("beta", 0.0)
    phi = params.get("phi", 1.0)
    c_tot = params.get("c_tot", 0.3)

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
            function = rate.tb1_output(noise),
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
            gain = 0.0025 * 10,
            slope = cpu4_slope_tuned,
            bias = cpu4_bias_tuned,
            noise = noise,
            background_activity = beta,
        ),
        "memory": AdvancedDyeLayer(
            epsilon = epsilon, 
            length = length, 
            T_half = T_half, 
            beta = beta,
            phi = phi,
            c_tot = c_tot
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
            function = rate.motor_output(noise),
        )
    })