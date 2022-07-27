import numpy as np
from pim.models.weights.shift import W_CPU4_CPU1a, W_CPU4_CPU1b, motor_output_theoretical

from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from abc import abstractmethod

from pim.models import rate
from pim.models.weights import cpu1a_pontine_output, cpu1b_pontine_output, motor_output, pontine_output
from ..network import Network, RecurrentForwardNetwork, InputLayer, Layer, FunctionLayer, Output, WeightedSynapse
from .constants import *


class DyeLayer(Layer):
    def __init__(self):
        self.weights = np.ones(N_CPU4) * 0.5
        super().__init__(initial = np.zeros(N_CPU4))

    def step(self, network: Network, dt: float):
        self.update_weights(network.output("CPU4"), dt)

    def internal(self):
        return self.weights

    @abstractmethod
    def update_weights(self, inputs: Output, dt: float):
        pass

    def output(self, network: Network) -> Output:
        # return network.output("CPU4") * self.weights
        return np.log10(network.output("CPU4") * self.weights)#* 30000

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

        self.last_c = np.ones(16) * 0#* phi * beta / self.k

        super().__init__()

    def internal(self):
            return self.last_c

    def update_weights(self, inputs: Output, dt: float):

        def T(c):
            A = self.epsilon * self.length * (self.c_tot - c)
            return 10 ** -A

        def dcdt(t, c):
            return (-self.k * c + self.phi * inputs * (1 - T(c)))

        solution = solve_ivp(dcdt, y0 = self.last_c, t_span=(0, dt))
        c = solution.y[:,-1]

        self.last_c = c

        # if ((c > self.c_tot).any()):
        #     print(c)

        self.weights = T(c)
        # print(self.weights)

def build_dye_network(params) -> Network:

    epsilon = params.get("epsilon", 1.0)
    length = params.get("length", 10e-4)
    T_half = params.get("T_half", 1.0)
    beta = params.get("beta", 0.0)
    phi = params.get("phi", 1.0)
    c_tot = params.get("c_tot", 0.3)

    noise = params.get("noise", 0.1)
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
            gain = pfn_weight_factor,
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
            function = pontine_output(noise),
            initial = np.zeros(N_Pontine)
        ),
        "theory": FunctionLayer(
            inputs = ["memory", "CPU4", "TB1", "Pontine"],
            function = motor_output_theoretical(noise)
        ),
        "CPU1a": FunctionLayer(
            inputs = [WeightedSynapse("TB1", rate.W_TB1_CPU1a), "memory", "Pontine"],
            function = cpu1a_pontine_output(
                noise,
                params.get("cpu1_slope", cpu1_pontine_slope_tuned),
                params.get("cpu1_bias", cpu1_pontine_bias_tuned),
            ),
            initial = np.zeros(N_CPU1A),
        ),
        "CPU1b": FunctionLayer(
            inputs = [WeightedSynapse("TB1", rate.W_TB1_CPU1b), "memory", "Pontine"],
            function = cpu1b_pontine_output(
                noise,
                params.get("cpu1_slope", cpu1_pontine_slope_tuned),
                params.get("cpu1_bias", cpu1_pontine_bias_tuned),
            ),
            initial = np.zeros(N_CPU1B),
        ),
        "motor": FunctionLayer(
            inputs = ["CPU1a", "CPU1b"],
            function = motor_output(noise),
        )
    })