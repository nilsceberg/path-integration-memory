from enum import Enum
import numpy as np
from pim.models.weights.shift import motor_output_theoretical

from scipy.integrate import solve_ivp
from scipy.special import expit
from abc import abstractmethod
from loguru import logger

from pim.models import rate
from pim.models.weights import motor_output, pontine_output
from ..network import Network, RecurrentForwardNetwork, InputLayer, Layer, FunctionLayer, Output, WeightedSynapse
from .constants import *
import pim.math

def cpu1a_pontine_output(noise, slope, bias, cheat, cheat_bias, cheat_slope):
    def f(inputs):
        """The memory and direction used together to get population code for
        heading."""
        reference, memory, pontine = inputs

        inputs = 0.5 * np.dot(rate.W_CPU4_CPU1a, memory)
        inputs -= 0.5 * np.dot(rate.W_pontine_CPU1a, pontine)

        #inputs = expit(100*(inputs-0.0001))
        if cheat:
            # inputs = (inputs > 0) * 1.0 # extra cheat
            inputs = expit(cheat_slope*(inputs-cheat_bias))

        inputs -= reference

        return rate.noisy_sigmoid(inputs, slope, bias, noise)
    return f

def cpu1b_pontine_output(noise, slope, bias, cheat, cheat_bias, cheat_slope):
    def f(inputs):
        """The memory and direction used together to get population code for
        heading."""
        reference, memory, pontine = inputs

        inputs = 0.5 * np.dot(rate.W_CPU4_CPU1b, memory)
        inputs -= 0.5 * np.dot(rate.W_pontine_CPU1b, pontine)

        if cheat:
            # inputs = (inputs > 0) * 1.0 # extra cheat
            inputs = expit(cheat_slope*(inputs-cheat_bias))

        inputs -= reference

        return rate.noisy_sigmoid(inputs, slope, bias, noise)
    return f

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
        return network.output("CPU4") * self.weights
        # return np.log10(network.output("CPU4") * self.weights)#* 30000

class SimpleDyeLayer(DyeLayer):
    def __init__(self, gain=0.0025):
        self.gain = gain # K (should be phi)
        self.fade_rate = -0.125 * self.gain # gamma
        self.backreaction_rate = self.gain - self.fade_rate # B (should be k)

        super().__init__()

    def update_weights(self, inputs: Output, dt: float):
        self.weights += -self.backreaction_rate*dt + self.gain*inputs*dt

class DyeReadout(Enum):
    WEIGHT = 1
    TRANSMITTANCE = 2
    CONCENTRATION = 3

class AdvancedDyeLayer(DyeLayer):
    def __init__(self, epsilon, length, k, phi, c_tot, volume, wavelength, W_max, readout: DyeReadout, model_transmittance):
        self.epsilon = epsilon
        self.length = length
        self.k = k
        E = 6.62697915 * 1e-34 * 299792458 / (wavelength * 1e-9)
        self.k_phi = W_max / (E * volume * 6.02214076*1e23)
        self.phi = phi
        self.c_tot = c_tot
        self.initialize(0.0)

        self.model_transmittance = model_transmittance
        self.readout = readout

        super().__init__()

    def internal(self):
        return self.last_c

    def transmittance(self, c):
        """The transmittance corresponds to the weight of the synapse."""
        if self.model_transmittance:
            A = self.epsilon * self.length * (self.c_tot - c)
            return 10 ** -A
        else:
            return c

    def dcdt(self, u):
        def f(t, c):
            T = self.transmittance(c)
            return -self.k * c + u * (1 - T) * self.phi #* self.k_phi
        return f

    def stable_point(self, u):
        from scipy.optimize import root_scalar
        roots = root_scalar(lambda c: self.dcdt(u)(0, c), bracket=[0, 1])
        return roots.root

    def initialize(self, c0):
        self.last_c = np.ones(16) * c0

    def update_weights(self, inputs: Output, dt: float):
        dcdt  = self.dcdt(inputs)
        self.last_c, T, Y = pim.math.step_ode(dcdt, self.last_c, dt)
        self.weights = self.transmittance(self.last_c)

        return T, Y

    def output(self, network: Network) -> Output:
        if self.readout == DyeReadout.WEIGHT:
            return network.output("CPU4") * self.weights
        elif self.readout == DyeReadout.TRANSMITTANCE:
            return self.transmittance(self.last_c)
        else: # CONCENTRATION
            return self.last_c / self.c_tot


def build_dye_network(params) -> Network:
    epsilon = params.get("epsilon", 1.0)
    length = params.get("length", 10e-4)
    T_half = params.get("T_half", 1.0)
    k = params.get("k", np.log(2) / T_half)
    beta = params.get("beta", 0.0)
    phi = params.get("phi", 1.0)
    c_tot = params.get("c_tot", 0.3)

    cheat = params.get("cheat", False)
    cheat_bias = params.get("cheat_bias", 0.0001)
    cheat_slope = params.get("cheat_slope", 100)

    volume = params.get("volume", 1e-18)
    wavelength = params.get("wavelength", 750)
    W_max = params.get("W_max", 1e-15)

    noise = params.get("noise", 0.1)
    pfn_weight_factor = params.get("pfn_weight_factor", 1)

    readout = DyeReadout[params.get("readout", "WEIGHT")]
    model_transmittance = params.get("model_transmittance", True)

    dye = AdvancedDyeLayer(
        epsilon = epsilon, 
        length = length, 
        k = k, 
        phi = phi,
        c_tot = c_tot,
        volume = volume,
        wavelength = wavelength,
        W_max = W_max,
        model_transmittance = model_transmittance,
        readout = readout,
    )

    if params.get("start_at_stable", False):
        stable_point = dye.stable_point(beta)
        logger.debug("starting at stable point: {}", stable_point)
        dye.initialize(stable_point)

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
        "memory": dye,
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
                cheat,
                cheat_bias,
                cheat_slope,
            ),
            initial = np.zeros(N_CPU1A),
        ),
        "CPU1b": FunctionLayer(
            inputs = [WeightedSynapse("TB1", rate.W_TB1_CPU1b), "memory", "Pontine"],
            function = cpu1b_pontine_output(
                noise,
                params.get("cpu1_slope", cpu1_pontine_slope_tuned),
                params.get("cpu1_bias", cpu1_pontine_bias_tuned),
                cheat,
                cheat_bias,
                cheat_slope,
            ),
            initial = np.zeros(N_CPU1B),
        ),
        "motor": FunctionLayer(
            inputs = ["CPU1a", "CPU1b"],
            function = motor_output(noise),
        )
    })
