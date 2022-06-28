from abc import abstractmethod
from pim.models.new.stone.rate import CXRatePontine, noisy_sigmoid


from ...network import Network, RecurrentForwardNetwork, Layer, FunctionLayer, Output
from .constants import *
import numpy as np

class DyeCPU4Layer(Layer):
    def __init__(self, TB1, TN1, TN2, W_TN, W_TB1, gain, slope, bias, noise, background_activity = 1.0):
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

        super().__init__(initial = np.ones(N_CPU4) * self.background_activity)

    def output(self, network: Network) -> Output:
        tb1 = network.output(self.TB1)
        tn1 = network.output(self.TN1)
        tn2 = network.output(self.TN2)

        mem_update = np.dot(self.W_TN, tn2)
        mem_update -= np.dot(self.W_TB1, tb1)
        return noisy_sigmoid(mem_update, self.slope, self.bias, self.noise) + self.background_activity

class DyeLayer(Layer):
    def __init__(self):
        self.weights = np.ones(N_CPU4) * 0.5
        super().__init__(initial = np.zeros(N_CPU4))

    def step(self, network: Network, dt: float):
        self.update_weights(network.output("CPU4"), dt)
        # print(self.weights)

    @abstractmethod
    def update_weights(self, inputs: Output, dt: float):
        pass

    def output(self, network: Network) -> Output:
        return network.output("CPU4") * self.weights


class SimpleDyeLayer(DyeLayer):
    def __init__(self, gain=0.0025):
        self.gain = gain # K
        self.fade_rate = -0.125 * self.gain # gamma
        self.backreaction_rate = self.gain - self.fade_rate # B

        super().__init__()

    def update_weights(self, inputs: Output, dt: float):
        self.weights += -self.backreaction_rate*dt + self.gain*inputs*dt

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
            "memory": SimpleDyeLayer(),
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

    def __init__(self, **kwargs):
        super().__init__(DyeCPU4Layer, **kwargs)