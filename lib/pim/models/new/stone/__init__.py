from abc import abstractmethod
from ...network import IdentityLayer, InputLayer, ForwardNetwork, FunctionLayer, Network, RecurrentNetwork
import numpy as np
import scipy.optimize

from .constants import *
from . import basic


def cpu4_model(args, x):
    return args[0] * np.cos(np.pi + 2*x + args[1])

def tb1_model(args, x):
    return 0.5 + np.cos(np.pi + x + args[0]) * 0.5

def fit_cpu4(data):
    xs = np.linspace(0, 2*np.pi, N_CPU4, endpoint = False)
    error = lambda args: cpu4_model(args, xs) - data
    params, _ = scipy.optimize.leastsq(error, np.array([0.1, 0.01]))
    return params

def fit_tb1(data):
    xs = np.linspace(0, 2*np.pi, N_TB1, endpoint = False)
    error = lambda args: tb1_model(args, xs) - data
    params, _ = scipy.optimize.leastsq(error, np.array([0.1]))
    return params


class CentralComplex:
    def __init__(self, tn_prefs=np.pi/4.0,
                 cpu4_mem_gain=0.005):
        self.tn_prefs = tn_prefs
        self.cpu4_mem_gain = cpu4_mem_gain
        self.smoothed_flow = 0

        self.tb1 = np.zeros(N_TB1)
        self.cpu4 = 0.5 * np.ones(N_CPU4)

        self.flow_input = InputLayer(initial = np.zeros(2))
        self.heading_input = InputLayer()

        self.network = self.build_network()

    @abstractmethod
    def build_network(self) -> Network:
        pass

    def update(self, dt, heading, velocity):
        flow = self.get_flow(heading, velocity)
        self.flow_input.set(flow)
        self.heading_input.set(np.array([heading]))

        self.network.step(dt)

        self.tb1 = self.network.output("TB1")
        self.cpu4 = self.network.output("CPU4")
        return self.network.output("motor")

    def estimate_position(self):
        return fit_cpu4(self.cpu4)

    def to_cartesian(self, polar):
        return np.array([
            np.cos(polar[1] + np.pi),
            np.sin(polar[1] + np.pi),
        ]) * polar[0]

    def estimate_heading(self):
        return fit_tb1(self.tb1)

    def get_flow(self, heading, velocity, filter_steps=0):
        """Calculate optic flow depending on preference angles. [L, R]"""
        A = np.array([[np.sin(heading + self.tn_prefs),
                       np.cos(heading + self.tn_prefs)],
                      [np.sin(heading - self.tn_prefs),
                       np.cos(heading - self.tn_prefs)]])
        flow = np.dot(A, velocity)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self.smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                  1.0 / filter_steps) * self.smoothed_flow)
            flow = self.smoothed_flow
        return flow


class CXBasic(CentralComplex):
    def build_network(self) -> Network:
        return RecurrentNetwork({
            "flow": self.flow_input,
            "TL2": self.heading_input,
            "CL1": IdentityLayer("TL2"),
            "TB1": FunctionLayer(
                inputs = ["CL1", "TB1"],
                function = basic.tb1_output,
                initial = self.tb1,
            ),
            "TN1": FunctionLayer(
                inputs = ["flow"],
                function = basic.tn1_output,
                initial = np.zeros(N_TN1),
            ),
            "TN2": FunctionLayer(
                inputs = ["flow"],
                function = basic.tn2_output,
                initial = np.zeros(N_TN2),
            ),
            "CPU4": FunctionLayer(
                inputs = ["CPU4", "TB1", "TN1", "TN2"],
                function = basic.cpu4_output(cpu4_mem_gain=0.01),
                initial = self.cpu4,
            ),
#           "CPU4": FunctionLayer(
#               inputs = ["CPU4", "TB1", "TN1", "TN2"],
#               function = cpu4_bistable_output(cpu4_mem_gain=0.05, N=400, dI=1/300, mI=1.0),
#               initial = self.cpu4,
#            ),
            "CPU1": FunctionLayer(
                inputs = ["TB1", "CPU4"],
                function = basic.cpu1_output,
                initial = np.zeros(N_CPU1),
            ),
            "motor": FunctionLayer(
                inputs = ["CPU1"],
                function = basic.motor_output,
            )
        })