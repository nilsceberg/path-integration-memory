from .models.weights import shift as weights_shift, inverting as weights_inverting
import numpy as np
import scipy.optimize

from .models import basic, rate, dye
from .models.constants import *
from .network import InputLayer, Network


def memory_model(args, x):
    return args[0] * np.cos(np.pi + 2*x + args[1])

def tb1_model(args, x):
    return 0.5 + np.cos(np.pi + x + args[0]) * 0.5

def fit_memory(data):
    xs = np.linspace(0, 2*np.pi, N_CPU4//2, endpoint = False)
    error = lambda args: memory_model(args, xs) - data
    params, _ = scipy.optimize.leastsq(func=error, x0=np.array([0, 0]))
    return params

def fit_memory_fft(data):
    data = data.reshape(2,-1)
    signal = np.sum(data, axis=0)
    fund_freq = np.fft.fft(signal)[1]
    angle = np.angle(np.conj(fund_freq))
    return (0, angle)

def fit_tb1(data):
    xs = np.linspace(0, 2*np.pi, N_TB1, endpoint = False)
    error = lambda args: tb1_model(args, xs) - data
    params, _ = scipy.optimize.leastsq(error, np.array([0.1]))
    return params

def to_cartesian(polar):
    return np.array([
        np.sin(polar[1] + np.pi),
        np.cos(polar[1] + np.pi),
    ]) * polar[0]


def build_network_from_json(params) -> Network:
    if params["type"] == "basic":
        return basic.build_network(params["params"])
    elif params["type"] == "rate":
        return rate.build_network(params["params"])
    elif params["type"] == "pontine":
        return rate.build_network_pontine(params["params"])
    elif params["type"] == "weights":
        return weights_shift.build_phase_shift_network(params["params"])
    elif params["type"] == "weights-inverting":
        return weights_inverting.build_inverting_network(params["params"])
    elif params["type"] == "dye":
        return dye.build_dye_network(params["params"])
    else:
        raise NotImplementedError()

def build_from_json(params):
    output_layer = params.get("output_layer", "motor")
    return CentralComplex(build_network_from_json(params), output_layer)


class CentralComplex:
    def __init__(self, network: Network, output_layer = "motor", tn_prefs=np.pi/4.0):
        self.tn_prefs = tn_prefs
        self.smoothed_flow = 0
        self.output_layer = output_layer

        self.tb1 = np.zeros(N_TB1)
        self.cpu4 = 0.5 * np.ones(N_CPU4)

        self.network = network
        assert isinstance(self.network.layers["flow"], InputLayer)
        assert isinstance(self.network.layers["heading"], InputLayer)
        self.flow_input = self.network.layers["flow"]
        self.heading_input = self.network.layers["heading"]

    def update(self, dt, heading, velocity):
        flow = self.get_flow(heading, velocity)
        self.flow_input.set(flow)
        self.heading_input.set(np.array([heading]))

        self.network.step(dt)

        self.tb1 = self.network.output("TB1")
        self.memory = self.network.output("memory")
        return self.network.output(self.output_layer)

    def setup(self):
        pass

    def estimate_position(self):
        return fit_memory(self.memory)

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

