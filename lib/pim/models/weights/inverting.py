from .. import rate
from ...network import Network, RecurrentForwardNetwork, InputLayer, FunctionLayer
from ..constants import *
from . import PlasticWeightLayer, motor_output

import numpy as np

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

