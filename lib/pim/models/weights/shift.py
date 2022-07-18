from .. import rate
from ...network import Network, RecurrentForwardNetwork, InputLayer, FunctionLayer, WeightedSynapse
from ..constants import *
from . import PlasticWeightLayer, pontine_output, motor_output, cpu1_pontine_output

import numpy as np


def motor_output_theoretical(noise):
    def f(inputs):
        memory, cpu4, tb1, pontine = inputs
        #tb1 = 0.5*tb1

        left_pontine = np.roll(memory[:8], 3)
        right_pontine = np.roll(memory[8:], -3)

        left = np.clip(0.5 * (np.roll(memory[:8], -1) - left_pontine), 0, 100) # type: ignore
        right = np.clip(0.5 * (np.roll(memory[8:], 1) - right_pontine), 0, 100) # type: ignore

        #pfn_left = np.clip(0.5 * (cpu4[:8] - np.roll(cpu4[:8], 4)), 0, 100)
        #pfn_right = np.clip(0.5 * (cpu4[8:] - np.roll(cpu4[8:], -4)), 0, 100)
        pfn_left = cpu4[:8]
        pfn_right = cpu4[8:]

        bias = 5.0
        slope = 100
        deltaright = rate.noisy_sigmoid(right - pfn_right, slope, bias, noise)
        deltaleft = rate.noisy_sigmoid(left - pfn_left, slope, bias, noise)

        #deltaright = np.clip(right - cpu4[8:], 0, 1000)
        #deltaleft = np.clip(left - cpu4[:8], 0, 1000)

        motor = np.sum(deltaright) - np.sum(deltaleft)
        return motor #+ np.random.normal(0, noise)
    return f


def build_phase_shift_network(params) -> Network:
    # TODO: allow noisy weights
    noise = params.get("noise", 0.1)
    mem_gain = params.get("mem_gain", 0.0025)
    mem_fade = params.get("mem_fade", 0.1)
    pfn_weight_factor = params.get("pfn_weight_factor", 1)

    W_CPU4_CPU1 = np.eye(N_CPU4)

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
        #"CL1": IdentityLayer("heading"),
        "TB1": FunctionLayer(
            inputs = ["CL1", "TB1"],
            #function = basic.tb1_output, #(noise),
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
        "CPU4.old": rate.CPU4PontineLayer(
            "TB1", "TN1", "TN2",
            rate.W_TN_CPU4,
            rate.W_TB1_CPU4,
            gain = 1.0,
            slope = cpu4_slope_tuned,
            bias = cpu4_bias_tuned,
            noise = noise,
        ),
        "CPU4": rate.MemorylessCPU4Layer(
            "TB1", "TN1", "TN2",
            rate.W_TN_CPU4,
            rate.W_TB1_CPU4,
            gain = pfn_weight_factor,
            slope = cpu4_slope_tuned,
            bias = cpu4_bias_tuned,
            noise = noise,
            background_activity = 0.0,
        ),
        "memory": PlasticWeightLayer(noise, mem_gain, mem_fade),
        "Pontine": FunctionLayer(
            inputs = ["memory"],
            function = pontine_output(noise),
            initial = np.zeros(N_Pontine)
        ),
        "theory": FunctionLayer(
            inputs = ["memory", "CPU4", "TB1", "Pontine"],
            function = motor_output_theoretical(noise)
        ),
        "CPU1": FunctionLayer(
            inputs = [WeightedSynapse("CPU4", W_CPU4_CPU1), "memory", "Pontine"],
            function = cpu1_pontine_output(
                noise,
                params.get("cpu1_slope", cpu1_pontine_slope_tuned),
                params.get("cpu1_bias", cpu1_pontine_bias_tuned),
            ),
            initial = np.zeros(N_CPU1),
        ),
        "motor": FunctionLayer(
            inputs = ["CPU1"],
            function = motor_output(noise),
        )
    })
