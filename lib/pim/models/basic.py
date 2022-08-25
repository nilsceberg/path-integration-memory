import numpy as np

from ..network import InputLayer, Network, Output, RecurrentForwardNetwork, FunctionLayer, IdentityLayer, Layer
from .constants import *


def tb1_output(inputs):
    """Sinusoidal response to solar compass."""
    theta, tb1 = inputs
    return (1.0 + np.cos(np.pi + column_angles + theta)) / 2.0

def tn1_output(inputs):
    """Linearly inverse sensitive to forwards and backwards motion."""
    flow, = inputs
    return np.clip((1.0 - flow) / 2.0, 0, 1)

def tn2_output(inputs):
    """Linearly sensitive to forwards motion only."""
    flow, = inputs
    return np.clip(flow, 0, 1)

class CPU4Layer(Layer):
    def __init__(self, TB1, TN1, TN2, gain):
        self.TB1 = TB1
        self.TN1 = TN1
        self.TN2 = TN2
        self.gain = gain

        self.memory = np.ones(N_CPU4) * 0.5
        super().__init__(initial = self.memory)

    def step(self, network: Network, dt: float):
        """Updates memory based on current TB1 and TN activity.
        Can think of this as summing sinusoid of TB1 onto sinusoid of CPU4.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        tb1 = network.output(self.TB1)
        tn1 = network.output(self.TN1)
        tn2 = network.output(self.TN2)

        mem_reshaped = self.memory.reshape(2, -1)

        # Idealised setup, where we can negate the TB1 sinusoid
        # for memorising backwards motion
        mem_update = (0.5 - tn1.reshape(2, 1)) * (1.0 - tb1)

        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        mem_update -= 0.5 * (0.5 - tn1.reshape(2, 1))

        # Constant purely to visualise same as rate-based model
        mem_reshaped += self.gain * mem_update * dt

        self.memory = np.clip(mem_reshaped.reshape(-1), 0.0, 1.0)

    def output(self, dt: float) -> Output:
        return self.memory

def cpu1_output(inputs):
    tb1, cpu4 = inputs
    """Offset CPU4 columns by 1 column (45 degrees) left and right
    wrt TB1."""

    cpu4_reshaped = cpu4.reshape(2, -1)
    cpu1 = (1.0 - tb1) * np.vstack([np.roll(cpu4_reshaped[1], 1),
                                    np.roll(cpu4_reshaped[0], -1)])
    return cpu1.reshape(-1)

def motor_output(inputs):
    cpu1, = inputs
    random_std=0.05
    """Sum CPU1 to determine left or right turn."""
    cpu1_reshaped = cpu1.reshape(2, -1)
    motor_lr = np.sum(cpu1_reshaped, axis=1)
    # We need to add some randomness, otherwise agent infinitely overshoots
    motor = (motor_lr[1] - motor_lr[0])
    if random_std > 0.0:
        motor += np.random.normal(0, random_std)
    return motor


def build_network(params) -> Network:
    cpu4_mem_gain = params.get("cpu4_mem_gain",0.0025)

    return RecurrentForwardNetwork({
        "flow": InputLayer(initial = np.zeros(2)),
        "heading": InputLayer(),
        "CL1": IdentityLayer("heading"),
        "TB1": FunctionLayer(
            inputs = ["CL1", "TB1"],
            function = tb1_output,
            initial = np.zeros(N_TB1),
        ),
        "TN1": FunctionLayer(
            inputs = ["flow"],
            function = tn1_output,
            initial = np.zeros(N_TN1),
        ),
        "TN2": FunctionLayer(
            inputs = ["flow"],
            function = tn2_output,
            initial = np.zeros(N_TN2),
        ),
        "memory": CPU4Layer("TB1", "TN1", "TN2", gain=cpu4_mem_gain),
        "CPU1": FunctionLayer(
            inputs = ["TB1", "memory"],
            function = cpu1_output,
            initial = np.zeros(N_CPU1),
        ),
        "motor": FunctionLayer(
            inputs = ["CPU1"],
            function = motor_output,
        )
    })
