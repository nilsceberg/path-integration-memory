import numpy as np

from ...network import Network, RecurrentNetwork, FunctionLayer, IdentityLayer
from .constants import *
from .bistable import bistable_neuron
from .cx import CentralComplex


def tb1_output(inputs):
    """Sinusoidal response to solar compass."""
    theta, tb1 = inputs
    return (1.0 + np.cos(np.pi + x + theta)) / 2.0

def tn1_output(inputs):
    """Linearly inverse sensitive to forwards and backwards motion."""
    flow, = inputs
    return np.clip((1.0 - flow) / 2.0, 0, 1)

def tn2_output(inputs):
    """Linearly sensitive to forwards motion only."""
    flow, = inputs
    return np.clip(flow, 0, 1)

def cpu4_bistable_output(cpu4_mem_gain, N, dI, mI):
    columns = [[bistable_neuron(I_up-dI, I_up) for I_up in np.linspace((mI-dI)/N, mI, N)] for i in range(N_CPU4)]

    def closure(inputs):
        """Updates memory based on current TB1 and TN activity.
        Can think of this as summing sinusoid of TB1 onto sinusoid of CPU4.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        cpu4_mem, tb1, tn1, tn2 = inputs
        cpu4_mem_reshaped = cpu4_mem.reshape(2, -1)

        # Idealised setup, where we can negate the TB1 sinusoid
        # for memorising backwards motion
        mem_update = (0.5 - tn1.reshape(2, 1)) * (1.0 - tb1)

        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        mem_update -= 0.5 * (0.5 - tn1.reshape(2, 1))

        # Input to memory layer is now feedback connection + update 
        mem_update = mem_update.reshape(-1)
        inputs = cpu4_mem + mem_update * cpu4_mem_gain

        activity = np.array([[neuron(x) for neuron in column] for (column, x) in zip(columns, inputs)])
        output = np.sum(activity, 1) / N * mI * 1.02 + np.random.normal(0.0, 0.03)

        return output

    return closure

def cpu4_output(cpu4_mem_gain):
    def closure(inputs):
        """Updates memory based on current TB1 and TN activity.
        Can think of this as summing sinusoid of TB1 onto sinusoid of CPU4.
        cpu4[0-7] store optic flow peaking at left 45 deg
        cpu[8-15] store optic flow peaking at right 45 deg."""
        cpu4_mem, tb1, tn1, tn2 = inputs
        cpu4_mem_reshaped = cpu4_mem.reshape(2, -1)

        # Idealised setup, where we can negate the TB1 sinusoid
        # for memorising backwards motion
        mem_update = (0.5 - tn1.reshape(2, 1)) * (1.0 - tb1)

        # Both CPU4 waves must have same average
        # If we don't normalise get drift and weird steering
        mem_update -= 0.5 * (0.5 - tn1.reshape(2, 1))

        # Constant purely to visualise same as rate-based model
        cpu4_mem_reshaped += cpu4_mem_gain * mem_update
        return np.clip(cpu4_mem_reshaped.reshape(-1), 0.0, 1.0)

    return closure

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

class CXBasic(CentralComplex):
    def build_network(self) -> Network:
        return RecurrentNetwork({
            "flow": self.flow_input,
            "TL2": self.heading_input,
            "CL1": IdentityLayer("TL2"),
            "TB1": FunctionLayer(
                inputs = ["CL1", "TB1"],
                function = tb1_output,
                initial = self.tb1,
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
            "CPU4": FunctionLayer(
                inputs = ["CPU4", "TB1", "TN1", "TN2"],
                function = cpu4_output(cpu4_mem_gain=0.01),
                initial = self.cpu4,
            ),
#           "CPU4": FunctionLayer(
#               inputs = ["CPU4", "TB1", "TN1", "TN2"],
#               function = cpu4_bistable_output(cpu4_mem_gain=0.05, N=400, dI=1/300, mI=1.0),
#               initial = self.cpu4,
#            ),
            "CPU1": FunctionLayer(
                inputs = ["TB1", "CPU4"],
                function = cpu1_output,
                initial = np.zeros(N_CPU1),
            ),
            "motor": FunctionLayer(
                inputs = ["CPU1"],
                function = motor_output,
            )
        })
