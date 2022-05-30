from abc import abstractmethod
from typing import Any, Callable
import numpy as np

class Layer:
    """Base class for CX model layers.
    Layers are allowed to have internal state."""

    def __init__(self):
        self._output = np.array([])

    @abstractmethod
    def step(self, dt: float):
        """Runs one timestep. Note that this method is allowed to have side effects."""
        pass

    @abstractmethod
    def output(self) -> np.ndarray:
        return self._output


class InputLayer(Layer):
    def __init__(self):
        pass

    def input(self, activity: np.ndarray):
        self._output = activity


class InputOutputLayer(Layer):
    @abstractmethod
    def input(self, input: Layer):
        pass


class FunctionLayer(InputOutputLayer):
    def __init__(self, function: Callable[[np.ndarray], np.ndarray]):
        self.function = function
        self._input = np.array([])

    def step(self, dt: float):
        self._output = self.function(self._input)

    def input(self, layer: Layer):
        self._input = layer.output()

    def output(self):
        return self._output


class IdentityLayer(FunctionLayer):
    def __init__(self):
        super(IdentityLayer, self).__init__(lambda x: x)


# Temporary home for reimplementation of Stone:
N_COLUMNS = 8  # Number of columns
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)

def tb1_output(theta):
    """Sinusoidal response to solar compass."""
    return (1.0 + np.cos(np.pi + x + theta)) / 2.0

def tn1_output(flow):
    """Linearly inverse sensitive to forwards and backwards motion."""
    return np.clip((1.0 - flow) / 2.0, 0, 1)

def tn2_output(flow):
    """Linearly sensitive to forwards motion only."""
    return np.clip(flow, 0, 1)

class StoneCX:
    def __init__(self):
        self.input = InputLayer()
        self.tl2 = IdentityLayer()
        self.cl1 = IdentityLayer()
        self.tb1 = FunctionLayer(tb1_output)
        self.tn1 = FunctionLayer(tn1_output)
        self.tn2 = FunctionLayer(tn2_output)
        self.cpu4 = CPU4Layer()
        self.cpu1 = CPU1Layer()
        self.motor = MotorLayer()

    def step(self, dt, theta, flow):
        self.input.input(np.array([theta]))
        self.input.step(dt)

        self.tl2.input(theta)
        self.tl2.step(dt)

        self.cl1.input(self.tl2)
        self.cl1.step(dt)

        self.tb1.input(self.cl1)
        self.tb1.input(self.tb1)
        self.tb1.step(dt)

        self.tn1.input(flow)
        self.tn1.step(dt)

        self.tn2.input(flow)
        self.tn2.step(dt)

        self.cpu4.input(self.tb1)
        self.cpu4.input(self.tn1)
        self.cpu4.input(self.tn2)
        self.cpu4.step(dt)

        self.cpu1.input(self.tb1)
        self.cpu1.input(self.cpu4)
        self.cpu1.step(dt)

        self.motor.input(self.cpu1)
        self.motor.step(dt)

        motor = self.motor.output()

