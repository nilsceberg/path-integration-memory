from abc import abstractmethod
from typing import Callable, Dict, List, Union
import numpy as np


class Network:
    def __init__(self, layers: Dict[str, "Layer"]):
        self.layers = layers

    def reset(self):
        for name, layer in self.layers.items():
            layer.reset()

    @abstractmethod
    def step(self, dt: float, query: List[str]) -> List[np.ndarray]:
        pass

    def __getitem__(self, name: str) -> "Layer":
        return self.layers[name]


class ForwardNetwork(Network):
    def step(self, dt: float, query: List[str]) -> List[np.ndarray]:
        for name, layer in self.layers.items():
            layer.begin()

        return [self.layers[layer].output(self, dt) for layer in query]


class Trap():
    @staticmethod
    def check(value):
        if isinstance(value, Trap):
            raise RuntimeError("dependency on uninitialized layer (cyclic?)")
        else:
            return value


class Layer:
    def __init__(self):
        self.reset()

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def begin(self):
        pass

    @abstractmethod
    def output(self, network: Network, dt: float) -> np.ndarray:
        pass


class MemoizedLayer(Layer):
    def __init__(self, initial: Union[Trap, np.ndarray] = Trap()):
        self.initial = initial
        super().__init__()

    def reset(self):
        self.previous_activity = None
        self.activity = self.initial

    def begin(self):
        self.previous_activity = self.activity
        self.activity = None

    @abstractmethod
    def evaluate(self, network: Network, dt: float) -> np.ndarray:
        pass

    def output(self, network: Network, dt: float) -> np.ndarray:
        if self.activity is not None:
            return Trap.check(self.activity)
        else:
            # Ensure that any recurrences see the last value of this layer.
            self.activity = self.previous_activity
            self.activity = self.evaluate(network, dt)
            return self.activity


class FunctionLayer(MemoizedLayer):
    def __init__(self, inputs: Union[str, List[str]], function, initial: Union[Trap, np.ndarray] = Trap()):
        super().__init__(initial)
        self.function = function
        self.inputs = inputs

    def evaluate(self, network: Network, dt: float) -> np.ndarray:
        if isinstance(self.inputs, str):
            inputs = network[self.inputs].output(network, dt)
        else:
            inputs = [network[layer].output(network, dt) for layer in self.inputs]
        return self.function(inputs)


def IdentityLayer(input):
    return FunctionLayer(input, lambda x: x)


class InputLayer(Layer):
    def __init__(self):
        super().__init__()
        self.input = Trap()

    def set_input(self, input: np.ndarray):
        self.input = input

    def output(self, network: Network, dt: float) -> np.ndarray:
        return Trap.check(self.input)
