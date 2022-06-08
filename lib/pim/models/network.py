from abc import abstractmethod
from typing import Callable, Dict, List, NewType, Union
import numpy as np
import networkx as nx

Input = str
Output = np.ndarray
#Output = NewType("Output", np.ndarray)


class Network:
    def __init__(self, layers: Dict[str, "Layer"]):
        """Constructs the network from a dictionary on the form name -> layer. Note that
        some network types (such as the forward network) relies on the order of this dictionary,
        which is a reliable property of dictionaries since Python 3.7."""
        self.layers = layers

    def reset(self):
        for name, layer in self.layers.items():
            layer.reset()

    def step(self, dt: float):
        self.begin_layers()
        self.step_layers(dt)

    def begin_layers(self):
        """Prepare all layers for this time step."""
        for layer in self.layers.values():
            layer.begin()

    def step_layers(self, dt: float):
        """Step through each layer; override this if order is important."""
        for layer in self.layers.values():
            layer.step(self, dt)

    def simulate(self, input_layer: "InputLayer", output_layer_name: str, input_vector: List[Output], dt: float = 1.0):
        def step(network, input_layer, x):
            input_layer.set(x)
            network.step(dt)
            return network.output(output_layer_name)

        return np.array([step(self, input_layer, x) for x in input_vector])

    @abstractmethod
    def output(self, layer: str) -> Output:
        pass

    def get_graph(self) -> nx.Graph:
        G = nx.DiGraph()

        for name, layer in self.layers.items():
            layer.add_to_graph(name, G)

        return G


class ForwardNetwork(Network):
    """Network that assumes that it is acyclic and can therefore be naively recursively evaluated."""
    def output(self, layer) -> np.ndarray:
        return self.layers[layer].output(self)


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

    def reset(self):
        pass

    def begin(self):
        pass

    def step(self, network: Network, dt: float):
        """Guaranteed to only be called once per step. Allowed to have side-effects."""
        pass

    @abstractmethod
    def output(self, network: Network) -> Output:
        """Should probably be suitable for memoization."""
        pass

    def add_to_graph(self, name: str, graph: nx.Graph):
        graph.add_node(name)
        self.add_edges_to_graph(name, graph)

    def add_edges_to_graph(self, name: str, graph: nx.Graph):
        pass


class FunctionLayer(Layer):
    def __init__(self, inputs: List[Input], function: Callable[[List[Output]], Output]):
        self.inputs = inputs
        self.function = function

    def output(self, network: Network) -> Output:
        inputs = [network.output(layer) for layer in self.inputs]
        return self.function(inputs)

    def add_edges_to_graph(self, name: str, graph: nx.Graph):
        for input in self.inputs:
            graph.add_edge(input, name)


def IdentityLayer(input):
    return FunctionLayer([input], lambda x: x[0])


class InputLayer(Layer):
    def __init__(self):
        super().__init__()
        self.input = Trap()

    def set(self, input: Output):
        """The input parameter is an Output! Think about that for a second."""
        self.input = input

    def output(self, network: Network) -> np.ndarray:
        return Trap.check(self.input)
