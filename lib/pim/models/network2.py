from abc import abstractmethod
from typing import Any, Union


class Network:
    def __init__(self):
        self.layers = {} # type: dict[str, Layer]
        self.connections = [] # type: list[Connection]
    
    def add_layer(self, layer):
        self.layers[layer.name] = layer
        return layer
    
    def add_connection(self, connection):
        self.connections.append(connection)
        return connection

    @abstractmethod
    def step(self, dt = 1):
        raise NotImplementedError

class Connection:
    @abstractmethod
    def is_post(self, layer):
        raise NotImplementedError

    @abstractmethod
    def is_pre(self, layer):
        raise NotImplementedError

    @abstractmethod
    def get_post(self, network: Network) -> "set[Layer]":
        raise NotImplementedError

    @abstractmethod
    def get_endpoint(self) -> Union[None, str]:
        return None

    @abstractmethod
    def function(self, value):
        raise NotImplementedError

class Layer:
    def __init__(self, name, initial_input = 0.0):
        self.name = name
        self.next_inputs = None # type: Any
        self.inputs = None
        self.output = None
        self.initial_input = initial_input

    def begin(self):
        self.inputs = self.next_inputs
        self.next_inputs = None

    def step(self):
        self.output = self.update(self.inputs if self.inputs is not None else self.initial_input)
        return self.output

    def input(self, value, endpoint=None):
        if endpoint:
            if self.next_inputs is None:
                self.next_inputs = {}
            elif type(self.next_inputs) != dict:
                raise RuntimeError("combined endpoint connections with non-endpoint connections")

            if endpoint not in self.next_inputs:
                self.next_inputs[endpoint] = self.initial_input
            self.next_inputs[endpoint] = self.reduce(self.next_inputs[endpoint], value)
        else:
            self.next_inputs = self.reduce(self.next_inputs if self.next_inputs is not None else self.initial_input, value)

    def reduce(self, a, b):
        return a + b

    @abstractmethod
    def update(self, input):
        """Processes input and returns output. Allowed to have side effects."""
        raise NotImplementedError

class InputLayer(Layer):
    def __init__(self, name):
        super().__init__(name)
        self.value = None

    def set(self, value):
        self.value = value

    def update(self, input):
        return self.value

class FunctionLayer(Layer):
    def __init__(self, name, f):
        super().__init__(name)
        self.f = f

    def update(self, input):
        return self.f(input)

class IdentityLayer(FunctionLayer):
    def __init__(self, name):
        super().__init__(name, lambda x: x) 

class WeightedConnection(Connection):
    def __init__(self, pre, post, weight = 1.0, endpoint = None):
        self.pre = pre
        self.post = post
        self.weight = weight
        self.endpoint = endpoint

    def get_endpoint(self):
        return self.endpoint

    def is_pre(self, layer):
        return self.pre == layer.name

    def is_post(self, layer):
        return self.post == layer.name

    def get_post(self, network):
        return { network.layers[self.post] }

    def function(self, value):
        return self.weight * value


class RecurrentNetwork(Network):
    def __init__(self):
        super().__init__()
    
    def step(self, dt = 1):
        for layer in self.layers.values():
            layer.begin()

        for layer in self.layers.values():
            output = layer.step()
            for connection in [connection for connection in self.connections if connection.is_pre(layer)]:
                for post_layer in connection.get_post(self):
                    post_layer.input(connection.function(output), connection.get_endpoint())
