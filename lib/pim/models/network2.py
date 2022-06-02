from abc import abstractmethod


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
    def function(self, value):
        raise NotImplementedError

class Layer:
    def __init__(self, name):
        self.name = name
        self.last_output = None

    def step(self, input):
        self.last_output = self.update(input)
        return self.last_output

    @abstractmethod
    def update(self, input):
        """Processes input and returns output. Allowed to have side effects."""
        raise NotImplementedError

class InputLayer(Layer):
    def __init__(self, name):
        self.name = name
        self.value = None

    def set(self, value):
        self.value = value

    def update(self, input):
        return self.value

class WeightedConnection(Connection):
    def __init__(self, pre, post, weight = 1.0):
        self.pre = pre
        self.post = post
        self.weight = weight

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

        self.inputs = {}
    
    def step(self, dt = 1):
        next_inputs = {}

        for layer in self.layers.values():
            output = layer.step(self.inputs[layer.name] if layer.name in self.inputs else 0.0)
            for connection in [connection for connection in self.connections if connection.is_pre(layer)]:
                for post_layer in connection.get_post(self):
                    if post_layer.name not in next_inputs:
                        next_inputs[post_layer.name] = 0
                    next_inputs[post_layer.name] += connection.function(output)

        self.inputs = next_inputs
