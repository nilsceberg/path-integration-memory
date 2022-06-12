import numpy as np
import scipy.optimize
from abc import abstractmethod

from ...network import Layer, InputLayer, Network
from ..stone.constants import *

class PhysicsLayer(Layer):
    def __init__(self):
        super().__init__()

class WingeNetwork(Network):
    def __init__(self):
        pass