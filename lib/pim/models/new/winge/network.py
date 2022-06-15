from typing import Dict, List
import numpy as np
from pim.models.new.winge.physics import Device

from ...network import Input, Layer, Network, Output, RecurrentNetwork
from .constants import *

Weights = np.ndarray

class PhysicsLayer(Layer):
    def __init__(self, inputs: Dict[Input,Weights], initial = np.array([0.0]), Vthres = 1.2):
        self.inputs = inputs
        self.N = len(initial)

        # Set up internal variables
        self.V = np.zeros((NV,self.N))
        self.B = np.zeros_like(self.V)
        self.dV= np.zeros_like(self.V)
        # I is the current through the LED
        self.I = np.zeros(self.N)
        # Power is the outputted light, in units of current
        self.P = np.zeros(self.N)
        self.ISD = np.zeros_like(self.I)
        self.Vthres = Vthres
        # Sequence of transistor threshold voltages, initialized to None
        self.Vt_vec = None 

        # Device object hold A, for example
        self.unity_coeff = 1.0
        
        super().__init__(initial)

    def begin(self, network: Network):
        for key, weights in self.inputs.items():
            self.B += weights @ network.output(key)

    def end(self):
        self.reset_B()

    def step(self, network: Network, dt: float):
        self.update_V(dt)
        self.update_I(dt)

    def output(self, network: Network) -> Output:
        return self.P * self.unity_coeff

    def assign_device(self, device: Device):
        """ Assing a Device object to all the hidden layer nodes."""
        self.device = device
        # Setup rule to scale B according to the capacitances
        self.Bscale = np.diag([1e-18/self.device.params['Cinh'],
                             1e-18/self.device.params['Cexc'],
                             0.])

    def set_unity_coeff(self, unity_coeff: float):
        self.unity_coeff = unity_coeff
        

    def get_dV(self):     
        """ Calculate the time derivative."""
        self.dV = self.device.A @ self.V + self.Bscale @ self.B
        return self.dV

    def update_V(self, dt):
        """ Using a fixed dt, update the voltages."""
        self.V += dt*self.dV
        self.V = np.clip(self.V,-self.Vthres,self.Vthres)
    
    def update_I(self, dt):
        """ Using a fixed dt, update the voltages."""
        # Get the source drain current from the transistor IV
        self.ISD = self.device.transistorIV(self.V[2],self.Vt_vec)
        self.I += dt*self.device.gammas[-1]*(self.ISD-self.I)
        # Convert current to power through efficiency function
        self.P = self.I*self.device.eta_ABC(self.I)
    
    def reset_B(self):
        """ Set elements of matrix B to 0"""
        self.B[:,:] = 0
    
    def reset_I(self):
        """ Set the currents to 0."""
        self.I[:] = 0
        
    def reset_V(self):
        """ Set all voltages to 0"""
        self.V = np.zeros((NV,self.N))
        
    def reset(self):
        """ Reset layer values."""
        self.reset_B()
        self.reset_I()
        self.reset_V()
    

class PhysicsNetwork(RecurrentNetwork):
    def __init__(self, layers: Dict[str, Layer]):
        self.dtmax = 0.1 # ns 
        self.dVmax = 0.01 # V # I loosen this a little bit now

        self.devices = {}

        super().__init__(layers)

    def step(self, dt: float):
        T = dt
        t = 0
        while t < T:
            self.preprocess_layers()
            dt = self.evolve()
            super().step_layers(dt)
            self.postprocess_layers()
            t += dt

    def get_physics_layers(self) -> Dict[str, PhysicsLayer]:
        return {key: layer for key, layer in self.layers.items() if isinstance(layer, PhysicsLayer)}

    def evolve(self) -> float:

        layers: Dict[str, PhysicsLayer] = self.get_physics_layers()
        all_dVs: List[np.ndarray] = [layer.get_dV() for layer in layers.values()]
        max_dVs: List[float] = [np.abs(dVs).max() for dVs in all_dVs]
        max_dV = max(max_dVs)

        dt = min(self.dVmax/max_dV, self.dtmax) if max_dV != 0 else self.dtmax

        return dt

    def assign_devices(self, devices: Dict[str, Device], unity_key: str) -> float:
        layers = self.get_physics_layers()
        for key, device in devices.items():
            layers[key].assign_device(device)

        unity_device = devices[unity_key]
        eta_handle = unity_device.eta_ABC
        Vthres = layers[unity_key].Vthres
        self.unity_coeff, self.Imax = unity_device.inverse_gain_coefficient(eta_handle,Vthres)

        for layer in layers.values():
            layer.set_unity_coeff(self.unity_coeff)

        return self.Imax