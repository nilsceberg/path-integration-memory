import json
from typing import Union
import numpy as np
from loguru import logger

class Device:
    def __init__(self, path: str):
        file = open(path, "r")
        input_dict = json.load(file)
        self.params = {key: items['value'] for key, items in input_dict.items()}
        self.parameter_units =  {key: items['unit'] for key, items in input_dict.items()}

        self.kT = 0.02585

        self.linslope = self.params['Cgate'] * self.params['vt'] * 1e9 # nA/V
        self.gammas = self.calc_gammas()
        self.A = self.calc_A(self.gammas)
    
    def set_parameter(self, key: str, value: Union[int, float]):
        self.params[key] = value
        self.gammas = self.calc_gammas()
        self.A = self.calc_A(self.gammas)

    def calc_gammas(self):
        # Sum the memory and gate capacitance, convert Lgate in um to cm
        Cmem = self.params['Cstore'] + self.params['Cgate']*self.params['Lgate']*1e-4 
        # System frequencies
        g11 = 1e-9/self.params['Cinh']/self.params['Rinh'] # ns^-1 # GHz
        g22 = 1e-9/self.params['Cexc']/self.params['Rexc'] # ns^-1 # GHz
        g13 = 1e-9/Cmem/self.params['Rinh'] # ns^-1 # GHz
        g23 = 1e-9/Cmem/self.params['Rexc'] # ns^-1 # GHz
        g33 = 1e-9/Cmem/self.params['Rstore'] # ns^-1 # GHz
        gled = 1e-9/self.params['CLED']/self.params['RLED'] # ns^-1 # GHz

        return np.array([g11,g22,g13,g23,g33,gled])

    def calc_A(self, gammas: np.ndarray):
        g11, g22, g13, g23, g33, gled = gammas
        gsum = g13 + g23 + g33
        A = np.array([[-g11, 0, g11],
                      [0, -g22, g22],
                      [g13, g23, -gsum]])
                    
        return A

    def calc_tau_gate(self):
        # Sum the memory and gate capacitance, convert Lg in um to cm
        Cmem = self.params['Cstore'] + self.params['Cgate']*self.params['Lg']*1e-4 
        # System frequencies
        # g13 = 1e-9/Cmem/self.params['Rinh'] # ns^-1 # GHz
        # g23 = 1e-9/Cmem/self.params['Rexc'] # ns^-1 # GHz
        g33 = 1e-9/Cmem/self.params['Rstore'] # ns^-1 # GHz
        # Modifying this as it is g33 that matter for a system in equilibrium
        return g33**-1 