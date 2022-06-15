from typing import Dict

from pim.models.new.stone.cx import CentralComplex
from pim.models.new.stone.rate import gen_tb_tb_weights, noisify_weights, noisy_sigmoid
from pim.models.network import FunctionLayer, Network

from .physics import Device
from .constants import *
from .network import PhysicsLayer, PhysicsNetwork

class PhysicsCX(CentralComplex):
    def build_network(self) -> Network:
        return PhysicsNetwork({
            "flow": self.flow_input,
            "heading": self.heading_input,
            "TL2": FunctionLayer(
                inputs = ["heading"],
                function = self.tl2_output,
                initial = np.zeros(N_TL2),
            ),
            "CL1": FunctionLayer(
                inputs = ["TL2"],
                function = self.cl1_output,
                initial = np.zeros(N_CL1),
            ),
            "TN2": FunctionLayer(
                inputs = ["flow"],
                function = self.tn2_output,
                initial = np.zeros(N_TN2),
            ),
            "TB1": PhysicsLayer(
                inputs = {"CL1" : self.W_CL1_TB1, "TB1" : self.W_TB1_TB1},
                initial = self.tb1,
            ),
            "CPU4": PhysicsLayer(
                inputs = {"TN2" : self.W_TN2_CPU4, "TB1" : self.W_TB1_CPU4},
                initial = self.cpu4,
            ),
            "Pontin": PhysicsLayer(
                inputs = {"CPU4": self.W_CPU4_Pontin},
                initial = np.zeros(N_Pontin)
            ),
            "CPU1a": PhysicsLayer(
                inputs = {"TB1": self.W_TB1_CPU1a, "CPU4" : self.W_CPU4_CPU1a, "Pontin": self.W_Pontin_CPU1a},
                initial = np.zeros(N_CPU1A)
            ),
            "CPU1b": PhysicsLayer(
                inputs = {"TB1": self.W_TB1_CPU1b, "CPU4" : self.W_CPU4_CPU1b,  "Pontin": self.W_Pontin_CPU1b},
                initial = np.zeros(N_CPU1B)
            ),
            "motor": FunctionLayer(
                inputs = ["CPU1a", "CPU1b"],
                function = self.motor_output,
            )
        })

    def __init__(self, tb1_c = 0.33, mem_update_h = 0.0025, cpu4_cpu1_m = 0.5, update_m = 0.0005, 
    pon_cpu1_m = 0.5, tb1_cpu1_m = 1.0, weight_noise = 0.0, Vt_noise = 0.0, noise = 0.1, inputscaling=0.9):
        if weight_noise > 0.0 :
            noisy_weights=True
        else :
            noisy_weights=False

        self.noise = noise
        self.update_m = update_m
        self.inputscaling = inputscaling

        self.W_CL1_TB1 = np.tile(np.diag([1.0]*N_TB1),2)
        self.W_CL1_TB1 *= (1.-tb1_c) # scaling factor to weigh TB1 and CL1 input to TB1
        if noisy_weights : W = noisify_weights(self.W_CL1_TB1,weight_noise)

        self.W_TB1_TB1 = gen_tb_tb_weights(weight=tb1_c)
        if noisy_weights : self.W_TB1_TB1 = noisify_weights(self.W_TB1_TB1,weight_noise)
        
        self.W_TB1_CPU4 = np.tile(np.diag([1.0]*N_TB1),(2,1)) 
        if noisy_weights : self.W_TB1_CPU4 = noisify_weights(self.W_TB1_CPU4,weight_noise)
        self.W_TB1_CPU4 *= mem_update_h


        self.W_TB1_CPU1a = np.tile(np.diag([1.0]*N_TB1),(2,1))
        if noisy_weights : self.W_TB1_CPU1a = noisify_weights(self.W_TB1_CPU4,weight_noise)
        self.W_TB1_CPU4 *= self.W_TB1_CPU4[1:-1]*tb1_cpu1_m
        
        self.W_TB1_CPU1b = np.zeros((2,N_TB1))
        self.W_TB1_CPU1b[0,-1] = 1.0
        self.W_TB1_CPU1b[1,0]  = 1.0
        if noisy_weights : self.W_TB1_CPU1b = noisify_weights(self.W_TB1_CPU1b,weight_noise)
        self.W_TB1_CPU1b *= tb1_cpu1_m
        
        self.W_TN2_CPU4 = np.concatenate((np.tile(np.array([1,0]),(N_CPU4//2,1)),np.tile(np.array([0,1]),(N_CPU4//2,1)))) 
        if noisy_weights : self.W_TN2_CPU4 = noisify_weights(self.W_TN2_CPU4,weight_noise)
        self.W_TN2_CPU4 *= mem_update_h
        
        self.W_CPU4_CPU1a =  np.array([[0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], #2
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                                    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], #15
                                    ])
        if noisy_weights : self.W_CPU4_CPU1a = noisify_weights(self.W_CPU4_CPU1a,weight_noise)
        self.W_CPU4_CPU1a *= self.W_CPU4_CPU1a*cpu4_cpu1_m
        
        self.W_CPU4_CPU1b = np.zeros((2,N_CPU4))
        self.W_CPU4_CPU1b[0,0]=1.0
        self.W_CPU4_CPU1b[-1,-1]=1.0
        if noisy_weights : self.W_CPU4_CPU1b = noisify_weights(self.W_CPU4_CPU1b,weight_noise)
        self.W_CPU4_CPU1a *= cpu4_cpu1_m
        
        self.W_CPU4_Pontin = np.diag([1.0]*N_CPU4)
        if noisy_weights : self.W_CPU4_Pontin = noisify_weights(self.W_CPU4_Pontin,weight_noise)
        
        self.W_Pontin_CPU1a = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], #2
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], #15])
                                    ],dtype=float)
        if noisy_weights : self.W_Pontin_CPU1a = noisify_weights(self.W_Pontin_CPU1a,weight_noise)
        self.W_Pontin_CPU1a *= pon_cpu1_m
        
        self.W_Pontin_CPU1b = np.zeros((2,N_Pontin))
        self.W_Pontin_CPU1b[0,4] = 1.0
        self.W_Pontin_CPU1b[1,11]= 1.0
        if noisy_weights : self.W_Pontin_CPU1b = noisify_weights(self.W_Pontin_CPU1b,weight_noise)
        self.W_Pontin_CPU1b *= pon_cpu1_m

        super().__init__()

    def assign_devices(self, devices: Dict[str, Device], unity_key: str):
        self.Imax = self.network.assign_devices(devices, unity_key)

    def tn2_output(self, inputs) :
        flow = inputs
        # Add noise
        flow += np.random.normal(scale=self.noise, size=len(flow))
        # scale by the standard current factor
        return np.clip(flow,0,1)*self.Imax*self.inputscaling

    def tl2_output(self,inputs):
        heading = inputs
        output = np.cos(heading - TL_angles)
        return noisy_sigmoid(output, tl2_slope_tuned, tl2_bias_tuned, noise=self.noise)
          
    def cl1_output(self,inputs):
        tl2 = inputs
        sig = noisy_sigmoid(-tl2, cl1_slope_tuned, cl1_bias_tuned, noise=self.noise)
        # scale by the standard current factor
        return sig*self.Imax*self.inputscaling

    def motor_output(self, inputs):
        """outputs a scalar where sign determines left or right turn."""
        cpu1a, cpu1b = inputs

        r_right = sum(cpu1a[:7]) + cpu1b[1]
        r_left  = sum(cpu1a[7:]) + cpu1b[0]
        return self.update_m*(r_right-r_left)