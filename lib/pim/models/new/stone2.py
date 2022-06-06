from ..network2 import InputLayer, Layer, RecurrentNetwork, FunctionLayer, IdentityLayer
import numpy as np
import scipy.optimize


N_COLUMNS = 8  # Number of columns
x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)

# Constants
N_TL2 = 16
N_CL1 = 16
N_TB1 = 8
N_TN1 = 2
N_TN2 = 2
N_CPU4 = 16
N_CPU1A = 14
N_CPU1B = 2
N_CPU1 = N_CPU1A + N_CPU1B


def tb1_output(inputs):
    """Sinusoidal response to solar compass."""
    theta, tb1 = inputs
    return (1.0 + np.cos(np.pi + x + theta)) / 2.0

def tn1_output(flow):
    """Linearly inverse sensitive to forwards and backwards motion."""
    return np.clip((1.0 - flow) / 2.0, 0, 1)

def tn2_output(flow):
    """Linearly sensitive to forwards motion only."""
    return np.clip(flow, 0, 1)

class IntegratorLayer(Layer):
    def __init__(self, name, gain) -> None:
        super().__init__(name)
        self.gain = gain
        self.memory = 0.5 * np.ones(N_CPU4)

    def update(self, input):
        mem = self.memory.reshape(2, -1)
        mem += self.gain * input
        self.memory = np.clip(mem.reshape(-1), 0.0, 1.0)
        return self.memory


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

def motor_output(cpu1):
    random_std=0.05
    """Sum CPU1 to determine left or right turn."""
    cpu1_reshaped = cpu1.reshape(2, -1)
    motor_lr = np.sum(cpu1_reshaped, axis=1)
    # We need to add some randomness, otherwise agent infinitely overshoots
    motor = (motor_lr[1] - motor_lr[0])
    if random_std > 0.0:
        motor += np.random.normal(0, random_std)
    return motor

def cpu4_model(args, x):
    return args[0] * np.cos(np.pi + 2*x + args[1])

def tb1_model(args, x):
    return 0.5 + np.cos(np.pi + x + args[0]) * 0.5

def fit_cpu4(data):
    xs = np.linspace(0, 2*np.pi, N_CPU4, endpoint = False)
    error = lambda args: cpu4_model(args, xs) - data
    params, _ = scipy.optimize.leastsq(error, np.array([0.1, 0.01]))
    return params

def fit_tb1(data):
    xs = np.linspace(0, 2*np.pi, N_TB1, endpoint = False)
    error = lambda args: tb1_model(args, xs) - data
    params, _ = scipy.optimize.leastsq(error, np.array([0.1]))
    return params


class CentralComplex:
    def __init__(self, tn_prefs=np.pi/4.0,
                 cpu4_mem_gain=0.005):
        self.tn_prefs = tn_prefs
        self.cpu4_mem_gain = cpu4_mem_gain
        self.smoothed_flow = 0

        self.tb1 = np.zeros(N_TB1)
        self.cpu4 = 0.5 * np.ones(N_CPU4)

        self.network = RecurrentNetwork()

        self.network.add_layer(InputLayer("TL2"))
        self.network.add_layer(IdentityLayer("CL1"))

        self.network.add_layer(InputLayer("flow"))
        self.network.add_layer(FunctionLayer("TN1", tn1_output))
        self.network.add_layer(FunctionLayer("TN2", tn2_output))

        self.network.add_layer(FunctionLayer("CPU4", cpu4_output(cpu4_mem_gain=0.01)))

        #{
        #    "flow": InputLayer(),
        #    "TL2": InputLayer(),
        #    "CL1": IdentityLayer("TL2"),
        #    "TB1": FunctionLayer(["CL1", "TB1"], tb1_output, initial = self.tb1),
        #    "TN1": FunctionLayer("flow", tn1_output),
        #    "TN2": FunctionLayer("flow", tn2_output),
        #    "CPU4": FunctionLayer(["CPU4", "TB1", "TN1", "TN2"], cpu4_output(cpu4_mem_gain=0.01), initial = self.cpu4),
        #    "CPU1": FunctionLayer(["TB1", "CPU4"], cpu1_output),
        #    "motor": FunctionLayer("CPU1", motor_output)
        #})

    def update(self, dt, heading, velocity):
        flow = self.get_flow(heading, velocity)
        self.network["flow"].set_input(flow) # type: ignore
        self.network["TL2"].set_input(np.array([heading])) # type: ignore
        self.tb1, self.cpu4, motor = self.network.step(dt, ["TB1", "CPU4", "motor"])
        return motor

    def estimate_position(self):
        return fit_cpu4(self.cpu4)

    def to_cartesian(self, polar):
        return np.array([
            np.cos(polar[1] + np.pi),
            np.sin(polar[1] + np.pi),
        ]) * polar[0]

    def estimate_heading(self):
        return fit_tb1(self.tb1)

    def get_flow(self, heading, velocity, filter_steps=0):
        """Calculate optic flow depending on preference angles. [L, R]"""
        A = np.array([[np.sin(heading + self.tn_prefs),
                       np.cos(heading + self.tn_prefs)],
                      [np.sin(heading - self.tn_prefs),
                       np.cos(heading - self.tn_prefs)]])
        flow = np.dot(A, velocity)

        # If we are low-pass filtering speed signals (fading memory)
        if filter_steps > 0:
            self.smoothed_flow = (1.0 / filter_steps * flow + (1.0 -
                                  1.0 / filter_steps) * self.smoothed_flow)
            flow = self.smoothed_flow
        return flow