import numpy as np

# Internal voltage degrees of freedom
NV = 3

# Tuned parameters from Stone
tl2_slope_tuned = 6.8
tl2_bias_tuned = 3.0   
cl1_slope_tuned = 3.0
cl1_bias_tuned = -0.5

# Preference angles? for CL1
TL_angles = np.tile(np.arange(0,8),2)*np.pi/4

# Constants
N_TL2 = 16
N_CL1 = 16
N_TB1 = 8
N_TN1 = 2
N_TN2 = 2
N_CPU4 = 16
N_Pontin = 16
N_CPU1A = 14
N_CPU1B = 2
N_CPU1 = N_CPU1A + N_CPU1B