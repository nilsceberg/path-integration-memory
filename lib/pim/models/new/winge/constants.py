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
