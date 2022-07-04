import numpy as np

N_COLUMNS = 8  # Number of columns
#x = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)

# Constants
N_TL2 = 16
N_CL1 = 16
N_TB1 = 8
N_TN1 = 2
N_TN2 = 2
N_CPU4 = 16
N_Pontine = 16
N_CPU1A = 14
N_CPU1B = 2
N_CPU1 = N_CPU1A + N_CPU1B

column_angles = np.linspace(0, 2 * np.pi, N_COLUMNS, endpoint=False)
tl2_prefs = np.tile(column_angles, 2)

# TUNED PARAMETERS:
tl2_slope_tuned = 6.8
tl2_bias_tuned = 3.0

cl1_slope_tuned = 3.0
cl1_bias_tuned = -0.5

tb1_slope_tuned = 5.0
tb1_bias_tuned = 0.0

cpu4_slope_tuned = 5.0
cpu4_bias_tuned = 2.5

cpu1_slope_tuned = 5.0
cpu1_bias_tuned = 2.5

motor_slope_tuned = 1.0
motor_bias_tuned = 3.0

cpu1_pontine_bias_tuned = -1.0
cpu1_pontine_slope_tuned = 7.5

pontine_slope_tuned = 5.0
pontine_bias_tuned = 2.5
