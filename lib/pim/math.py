from typing import Union
import numpy as np
from scipy.integrate import solve_ivp

# If this is set to an integer N, step_ode will return N solution points;
# useful for creating figures.
num_solutions: Union[int, None] = None

# Use scipy solver instead of naive Euler method;
# should not be necessary but may be useful for validation.
slow_solver = False

# Euler method constant step size
step_size = 1

# Solves a time-invariant ODE; returns yf, T, Y
def step_ode(dydt, y0, Dt):
    T = None if num_solutions is None else np.linspace(0, Dt, num_solutions)
    if slow_solver:
        solution = solve_ivp(dydt, (0, Dt), y0, t_eval=T)
        return solution.y[:,-1], solution.t, solution.y
    else:
        if T is None:
            # Don't keep track of (or return) T or Y, for performance:
            dt = step_size
            return y0 + dydt(0, y0) * dt, None, None

        # Otherwise, save internal solutions:
        dt = T[1]
        Y = [y0]
        for t in T[1:]:
            Y.append(Y[-1] + dydt(t, Y[-1]) * dt)
        return Y[-1], T, np.transpose(Y)
