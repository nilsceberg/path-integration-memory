"""Experiments to reproduce results from Stone 2017."""

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.interpolate import interp1d
import random

from .experiment import Experiment, ExperimentResults
from . import cx
from . import plotter

default_acc = 0.15  # A good value because keeps speed under 1
default_drag = 0.15

def generate_random_route(T=1500, mean_acc=default_acc, drag=default_drag,
                   kappa=100.0, max_acc=default_acc, min_acc=0.0,
                   vary_speed=False, min_homing_distance=0.0):
    """Generate a random outbound route using bee_simulator physics.
    The rotations are drawn randomly from a von mises distribution and smoothed
    to ensure the agent makes more natural turns."""
    # Generate random turns
    mu = 0.0
    vm = np.random.vonmises(mu, kappa, T)
    rotation = lfilter([1.0], [1, -0.4], vm)
    rotation[0] = 0.0

    position = np.zeros(2)

    # Randomly sample some points within acceptable acceleration and
    # interpolate to create smoothly varying speed.
    if vary_speed:
        if T > 200:
            num_key_speeds = T // 50
        else:
            num_key_speeds = 4
        x = np.linspace(0, 1, num_key_speeds)
        y = np.random.random(num_key_speeds) * (max_acc - min_acc) + min_acc
        f = interp1d(x, y, kind='cubic')
        xnew = np.linspace(0, 1, T, endpoint=True)
        acceleration = f(xnew)
    else:
        acceleration = mean_acc * np.ones(T)

    # Get headings and velocity for each step
    headings = np.zeros(T)
    velocity = np.zeros([T, 2])

    for t in range(1, T):
        headings[t], velocity[t, :] = get_next_state(
            dt=1.0,
            heading=headings[t-1], velocity=velocity[t-1, :],
            rotation=rotation[t], acceleration=acceleration[t], drag=drag)
        position += velocity[t, :]

    if np.linalg.norm(position) < min_homing_distance:
        return generate_random_route(T, mean_acc, drag, kappa, max_acc, min_acc, vary_speed, min_homing_distance)

    return headings, velocity


def generate_path_from_parameters(path, speed = 0.35):
    steps = int(np.sum(np.array(path)[:,0])) # first column, heading duration
    headings = np.zeros(steps)
    velocities = np.zeros((steps, 2))

    t = 0
    for duration, heading in path:
        for i in range(int(duration)):
            headings[t] = heading
            # This order is so weird...
            velocities[t,:] = speed * np.array([
                np.sin(heading),
                np.cos(heading),
            ])
            t += 1

    return headings, velocities


def rotate(dt, theta, r):
    """Return new heading after a rotation around Z axis."""
    return (theta + r * dt + np.pi) % (2.0 * np.pi) - np.pi


def thrust(dt, theta, acceleration):
    """Thrust vector from current heading and acceleration

    theta: clockwise radians around z-axis, where 0 is forward
    acceleration: float where max speed is ....?!?
    """
    return np.array([np.sin(theta), np.cos(theta)]) * acceleration * dt


def get_next_state(dt, heading, velocity, rotation, acceleration, drag=0.5):
    """Get new heading and velocity, based on relative rotation and
    acceleration and linear drag."""
    theta = rotate(dt, heading, rotation)
    v = velocity + thrust(dt, theta, acceleration)
    v *= (1.0 - drag)**dt
    return theta, v


def load_results(data):
    return SimulationResults(
        name = data["name"],
        config_id = data["config_id"],
        parameters = data["parameters"],
        headings = data["results"]["headings"],
        velocities = data["results"]["velocities"],
        recordings = data["results"]["recordings"],
    )


class SimulationResults(ExperimentResults):
    def __init__(self, name: str, config_id: str, parameters: dict, headings, velocities, recordings: dict) -> None:
        super().__init__(name, config_id, parameters)

        # Make sure these fields are np arrays (shouldn't matter if they already are):
        self.headings = np.array(headings)
        self.velocities = np.array(velocities)

        self.recordings = recordings

    def report(self):
        logger.info("plotting route")
        fig, ax = plotter.plot_route(
            h = self.headings,
            v = self.velocities,
            T_outbound = self.parameters["T_outbound"],
            T_inbound = self.parameters["T_inbound"],
            plot_speed = True,
            plot_heading = True,
            quiver_color = "black",
            )

        plt.show()

    def serialize(self):
        return {
            "headings": self.headings,
            "velocities": self.velocities,
            "recordings": self.recordings,
        }

    def reconstruct_path(self):
        position = np.zeros(2)
        positions = [position]
        for velocity in self.velocities:
            position = position + velocity
            positions.append(position)
        return positions

    def closest_position(self):
        path = self.reconstruct_path()
        return min(path[self.parameters["T_outbound"]:], key = np.linalg.norm)

    def closest_position_timestep(self):
        path = self.reconstruct_path()
        return min(range(self.parameters["T_outbound"], self.parameters["T_outbound"] + self.parameters["T_inbound"]), key = lambda i: np.linalg.norm(path[i]))

    def plot_path(self, ax):
        T_in = self.parameters["T_inbound"]
        T_out = self.parameters["T_outbound"]
        path = np.array(self.reconstruct_path())

        ax.plot(path[:T_out,0], path[:T_out,1])
        ax.plot(path[T_out:,0], path[T_out:,1])

        closest = self.closest_position()
        ax.plot([0, closest[0]], [0, closest[1]], "--", label=f"closest distance of {np.linalg.norm(self.closest_position()):.2f} at t={self.closest_position_timestep()}")

        ax.plot(0, 0, "*")


class SimulationExperiment(Experiment):
    def __init__(self, parameters: dict) -> None:
        super().__init__()

        self.parameters = parameters
        
        self.seed = parameters.get("seed", random.randint(0, 2**32-1))
        self.layers_to_record = self.parameters["record"] if "record" in self.parameters else []
        self.recordings = {layer: { "output": [], "internal": [] } for layer in self.layers_to_record}

        logger.info("recording {}", self.layers_to_record)

    def _record(self):
        for layer in self.layers_to_record:
            output = self.cx.network.output(layer)
            self.recordings[layer]["output"].append(output)
            logger.trace("layer {} output: {}", layer, output)
            internal = self.cx.network.layers[layer].internal()
            if internal is not None:
                self.recordings[layer]["internal"].append(internal)
                logger.trace("layer {} internal: {}", layer, internal)


    def run(self, name: str, config_id: str) -> ExperimentResults:
        logger.info("seeding: {}", self.seed)
        np.random.seed(self.seed)

        # extract some parameters
        time_subdivision = self.parameters["time_subdivision"] if "time_subdivision" in self.parameters else 1

        if self.parameters["cx"]["type"] == "random":
            # Useful for control
            T_outbound = self.parameters.get("T_outbound", 0)
            T_inbound = self.parameters["T_inbound"]
            headings, velocities = generate_random_route(
                T = T_outbound + T_inbound,
                vary_speed = True,
            )
        else:
            self.cx = cx.build_from_json(self.parameters["cx"])

            logger.info("initializing central complex")

            logger.info("generating outbound path")

            path = self.parameters.get("path", "random")
            if path == "random": 
                T_outbound = self.parameters["T_outbound"]
                T_inbound = self.parameters["T_inbound"]
                headings = np.zeros(T_outbound + T_inbound)
                velocities = np.zeros((T_outbound + T_inbound, 2))

                headings[0:T_outbound], velocities[0:T_outbound, :] = generate_random_route(
                    T = T_outbound,
                    vary_speed = True,
                    min_homing_distance = self.parameters.get("min_homing_distance", 0),
                )
            elif isinstance(path, list) or isinstance(path, np.ndarray):
                T_inbound = self.parameters["T_inbound"]
                headings, velocities = generate_path_from_parameters(
                    path = path,
                )
                T_outbound = len(headings)
                self.parameters["T_outbound"] = T_outbound
                headings = np.hstack([headings, np.zeros(T_inbound)])
                velocities = np.vstack([velocities, np.zeros((T_inbound, 2))])
            else:
                raise RuntimeError(f"unkonown path type {str(type(path))}")

            logger.info("simulating outbound path")
            dt = 1.0 / time_subdivision
            for heading, velocity in zip(headings[0:T_outbound], velocities[0:T_outbound, :]):
                for ts in range(time_subdivision):
                    self.cx.update(dt, heading, velocity)
                self._record()

            for t in range(T_outbound, T_outbound + T_inbound):
                heading = headings[t-1]
                velocity = velocities[t-1,:]

                for ts in range(time_subdivision):
                    motor = self.cx.update(dt, heading, velocity)
                    rotation = motor * self.parameters.get("motor_factor", 1.0)

                    heading, velocity = get_next_state(
                        dt=dt,
                        velocity=velocity,
                        heading=heading,
                        acceleration=0.1,
                        drag=default_drag,
                        rotation=rotation,
                    )

                headings[t], velocities[t,:] = heading, velocity
                self._record()

        return SimulationResults(name, config_id, self.parameters, headings, velocities, recordings = self.recordings)


