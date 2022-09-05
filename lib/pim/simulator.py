"""Experiments to reproduce results from Stone 2017."""

from types import MemberDescriptorType
from typing import List, Tuple
from flask import current_app
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.interpolate import interp1d
import random

from .experiment import Experiment, ExperimentResults
from . import cx
from . import plotter
from .models import constants

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

def angular_distance(a, b):
    """Returns angular distance between a and b, between -pi and +pi"""
    x = a - b
    return (x + np.pi) % (np.pi * 2) - np.pi


def generate_path_from_waypoints(waypoints: List[Tuple[float, float]], rotation_speed=0.1, mean_acc=default_acc, drag=default_drag,
                   kappa=100.0, max_acc=default_acc, min_acc=0.0,
                   vary_speed=False):
    """Generate a path following a list of waypoints using bee_simulator physics."""
    # Start at the first waypoint
    position = np.array(waypoints[0])

    headings = [0]
    velocities = [[0, 0]]

    def get_direction(waypoint, position):
        direction = waypoint - position
        direction = direction / np.linalg.norm(direction)
        angle = np.arctan2(direction[0], direction[1])
        return direction, angle

    dt = 1.0
    for waypoint in waypoints[1:]:
        direction, start_angle = get_direction(waypoint, position)
        current_angle = start_angle

        # When have we reached a waypoint?
        # Using a tolerance distance might result in the bee simply orbiting the waypoint
        # if it is unable to turn fast enough.
        # Instead, let's considered it reached when it has passed the tangent line
        # in relation to where it started, i.e. when the angular difference is >= 90 degrees.
        while np.abs(angular_distance(start_angle, current_angle)) < np.pi / 2:
            # Let's always use mean_acc for acceleration for now
            acceleration = mean_acc
            heading_error = angular_distance(current_angle, headings[-1])
            rotation = np.minimum(rotation_speed, np.abs(heading_error)) * np.sign(heading_error)

            heading, velocity = get_next_state(
                dt = dt,
                heading = headings[-1],
                velocity = np.array(velocities[-1]),
                rotation = rotation,
                acceleration = acceleration,
                drag = drag,
            )

            position += velocity * dt
            headings.append(heading)
            velocities.append(velocity)

            direction, current_angle = get_direction(waypoint, position)

    return np.array(headings), np.array(velocities)


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

def reconstruct_path(velocities):
    position = np.zeros(2)
    positions = [position]
    for velocity in velocities:
        position = position + velocity
        positions.append(position)
    return positions

def farthest_position(path):
    return max(np.linalg.norm(path, axis=1))

def path_center_of_mass(path):
    return np.mean(path, axis=0)

def estimate_search_pattern(path, tol = 0.05):
    # Consider a (from the end) growing averaging window computing
    # the "center of mass" of the path;
    # when the CoM moves less than some tolerance level
    # for convergence, we have included the entire search pattern.
    pattern = path
    com = path_center_of_mass(pattern)

    N = np.shape(path)[0]
    n = N-1

    while n >= 0:
        candidate_pattern = path[n:]
        new_com = path_center_of_mass(candidate_pattern)
        if np.linalg.norm(com - new_com) < tol:
            com = new_com
            pattern = candidate_pattern
            break
        com = new_com
        n -= 1

    return pattern

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

        self._cached_path = None

    def report(self, decode=False):
        logger.info("plotting route")
        #fig, ax = plotter.plot_route(
        #    h = self.headings,
        #    v = self.velocities,
        #    T_outbound = self.parameters["T_outbound"],
        #    T_inbound = self.parameters["T_inbound"],
        #    plot_speed = True,
        #    plot_heading = True,
        #    quiver_color = "black",
        #    )

        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(20, 20))
        fig.suptitle(self.name.split(".")[0])

        ax1.set_aspect(1)
        self.plot_path(ax1, decode=True)
        ax1.set_xlabel("x (steps)")
        ax1.set_ylabel("y (steps)")
        #ax1.legend()

        T_total = self.parameters["T_outbound"] + self.parameters["T_inbound"]
        T, optimal_time, y1, y2, y3 = self.homing_tortuosity()

        ax2.plot(T + self.parameters["T_outbound"], 100*y1, "--", label="distance from home", color="lightblue")
        ax2.plot(T + self.parameters["T_outbound"], 100*y2, label="best distance from home", color="blue")
        ax2.plot(T + self.parameters["T_outbound"], 100*y3, label="optimal", color="orange")
        ax2.set_xlabel("time (steps)")
        ax2.set_ylabel("% of homing distance remaining")
        ax2.set_xlim(0, T_total)
        ax2.legend()

        if "memory" in self.recordings:
            ax3.plot(self.concentrations())
            ax3.set_xlabel("time (steps)")
            ax3.set_ylabel("concentration (M)")
            ax3.set_xlim(0, T_total)

            ax4.plot(self.transmittances() * 100)
            ax4.set_xlabel("time (steps)")
            ax4.set_ylabel("% transmittance")
            ax4.set_xlim(0, T_total)

            max_delta_c = np.max(self.concentrations()[:,:], axis=1) - np.min(self.concentrations()[:,:], axis=1)
            max_delta_T = np.max(self.transmittances()[:,:], axis=1) - np.min(self.transmittances()[:,:], axis=1)
            ax5.set_xlim(0, T_total)
            ax5.plot(max_delta_c, "--", label=r"$max \Delta c$")
            ax5.plot(max_delta_T, "--", label=r"$max \Delta T$")
            #ax5.plot(max_delta_T / max_delta_c, label="amplification")
            ax5.legend()

        plt.show()

    def serialize(self):
        return {
            "headings": self.headings,
            "velocities": self.velocities,
            "recordings": self.recordings,
        }

    def reconstruct_path(self):
        if self._cached_path != None:
            return self._cached_path

        positions = reconstruct_path(self.velocities)
        self._cached_path = positions
        return positions

    def closest_position(self):
        path = self.reconstruct_path()
        return min(path[self.parameters["T_outbound"]:], key = np.linalg.norm)

    def farthest_position(self):
        path = self.reconstruct_path()
        return max(path[self.parameters["T_outbound"]:], key = np.linalg.norm)

    def homing_position(self):
        return self.reconstruct_path()[self.parameters["T_outbound"]]

    def concentrations(self):
        return np.array(self.recordings["memory"]["internal"])[:,0]

    def transmittances(self):
        return np.array(self.recordings["memory"]["internal"])[:,1]

    def memory(self):
        return self.transmittances()

    def memory_headings(self):
        return [cx.fit_memory_fft(mem)[1] - np.pi for mem in self.memory()]

    def readout_headings(self):
        return [cx.fit_memory_fft(mem)[1] - np.pi for mem in self.recordings["memory"]["output"]]

    def home_headings(self):
        return [np.arctan2(x,y) + np.pi for x,y in self.reconstruct_path()]

    def angular_error(self):
        angles = self.home_headings()
        decoded_angles = self.memory_headings()
        return angular_distance(angles[1:], decoded_angles[:])

    def memory_error(self):
        path = self.reconstruct_path()
        distances = np.linalg.norm(path,axis=1)
        alpha = np.abs(self.angular_error())
        return [d*np.sin(a) if a < np.pi else d for d,a in zip(distances[1:],alpha)]

    def memory_rmse(self):
        if "memory" in self.recordings and "internal" in self.recordings["memory"]:
            error = self.memory_error()
            rmse = np.sqrt(np.mean(np.power(error,2)))
            return rmse #, angles[1:], decoded_angles
        return 0

    def closest_position_timestep(self):
        path = self.reconstruct_path()
        return min(range(self.parameters["T_outbound"], self.parameters["T_outbound"] + self.parameters["T_inbound"]), key = lambda i: np.linalg.norm(path[i]))
    
    def search_pattern(self):
        pattern = estimate_search_pattern(self.reconstruct_path()[self.parameters["T_outbound"]:], tol=0.05)
        center = path_center_of_mass(pattern)
        return center, farthest_position(pattern - center)

    def homing_tortuosity(self):
        inbound_path = self.reconstruct_path()[self.parameters["T_outbound"]:]
        T = np.arange(0, len(inbound_path))

        turning_point_distance = np.linalg.norm(inbound_path[0])
        average_homing_speed = np.mean(np.linalg.norm(self.velocities[self.parameters["T_outbound"]:], axis=1))
        distance_from_home = np.linalg.norm(inbound_path, axis=1)
        straight_line_distance = T * average_homing_speed
        optimal_distance_from_home = np.maximum(0, turning_point_distance - straight_line_distance)
        optimal_homing_time = turning_point_distance / average_homing_speed

        return T, optimal_homing_time, distance_from_home / turning_point_distance, np.minimum.accumulate(distance_from_home) / turning_point_distance, optimal_distance_from_home / turning_point_distance


    def plot_path(self, ax, search_pattern=True, decode=False):
        T_in = self.parameters["T_inbound"]
        T_out = self.parameters["T_outbound"]
        path = np.array(self.reconstruct_path())

        ax.plot(path[:T_out,0], path[:T_out,1], label="outbound")
        ax.plot(path[T_out:,0], path[T_out:,1], label="inbound")

        closest = self.closest_position()
        ax.plot([0, closest[0]], [0, closest[1]], "--", label=f"closest distance of {np.linalg.norm(self.closest_position()):.2f} at t={self.closest_position_timestep()}")

        if search_pattern:
            center, radius = self.search_pattern()
            circle = plt.Circle(center, radius, fill=False, color="r", label="estimated search pattern")
            ax.add_patch(circle)
            ax.plot(center[0], center[1], '*', label="search pattern center")

        if decode and "memory" in self.recordings:
            memory_headings = self.memory_headings()
            nth = 50
            ax.quiver(
                path[1::nth,0],
                path[1::nth,1],
                np.sin(memory_headings[::nth]),
                np.cos(memory_headings[::nth]),
                scale=10,
                width=0.003,
            )
            #readout_headings = self.readout_headings()
            #ax.quiver(
            #    path[1::nth,0],
            #    path[1::nth,1],
            #    np.sin(readout_headings[::nth]),
            #    np.cos(readout_headings[::nth]),
            #    scale=10,
            #    width=0.003,
            #    color="green"
            #)

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


