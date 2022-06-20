"""Experiments to reproduce results from Stone 2017."""

from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from .cx import PhysicsCX
from . import physics
from ....setup import Experiment, ExperimentResults

from pim.models.new.stone import trials
from pim.models.new.stone import plotter
from pim.models.new.stone import bee_simulator

class WingeResults(ExperimentResults):
    def __init__(self, name: str, parameters: dict, headings, velocities, log, cpu4_snapshot) -> None:
        super().__init__(name, parameters)
        self.headings = headings
        self.velocities = velocities
        self.log = log
        self.cpu4_snapshot = cpu4_snapshot

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
            "headings": self.headings.tolist(),
            "velocities": self.headings.tolist(),
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

class WingeExperiment(Experiment):
    def __init__(self, parameters: dict) -> None:
        super().__init__()
        self.parameters = parameters

        noise = self.parameters["noise"]
        cx_type = self.parameters["cx"]

        if cx_type == "physics":
            cx = PhysicsCX(update_m=0.001, noise=noise)
        else:
            raise RuntimeError("unknown cx type: " + cx_type)

        self.cx = cx
        self.cx.setup()

        path = '../lib/pim/models/new/winge/parameters/device_parameters.json'
        devices = {}
        devices['TB1']=physics.Device(path)
        devices['CPU4']=physics.Device(path)
        #devices['CPU4'].set_parameter('Cstore',7e-16) # Original is 0.07 10^-15
        devices['CPU4'].set_parameter('Rstore',2e11) # Original 2e6
        devices['CPU1a']=physics.Device(path)
        devices['CPU1b']=physics.Device(path)
        devices['Pontine']=physics.Device(path)

        cx.assign_devices(devices, unity_key='TB1')

    def run(self, name: str) -> ExperimentResults:
        # extract some parameters
        T_outbound = self.parameters["T_outbound"]
        T_inbound = self.parameters["T_inbound"]
        noise = self.parameters["noise"]
        cx_type = self.parameters["cx"]
        time_subdivision = self.parameters["time_subdivision"] if "time_subdivision" in self.parameters else 1

        logger.info(f"generating outbound route")
        headings, velocities = trials.generate_route(T = T_outbound, vary_speed = True)

        logger.info("initializing central complex")

        headings = np.zeros(T_outbound + T_inbound)
        velocities = np.zeros((T_outbound + T_inbound, 2))

        logger.info("generating outbound path")

        headings[0:T_outbound], velocities[0:T_outbound, :] = trials.generate_route(
            T = T_outbound,
            vary_speed = True
        )

        logger.info("simulating outbound path")
        dt = 1.0 / time_subdivision
        for heading, velocity in zip(headings[0:T_outbound], velocities[0:T_outbound, :]):
            for ts in range(time_subdivision):
                self.cx.update(dt, heading, velocity)

        for t in range(T_outbound, T_outbound + T_inbound):
            heading = headings[t-1]
            velocity = velocities[t-1,:]

            for ts in range(time_subdivision):
                motor = self.cx.update(dt, heading, velocity)
                rotation = motor

                heading, velocity = bee_simulator.get_next_state(
                    dt=dt,
                    velocity=velocity,
                    heading=heading,
                    acceleration=0.1,
                    drag=trials.default_drag,
                    rotation=rotation,
                )

            headings[t], velocities[t,:] = heading, velocity

        return WingeResults(name, self.parameters, headings, velocities, log = None, cpu4_snapshot = None)

