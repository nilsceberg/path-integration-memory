"""Experiments to reproduce results from Stone 2017."""

from re import A
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt

from ....setup import Experiment, ExperimentResults

from . import trials
from . import rate
from . import basic
from . import plotter
from . import bee_simulator

class StoneResults(ExperimentResults):
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

            # annoying to serialize:
            #"log": self.log,
            #"cpu4_snapshot": self.cpu4_snapshot,
        }

class StoneExperiment(Experiment):
    def __init__(self, parameters: dict) -> None:
        super().__init__()
        self.parameters = parameters

    def run(self, name: str) -> ExperimentResults:
        # extract some parameters
        T_outbound = self.parameters["T_outbound"]
        T_inbound = self.parameters["T_inbound"]
        noise = self.parameters["noise"]
        cx_type = self.parameters["cx"]

        logger.info(f"generating outbound route")
        headings, velocities = trials.generate_route(T = T_outbound, vary_speed = True)

        logger.info("initializing central complex")

        if cx_type == "basic":
            cx = basic.CXBasic()
        elif cx_type == "rate":
            cx = rate.CXRate(noise = noise)
        elif cx_type == "pontin":
            cx = rate.CXRatePontin(noise = noise)
        else:
            raise RuntimeError("unknown cx type: " + cx_type)

        cx.setup()

        headings = np.zeros(T_outbound + T_inbound)
        velocities = np.zeros((T_outbound + T_inbound, 2))

        logger.info("generating outbound path")

        headings[0:T_outbound], velocities[0:T_outbound, :] = trials.generate_route(
            T = T_outbound,
            vary_speed = True
        )

        logger.info("simulating outbound path")
        for heading, velocity in zip(headings[0:T_outbound], velocities[0:T_outbound, :]):
            cx.update(1.0, heading, velocity)

        for t in range(T_outbound, T_outbound + T_inbound):
            heading = headings[t-1]
            velocity = velocities[t-1,:]

            motor = cx.update(1.0, heading, velocity)
            rotation = motor

            headings[t], velocities[t,:] = bee_simulator.get_next_state(
                velocity=velocity,
                heading=heading,
                acceleration=0.1,
                drag=trials.default_drag,
                rotation=rotation,
            )

        return StoneResults(name, self.parameters, headings, velocities, log = None, cpu4_snapshot = None)

