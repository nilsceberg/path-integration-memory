"""Experiments to reproduce results from Stone 2017."""

from loguru import logger
from numpy import float64, ndarray
import matplotlib.pyplot as plt

from ....setup import Experiment, ExperimentResults

from . import trials
from . import rate
from . import plotter

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
            raise NotImplementedError()
            cx = cx_basic.CXBasic()
        elif cx_type == "rate":
            cx = rate.CXRate(noise = noise)
        elif cx_type == "pontin":
            cx = rate.CXRatePontin(noise = noise)
        else:
            raise RuntimeError("unknown cx type: " + cx_type)

        cx.setup()

        logger.info("generating outbound path")
        headings, velocities = trials.generate_route(
            T = T_outbound
        )

        logger.info("simulating outbound path")
        for heading, velocity in zip(headings, velocities):
            cx.update(1.0, heading, velocity)

        return StoneResults(name, self.parameters, headings, velocities, log = None, cpu4_snapshot = None)

