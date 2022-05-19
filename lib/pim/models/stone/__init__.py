from loguru import logger
from numpy import float64, ndarray
import matplotlib.pyplot as plt

from ...setup import Experiment, ExperimentResults

from . import trials
from . import cx_basic
from . import cx_rate
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
            cx = cx_basic.CXBasic()
        elif cx_type == "rate":
            cx = cx_rate.CXRate(noise = noise)
        elif cx_type == "pontin":
            cx = cx_rate.CXRatePontin(noise = noise)
        else:
            raise RuntimeError("unknown cx type: " + cx_type)

        logger.info("running trial")
        headings, velocities, log, cpu4_snapshot = trials.run_trial(
            logging = True,
            T_outbound = T_outbound,
            T_inbound = T_inbound,
            noise = self.parameters["noise"],
            cx = cx,
            route = (headings[:T_outbound], velocities[:T_outbound])
        )

        return StoneResults(name, self.parameters, headings, velocities, log, cpu4_snapshot)
