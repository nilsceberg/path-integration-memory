from loguru import logger
from numpy import float64, ndarray
import matplotlib.pyplot as plt

from ...setup import Experiment, ExperimentResults

from . import trials
from . import cx_rate
from . import plotter

class StoneResults(ExperimentResults):
    def __init__(self, name: str, parameters: dict, route, log, cpu4_snapshot) -> None:
        super().__init__(name, parameters)
        self.route = route
        self.log = log
        self.cpu4_snapshot = cpu4_snapshot

    def report(self):
        logger.info("plotting route")
        h, v = self.route
        fig, ax = plotter.plot_route(
            h = h,
            v = v,
            T_outbound = self.parameters["T_outbound"],
            T_inbound = self.parameters["T_inbound"],
            plot_speed = True,
            plot_heading = True,
            quiver_color = "black",
            )

        plt.show()

    def serialize(self):
        return {
            #"route": self.route,

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

        logger.info(f"generating outbound route")
        h, v = trials.generate_route(T = T_outbound, vary_speed = True)

        logger.info("initializing central complex")
        cx = cx_rate.CXRatePontin(noise = noise)

        logger.info("running trial")
        h, v, log, cpu4_snapshot = trials.run_trial(
            logging = True,
            T_outbound = T_outbound,
            T_inbound = T_inbound,
            noise = self.parameters["noise"],
            cx = cx,
            route = (h[:T_outbound], v[:T_outbound])
        )

        return StoneResults(name, self.parameters, (h, v), log, cpu4_snapshot)
