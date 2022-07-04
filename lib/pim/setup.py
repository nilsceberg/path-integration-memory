from abc import abstractmethod
from typing import Callable, Tuple, Union
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from loguru import logger

import sys
import json

from .simulator import SimulationExperiment
from .experiment import Experiment

experiment_log_colors = ["red", "magenta", "yellow", "red", "green", "blue"]
def run(setup_name: str, setup_config: dict, models: Callable[[str, dict], Union[Experiment, None]], report = True, save = False):
    threads = setup_config["threads"] if "threads" in setup_config else 1
    
    timestamp = datetime.now()
    experiments = []
    color_index = 0
    for name, parameters in setup_config["experiments"].items():
        experiment = parameters["type"]
        if experiment == "simulation":
            experiment = SimulationExperiment(parameters)
        else:
            raise RuntimeError(f"unknown experiment type: {experiment}")

        # choose a color for logging
        color = experiment_log_colors[color_index]
        color_index = (color_index + 1) % len(experiment_log_colors)

        experiments.append((setup_name, name, timestamp, experiment, color, report, save))

    logger.info(f"running {len(experiments)} experiments on {threads} threads")
    with Pool(threads) as p:
        p.map(run_experiment, experiments)


def run_experiment(task: Tuple[str, str, datetime, Experiment, str, bool, bool]):
    setup_name, name, timestamp, experiment, color, report, save = task

    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> <" + color + ">{extra[experiment]}</" + color + ">   <level>{message}</level>",
        filter=lambda record: "experiment" in record["extra"] and record["extra"]["experiment"] == name,
    )

    with logger.contextualize(experiment = name):
        try:
            logger.info(f"running experiment {name} of type {experiment.__class__.__name__}")
            results = experiment.run(name)
            logger.info(f"done running {name}")

            if save:
                results.save(setup_name, timestamp)

            if report:
                results.report()
        except Exception:
            logger.exception("unhandled exception")

