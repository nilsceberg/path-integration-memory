from abc import abstractmethod
from typing import Callable, Tuple, Union
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from loguru import logger
import random

import sys
import json

from . import simulator
from .experiment import Experiment

experiment_log_colors = ["red", "magenta", "yellow", "red", "green", "blue"]
def run(setup_name: str, setup_config: dict, report = True, save = False, experiment_loggers = True):
    threads = setup_config.get("threads", 1)
    
    timestamp = datetime.now()
    experiments = []
    color_index = 0
    for name, parameters in setup_config["experiments"].items():
        N = parameters.get("N", 1)
        for i in range(N):
            experiment = parameters["type"]
            if experiment == "simulation":
                experiment = simulator.SimulationExperiment(parameters)
            else:
                raise RuntimeError(f"unknown experiment type: {experiment}")

            # choose a color for logging
            color = experiment_log_colors[color_index]
            color_index = (color_index + 1) % len(experiment_log_colors)

            experiments.append((setup_name, f"{name}-{i}" if N > 1 else name, timestamp, experiment, color, report, save, experiment_loggers))

    logger.info(f"running {len(experiments)} experiments on {threads} threads")
    return Pool(threads).imap_unordered(run_experiment, experiments), len(experiments)


def run_experiment(task: Tuple[str, str, datetime, Experiment, str, bool, bool, bool]):
    setup_name, name, timestamp, experiment, color, report, save, experiment_loggers = task

    if experiment_loggers:
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

            return results
        except Exception:
            logger.exception("unhandled exception")



def load_results(filename):
    path = Path(filename)

    # If a directory is specified, load all results:
    if path.is_dir():
        paths = list(path.iterdir())
    else:
        paths = [path]

    results = []
    for path in paths:
        with path.open() as f:
            data = json.load(f)
            experiment_type = data["parameters"]["type"]
            if experiment_type == "simulation":
                results.append(simulator.load_results(data))
            else:
                raise RuntimeError(f"unknown experiment type: {experiment_type}")

    return results

