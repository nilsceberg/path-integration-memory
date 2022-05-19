from abc import abstractmethod
from typing import Callable, Tuple, Union
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from loguru import logger

import sys
import json

class ExperimentResults:
    def __init__(self, name: str, parameters: dict) -> None:
        self.name = name
        self.parameters = parameters

    @abstractmethod
    def report(self):
        pass

    @abstractmethod
    def serialize() -> object:
        pass

    def save(self, setup: str, timestamp: datetime, results_dir="results"):
        output = {
            "setup": setup,
            "timestamp": timestamp.isoformat(),
            "name": self.name,
            "parameters": self.parameters,
            "results": self.serialize()
        }
        filename = Path(f"{results_dir}/{setup}_{timestamp.strftime('%Y%m%d-%H%M%S')}/{self.name}.json")
        path = filename.parent
        path.mkdir(parents = True, exist_ok = True)
        with open(filename, "w") as f:
            json.dump(output, f, indent = 2)

class Experiment:
    @abstractmethod
    def run(self, name: str) -> ExperimentResults:
        pass

experiment_log_colors = ["red", "magenta", "yellow", "red", "green", "blue"]
def run(setup_name: str, setup_config: dict, models: Callable[[str, dict], Union[Experiment, None]], report = True, save = False):
    threads = setup_config["threads"] if "threads" in setup_config else 1
    
    timestamp = datetime.now()
    experiments = []
    color_index = 0
    for name, parameters in setup_config["experiments"].items():
        model = parameters["model"]
        experiment = models(model, parameters)
        if not experiment:
            raise RuntimeError(f"unknown model: {model}")

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
        logger.info(f"running experiment {name} of type {experiment.__class__.__name__}")
        results = experiment.run(name)
        logger.info(f"done running {name}")

        if save:
            results.save(setup_name, timestamp)

        if report:
            results.report()
