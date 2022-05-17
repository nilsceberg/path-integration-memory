from abc import abstractmethod
from typing import Callable, Tuple, Union
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool
from loguru import logger

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

def run(setup_name: str, setup_config: dict, models: Callable[[str, dict], Union[Experiment, None]]):
    threads = setup_config["threads"] if "threads" in setup_config else 1
    
    timestamp = datetime.now()
    experiments = []
    for name, parameters in setup_config["experiments"].items():
        model = parameters["model"]
        experiment = models(model, parameters)
        if not experiment:
            raise RuntimeError(f"unknown model: {model}")
        experiments.append((setup_name,name,timestamp,experiment))

    logger.info(f"running {len(experiments)} experiments on {threads} threads")
    with Pool(threads) as p:
        p.map(run_experiment, experiments)


def run_experiment(task: Tuple[str,str,datetime,Experiment]):
    setup_name, name, timestamp, experiment = task
    logger.info(f"running experiment {name} of type {type(experiment)})")
    results = experiment.run(name)
    logger.info(f"done running {name}")
    results.save(setup_name, timestamp)
    results.report()