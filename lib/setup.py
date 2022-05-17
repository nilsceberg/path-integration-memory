from abc import abstractmethod
import json
from typing import Callable, Union
from datetime import datetime
from pathlib import Path
import os

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
    timestamp = datetime.now()
    for name, parameters in setup_config["experiments"].items():
        print(f"running experiment {name} (model: {parameters['model']})")

        model = parameters["model"]

        experiment = models(model, parameters)
        if not experiment:
            raise RuntimeError(f"unknown model: {model}")

        results = experiment.run(name)

        results.save(setup_name, timestamp)
        results.report()
