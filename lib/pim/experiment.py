from abc import abstractmethod
from datetime import datetime
from pathlib import Path
import pickle
from loguru import logger


class ExperimentResults:
    def __init__(self, name: str, config_id: str, parameters: dict) -> None:
        self.name = name
        self.config_id = config_id
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
            "config_id": self.config_id,
            "parameters": self.parameters,
            "results": self.serialize()
        }
        filename = Path(f"{results_dir}/{setup}_{timestamp.strftime('%Y%m%d-%H%M%S')}/{self.name}.json")
        path = filename.parent
        path.mkdir(parents = True, exist_ok = True)
        with open(filename, "wb") as f:
            pickle.dump(output, f)
        logger.info("saved experiment results as {}", filename)

class Experiment:
    @abstractmethod
    def run(self, name: str, config_id: str) -> ExperimentResults:
        pass

