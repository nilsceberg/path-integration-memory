from loguru import logger

from ...setup import Experiment, ExperimentResults

class StoneResults(ExperimentResults):
    def __init__(self, name: str, parameters: dict) -> None:
        super().__init__(name, parameters)

    def report(self):
        print(self.serialize())

    def serialize(self):
        return "stone results"

class StoneExperiment(Experiment):
    def __init__(self, parameters: dict) -> None:
        super().__init__()
        self.parameters = parameters

    def run(self, name: str) -> ExperimentResults:
        logger.info(f"running stone model (experiment {name})")
        return StoneResults(name, self.parameters)
