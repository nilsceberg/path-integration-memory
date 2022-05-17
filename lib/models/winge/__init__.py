from datetime import datetime
from lib.setup import Experiment, ExperimentResults
import time

class WingeResults(ExperimentResults):
    def __init__(self, name: str, parameters: dict) -> None:
        super().__init__(name, parameters)

    def report(self):
        pass

    def serialize(self):
        return "winge results"

class WingeExperiment(Experiment):
    def __init__(self, parameters: dict) -> None:
        super().__init__()
        self.parameters = parameters

    def run(self, name: str) -> ExperimentResults:
        print(f"running winge model (experiment {name})")
        time.sleep(5)
        return WingeResults(name, self.parameters)
