from lib.setup import ExperimentSetup, ExperimentResults

class StoneResults(ExperimentResults):
    def report(self):
        pass

class StoneSetup(ExperimentSetup):
    def __init__(self, parameters: dict) -> None:
        super().__init__()

    def run(self, name: str) -> ExperimentResults:
        print(f"running stone setup (experiment {name})")
        return StoneResults()
