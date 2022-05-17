from lib.setup import ExperimentSetup, ExperimentResults

class WingeResults(ExperimentResults):
    def report(self):
        pass

class WingeSetup(ExperimentSetup):
    def __init__(self, parameters: dict) -> None:
        super().__init__()

    def run(self, name: str) -> ExperimentResults:
        print(f"running winge setup (experiment {name})")
        return WingeResults()
