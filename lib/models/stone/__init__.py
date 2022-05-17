from lib.setup import ExperimentSetup, ExperimentResults

class StoneResults(ExperimentResults):
    def __init__(self, name: str, parameters: dict) -> None:
        super().__init__(name, parameters)

    def report(self):
        pass

    def serialize(self):
        return "stone results"

class StoneSetup(ExperimentSetup):
    def __init__(self, parameters: dict) -> None:
        super().__init__()
        self.parameters = parameters

    def run(self, name: str) -> ExperimentResults:
        print(f"running stone setup (experiment {name})")
        return StoneResults(name, self.parameters)
