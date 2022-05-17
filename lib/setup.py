from abc import abstractmethod
from typing import Callable, Union

class ExperimentResults:
    @abstractmethod
    def report(self):
        pass

class ExperimentSetup:
    @abstractmethod
    def run(self, name: str) -> ExperimentResults:
        pass

def run(setup_config: dict, models: Callable[[str, dict], Union[ExperimentSetup, None]]):
    for name, parameters in setup_config.items():
        print(f"running experiment {name} (model: {parameters['model']})")

        model = parameters["model"]

        setup = models(model, parameters)
        if not setup:
            raise RuntimeError(f"unknown model: {model}")

        results = setup.run(name)
        results.report()
