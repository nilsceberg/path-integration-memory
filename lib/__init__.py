from abc import abstractmethod

from .setup import ExperimentSetup
from .models.stone import StoneSetup

def models(model: str, parameters: dict) -> ExperimentSetup | None:
    if model == "stone":
        return StoneSetup(parameters)
    else:
        return None
