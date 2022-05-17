from abc import abstractmethod

from .setup import ExperimentSetup
from .models.stone import StoneSetup
from .models.winge import WingeSetup

def models(model: str, parameters: dict) -> ExperimentSetup | None:
    if model == "stone":
        return StoneSetup(parameters)
    if model == "winge":
        return WingeSetup(parameters)
    else:
        return None
