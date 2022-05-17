from abc import abstractmethod
from typing import Union

from .setup import ExperimentSetup
from .models.stone import StoneSetup
from .models.winge import WingeSetup

def models(model: str, parameters: dict) -> Union[ExperimentSetup, None]:
    if model == "stone":
        return StoneSetup(parameters)
    if model == "winge":
        return WingeSetup(parameters)
    else:
        return None
