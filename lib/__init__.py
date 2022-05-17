from abc import abstractmethod
from typing import Union

from .setup import ExperimentSetup
from .models.stone import StoneSetup

def models(model: str, parameters: dict) -> Union[ExperimentSetup, None]:
    if model == "stone":
        return StoneSetup(parameters)
    else:
        return None
