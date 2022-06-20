from abc import abstractmethod
from typing import Union

from .setup import Experiment
from .models.new.stone import StoneExperiment
from .models.stone import StoneExperiment as OldStoneExperiment
from .models.new.winge import WingeExperiment

def models(model: str, parameters: dict) -> Union[Experiment, None]:
    if model == "stone":
        return StoneExperiment(parameters)
    if model == "stone.old":
        return OldStoneExperiment(parameters)
    if model == "winge":
        return WingeExperiment(parameters)
    else:
        return None
