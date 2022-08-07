import json
import sys
import numpy as np

from datetime import datetime
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from pim.simulator import SimulationResults


# TODO: this assumes that result is a SimulationResult
def print_analysis(results):
    if len(results) == 1:
        result = results[0]
        closest_position = result.closest_position()
        closest_distance = np.linalg.norm(closest_position)
    
        print(f"Closest distance to home: {closest_distance} steps")
        result.report()
    else:
        closest_distances = (
            np.linalg.norm(result.closest_position()) for result in results
        )
        print(type(closest_distances))
        sys.exit(0)
        mean_closest_distance = np.mean(list(tqdm(closest_distances, total=len(results), colour="green")))

        print(f"Mean closest distance to home: {mean_closest_distance} steps")

def has_several_configs(results: list[SimulationResults]):
    configs = set()
    for result in results:
        configs.add(result.config_id)
        if len(configs) > 1: return True
    return False

def save_analysis(results: list[SimulationResults], results_dir="results"):
    timestamp = datetime.now()

    logger.info("calculating distances")

    configs = {}
    for result in results:
        if result.config_id in configs:
            configs[result.config_id]["mean_distance"] += np.linalg.norm(result.closest_position())
        else:
            configs[result.config_id] = {
                "mean_distance": 0,
                "parameters": result.parameters,
            }
    for config in configs:
        configs[config]["mean_distance"] /= configs[config]["parameters"]["N"]

    filename = Path(f"{results_dir}/distances/{timestamp.strftime('%Y%m%d-%H%M%S')}.json")
    path = filename.parent
    path.mkdir(parents = True, exist_ok = True)

    with open(filename, "w") as f:
        json.dump(configs, f, indent = 2, default=lambda x: x.tolist()) # assume that anything non-serializable is a numpy array
    logger.info("saved analysis results as {}", filename)