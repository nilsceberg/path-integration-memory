import json
import sys
import numpy as np

from typing import Iterable
from datetime import datetime
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from pim.simulator import SimulationResults


# TODO: this assumes that result is a SimulationResult
def print_analysis(results: "Iterable[SimulationResults]", individual = False):
    if individual:
        for result in results:
            closest_position = result.closest_position()
            closest_distance = np.linalg.norm(closest_position)
        
            print(f"Closest distance to home: {closest_distance} steps")
            result.report()
    else:
        closest_distances = (
            np.linalg.norm(result.closest_position()) for result in results
        )
        #print(type(closest_distances))
        #sys.exit(0)
        mean_closest_distance = np.mean(list(closest_distances))

        print(f"Mean closest distance to home: {mean_closest_distance} steps")

def has_several_configs(results: "Iterable[SimulationResults]"):
    configs = set()
    for result in results:
        configs.add(result.config_id)
        if len(configs) > 1: return True
    return False

def save_analysis(results: "Iterable[SimulationResults]", results_dir="results"):
    timestamp = datetime.now()

    logger.info("calculating distances")

    configs = {}
    for result in results:
        if result.config_id not in configs:
            configs[result.config_id] = {
                "mean_min_distance": 0,
                "mean_max_distance": 0,
                "mean_extra_distance": 0,
                "mean_homing_distance": 0,
                "mean_mean_distance": 0,
                "mean_angle_rmse": 0,
                "parameters": result.parameters,
            }
        
        configs[result.config_id]["mean_min_distance"] += np.linalg.norm(result.closest_position())
        configs[result.config_id]["mean_max_distance"] += np.linalg.norm(result.farthest_position())
        configs[result.config_id]["mean_homing_distance"] += np.linalg.norm(result.homing_position())
        configs[result.config_id]["mean_extra_distance"] += np.linalg.norm(result.farthest_position()) - np.linalg.norm(result.homing_position())
        configs[result.config_id]["mean_mean_distance"] += np.mean(np.linalg.norm(result.reconstruct_path()[result.parameters["T_outbound"]:], axis=1))
        configs[result.config_id]["mean_angle_rmse"] += result.angular_rmse()

    for config in configs:
        configs[config]["mean_min_distance"] /= configs[config]["parameters"]["N"]
        configs[config]["mean_max_distance"] /= configs[config]["parameters"]["N"]
        configs[config]["mean_extra_distance"] /= configs[config]["parameters"]["N"]
        configs[config]["mean_homing_distance"] /= configs[config]["parameters"]["N"]
        configs[config]["mean_mean_distance"] /= configs[config]["parameters"]["N"]
        configs[config]["mean_angle_rmse"] /= configs[config]["parameters"]["N"]

    filename = Path(f"{results_dir}/distances/{timestamp.strftime('%Y%m%d-%H%M%S')}.json")
    path = filename.parent
    path.mkdir(parents = True, exist_ok = True)

    with open(filename, "w") as f:
        json.dump(configs, f, indent = 2, default=lambda x: x.tolist()) # assume that anything non-serializable is a numpy array
    logger.info("saved analysis results as {}", filename)