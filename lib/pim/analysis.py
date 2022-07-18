import numpy as np
from tqdm import tqdm

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
        mean_closest_distance = np.mean(list(tqdm(closest_distances, total=len(results))))

        print(f"Mean closest distance to home: {mean_closest_distance} steps")
