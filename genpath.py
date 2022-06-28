from pim.models.new.stone import trials
import pickle

headings, velocities = trials.generate_route(T = 1500, vary_speed = True)

with open("../fastbee/path.pickle", "wb") as f:
    pickle.dump((headings.tolist(), velocities.tolist()), f)
