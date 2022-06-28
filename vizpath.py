from pim.models.new.stone import trials
from pim.models.new.stone import plotter
import matplotlib.pyplot as plt
import pickle

headings, velocities = trials.generate_route(T = 1500, vary_speed = True)

with open("../fastbee/path.pickle", "rb") as outbound, open("../fastbee/inbound.pickle", "rb") as inbound:
    out_path = pickle.load(outbound)
    in_path = pickle.load(inbound)

fig, ax = plotter.plot_route(
    h = out_path[0] + in_path[0],
    v = out_path[1] + in_path[1],
    T_outbound = len(out_path[0]),
    T_inbound = len(in_path[0]),
    plot_speed = True,
    plot_heading = True,
    quiver_color = "black",
    )
plt.show()
