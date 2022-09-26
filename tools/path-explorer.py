import random
from matplotlib import animation
import pygame
import math
import numpy as np
from loguru import logger
from pygame.locals import *
import scipy

from pim.simulator import estimate_search_pattern, farthest_position, generate_path_from_waypoints, path_center_of_mass, reconstruct_path

WINDOW_SIZE=(1440, 920)
display = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Path Explorer")

center = np.array(WINDOW_SIZE) / 2

def draw_path(points, color):
    for a, b in zip(points, points[1:]):
        pygame.draw.line(display, color, center + a, center + b)

waypoints = []
path = []
search_pattern_radius = 0
search_pattern_center = np.zeros(2)
path_center = np.zeros(2)

def place_waypoint(waypoint):
    global path, search_pattern_radius, search_pattern_center, path_center, velocities, headings
    if len(waypoints) == 0 or np.all(waypoint != waypoints[-1]):
        waypoints.append(waypoint)

        headings, velocities = generate_path_from_waypoints(waypoints, rotation_speed=0.05)
        path = reconstruct_path(velocities) + np.array(waypoints[0])
        path_center = path_center_of_mass(path)
        search_pattern = estimate_search_pattern(path)
        search_pattern_center = path_center_of_mass(search_pattern)
        search_pattern_radius = farthest_position(search_pattern - search_pattern_center)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            # place waypoint
            pass

    if pygame.mouse.get_pressed()[0]:
        place_waypoint(pygame.mouse.get_pos() - center)

    # render:
    display.fill((0,0,0))

    draw_path(waypoints, (100, 100, 100))
    draw_path(path, "white")
    pygame.draw.circle(display, "red", center + search_pattern_center, search_pattern_radius, width=1)
    pygame.draw.circle(display, "blue", center + path_center, 2, width=0)
    pygame.draw.circle(display, "green", center + search_pattern_center, 2, width=0)

    pygame.display.update()

print(waypoints)
