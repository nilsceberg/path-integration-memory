# type: ignore

import random
from matplotlib import animation
import pygame
import math
import numpy as np
from loguru import logger
from pygame.locals import *
from pim.models.stone import bee_simulator
from pim.models.new import stone
from pim.models.new.stone.cx import tb1_model, cpu4_model
from pim.models.new.stone.rate import CXRate, CXRatePontin
import scipy

def world_to_screen(pos):
    pos = np.array([1.0, -1.0]) * 0.01 * pos
    return (pos + np.array([1, 1])) * 0.5 * AREA + np.array([AREA, 0])

def graph(display, title, f, origin, size, xaxis, yaxis = (-1.0, 1.0), points = []):
    width, height = size

    xscale = abs(xaxis[1] - xaxis[0]) / width
    yscale = abs(yaxis[1] - yaxis[0]) / height

    origin = np.array(origin)
    pygame.draw.line(surface=display, start_pos=origin + (xaxis[0] / xscale, 0), end_pos=origin + (xaxis[1] / xscale, 0), color=(128, 128, 128))
    pygame.draw.line(surface=display, start_pos=origin - (0, yaxis[0] / yscale), end_pos=origin - (0, yaxis[1] / yscale), color=(128, 128, 128))

    title_text = font.render(title, False, (255, 255, 255))
    display.blit(title_text, origin + (10, -height/2))

    for p in range(0, width):
        u = xaxis[0] + p * xscale
        v = f(u)
        x = int(u / xscale)
        y = int(v / yscale)
        display.set_at(origin + (x, -y), (255, 255, 255))

    for p in points:
        if xaxis[0] <= p[0] <= xaxis[1] and yaxis[0] <= p[1] <= yaxis[1]:
            pygame.draw.circle(surface=display, center=origin + (p[0] / xscale, -p[1] / yscale), color=(255, 0, 0), radius=4)


def get_image(sheet, frame, width=23, height=23, scale=1.0):
    image = pygame.Surface((width, height), pygame.SRCALPHA)
    image.blit(sheet, (0, 0), area=(frame * width, 0, width, height))
    image = pygame.transform.scale(image, (width * scale, height * scale))
    return image

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

pygame.init()

AREA = math.floor(1920 / 2.1)
WINDOW_SIZE = (2 * AREA, AREA)

MAX_SPEED = 10.0
MAX_ANGULAR_SPEED = 3.0
TURN_SHARPNESS = 1.0

display = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("Interactive Bee Simulator")

# load sprites for bee animation
sprite_sheet_image = pygame.image.load('sprites/bee_11.png').convert_alpha()
animation_list = [get_image(sprite_sheet_image, frame, scale=2) for frame in range(11)]
last_update = pygame.time.get_ticks()
current_frame = 0

running = True

position = np.array([0.0, 0.0])
heading = 0.0

homing = False
auto = False
direction_change_timer = 0
angular_velocity = 0.0

pygame.font.init()
font = pygame.font.SysFont("Monospace", 16)
clock = pygame.time.Clock()

# Central complex:
#cx = stone.CXBasic()
# cx = CXRate()
cx = CXRatePontin()
cx.setup()
motor = 0
last_estimates = []
estimate_scaling = 600.0

cx_update_interval = 1/30
cx_update_timer = 0
cx_update_velocity = np.array([0.0, 0.0])

decoded_polar = np.array([0, 0])
while running:
    dt = clock.get_time() / 1000
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            homing = not homing

    if auto or homing:
        speed = 1.0

        if auto:
            if direction_change_timer == 0:
                angular_velocity = (random.random() * 2.0 - 1.0) * 0.1
                direction_change_timer = math.floor(random.random() * 100)
            else:
                direction_change_timer -= 1
    else:
        #velocity = np.array([0, 0])
        speed = 0.0
        angular_velocity = 0.0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            angular_velocity += 1.0
        if keys[pygame.K_RIGHT]:
            angular_velocity -= 1.0
        if keys[pygame.K_UP]:
            speed += 1.0
            #velocity += np.array([0, -1])
        if keys[pygame.K_DOWN]:
            speed -= 1.0
            #velocity += np.array([0, 1])

    if homing:
        angular_velocity = motor * 2.0 * MAX_ANGULAR_SPEED * dt
    else:
        angular_velocity = angular_velocity * MAX_ANGULAR_SPEED * dt

    heading = bee_simulator.rotate(heading, angular_velocity)

    # Velocity is represented as sin, cos in bee_simulator.py / thrust. Mistake? We flip it when inputting to update_cells
    direction = np.array([np.cos(heading), np.sin(heading)])
    velocity = direction * speed * MAX_SPEED * dt

    if speed > 0.00001:
        position += velocity
        # Is this where we went wrong? trials.py line 138, 139

    h = heading #(2.0 * np.pi - (heading + np.pi)) % (2.0 * np.pi)
    v = np.array([np.sin(h), np.cos(h)]) * speed * MAX_SPEED
    cx_update_velocity += np.array([np.sin(h), np.cos(h)]) * speed * MAX_SPEED * dt
    #if cx_update_timer > cx_update_interval:
    #    v = cx_update_velocity / cx_update_interval
    #    motor = cx.update(cx_update_interval, h, v)
    #    cx_update_timer -= cx_update_interval
    #    cx_update_velocity = np.array([0.0, 0.0])
    #else:
    #    cx_update_timer += dt
    motor = cx.update(dt * 15.0, h, v)

    

    estimated_polar = cx.estimate_position()

    last_estimates = (last_estimates + [cx.to_cartesian(estimated_polar) * estimate_scaling])[-16:]
    estimated_position = np.mean(np.array(last_estimates), 0)
    estimated_heading = cx.estimate_heading()

    # background of window
    display.fill((50,50,50))

    # Draw bee
    # animate bee
    current_time = pygame.time.get_ticks()
    if current_time - last_update >= 20:
        current_frame = (current_frame + 1) % len(animation_list)
        last_update = current_time

    # home vector
    pygame.draw.line(surface=display, start_pos=world_to_screen((0, 0)), end_pos=world_to_screen(position), color=(128, 128, 128))    

    # rotate bee
    frame_copy = pygame.transform.rotate(animation_list[current_frame], np.degrees(estimated_heading) - 90)
    center_position = tuple(map(lambda i, j: i-j, world_to_screen(estimated_position), (int(frame_copy.get_width() / 2), int(frame_copy.get_height() / 2))))
    frame_copy.set_alpha(128)
    display.blit(frame_copy, center_position)
    frame_copy = pygame.transform.rotate(animation_list[current_frame], np.degrees(heading) - 90)
    center_position = tuple(map(lambda i, j: i-j, world_to_screen(position), (int(frame_copy.get_width() / 2), int(frame_copy.get_height() / 2))))
    frame_copy.set_alpha(255)
    display.blit(frame_copy, center_position)

    #pygame.draw.circle(surface=display, center=world_to_screen(decoded), color=(128, 128, 0), radius=8)
    #pygame.draw.line(surface=display, start_pos=world_to_screen(decoded), end_pos=world_to_screen(decoded + direction * 0.1), color=(128, 128, 128))

    # GUI divider
    pygame.draw.line(surface=display, start_pos=(AREA, 0), end_pos=(AREA, AREA), color=(128, 128, 128))

    #print(memory)
    graph(display, "TB1 / Delta7", lambda x: tb1_model(np.array([estimated_heading]), x), (20, AREA - 30), (AREA - 40, 300), (0, 2 * np.pi), points=[(x * (2 * np.pi / 8), y) for x, y in enumerate(cx.tb1)])
    graph(display, "CPU4 / P-FN", lambda x: cpu4_model(estimated_polar, x) * 2.0, (20, AREA - 380), (AREA - 40, 380), (0, 2 * np.pi), (-0.32, 0.32), points=[(n / 16 * 2 * np.pi, y) for n, y in enumerate((cx.cpu4 - 0.5) * 2.0)])

    debug_text = font.render(f"Homing: {homing} | True distance from home: {np.linalg.norm(position):.02f} | Perceived distance from home: {np.linalg.norm(estimated_position):.02f}", False, (255, 255, 255))
    display.blit(debug_text, (8, 8))

    pygame.display.update()
    clock.tick(1000)
