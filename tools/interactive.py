# type: ignore

import random
from matplotlib import animation
import pygame
import math
import numpy as np

from pygame.locals import *
from pim.models.stone import bee_simulator, central_complex, cx_basic, cx_rate, trials

def world_to_screen(pos):
    pos = np.array([1.0, -1.0]) * 0.1 * pos
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
        x = np.int(u / xscale)
        y = np.int(v / yscale)
        display.set_at(origin + (x, -y), (255, 255, 255))

    for p in points:
        if xaxis[0] <= p[0] <= xaxis[1] and yaxis[0] <= p[1] <= yaxis[1]:
            pygame.draw.circle(surface=display, center=origin + (p[0] / xscale, -p[1] / yscale), color=(255, 0, 0), radius=4)


def get_image(sheet, frame, width=23, height=23, scale=1.0):
    image = pygame.Surface((width, height), pygame.SRCALPHA)
    image.blit(sheet, (0, 0), area=(frame * width, 0, width, height))
    image = pygame.transform.scale(image, (width * scale, height * scale))
    return image


pygame.init()

AREA = math.floor(1920 / 2.1)
WINDOW_SIZE = (2 * AREA, AREA)

MAX_SPEED = 1.0
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

# TB and memory:
cx = cx_basic.CXBasic()
#cx = cx_rate.CXRatePontin(0.1)
tb1 = np.zeros(central_complex.N_TB1)
memory = 0.5 * np.ones(central_complex.N_CPU4)
motor = 0

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
        angular_velocity = motor
    else:
        angular_velocity = angular_velocity * MAX_ANGULAR_SPEED * dt

    heading = bee_simulator.rotate(heading, angular_velocity)

    # Velocity is represented as sin, cos in bee_simulator.py / thrust. Mistake? We flip it when inputting to update_cells
    direction = np.array([np.cos(heading), np.sin(heading)])
    velocity = direction * speed * MAX_SPEED * dt

    if speed > 0.00001:
        position += velocity
        # Is this where we went wrong? trials.py line 138, 139
        tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor = trials.update_cells(heading, np.flip(velocity), tb1, memory, cx)

    # background of window
    display.fill((50,50,50))


    # Draw bee
    decoded_polar = cx.decode_cpu4(cx.cpu4_output(memory)) #decode_position(cpu4_mem.reshape(2, -1), cpu4_mem_gain)
    decoded = np.array([
        np.cos(decoded_polar[0]),
        np.sin(decoded_polar[0]),
    ]) * decoded_polar[1] * 1.0

    # animate bee
    current_time = pygame.time.get_ticks()
    if current_time - last_update >= 80:
        current_frame = (current_frame + 1) % len(animation_list)
        last_update = current_time

    # rotate bee
    frame_copy = pygame.transform.rotate(animation_list[current_frame], np.degrees(heading) - 90)
    center_position = tuple(map(lambda i, j: i-j, world_to_screen(position), (int(frame_copy.get_width() / 2), int(frame_copy.get_height() / 2))))
    display.blit(frame_copy, center_position)

    pygame.draw.circle(surface=display, center=world_to_screen(decoded), color=(128, 128, 0), radius=8)
    pygame.draw.line(surface=display, start_pos=world_to_screen(decoded), end_pos=world_to_screen(decoded + direction * 0.1), color=(128, 128, 128))

    pygame.draw.line(surface=display, start_pos=world_to_screen((0, 0)), end_pos=world_to_screen(position), color=(128, 128, 128))    
    pygame.draw.line(surface=display, start_pos=(AREA, 0), end_pos=(AREA, AREA), color=(128, 128, 128))

    # print(memory)
    tb1f = lambda x: (1.0 + np.cos(np.pi + x + heading)) / 2.0
    graph(display, "TB1", tb1f, (20, AREA - 70), (AREA - 40, 100), (0, 2 * np.pi), points=[(x * (2 * np.pi / 8), y) for x, y in enumerate(tb1)])
    graph(display, "CPU4", lambda x: 0.5, (20, AREA - 200), (AREA - 40, 100), (0, 16), (-0.02, 0.02), points=[(n, y) for n, y in enumerate(memory - 0.5)])

    debug_text = font.render(f"Homing: {homing} | True distance from home: {np.linalg.norm(position):.02f} | Perceived distance from home: {np.linalg.norm(decoded):.02f}", False, (255, 255, 255))
    display.blit(debug_text, (8, 8))

    pygame.display.update()
    clock.tick(1000)
