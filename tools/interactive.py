import asyncio
import json
import datetime
from typing import Any, Union
import uuid
import websockets.server
import numpy as np
import scipy.optimize
from websockets.exceptions import ConnectionClosedOK
from loguru import logger

from pim.models.stone import bee_simulator, central_complex, cx_basic, cx_rate, trials

# Settings
TIME_STEP = 0.016
MAX_SPEED = 10.0
MAX_ANGULAR_SPEED = 3.0
TURN_SHARPNESS = 1.0

# Globals:
connections = {}
auto = False
homing = True

async def socket_send(socket, connection_uuid, message):
    encoded = json.dumps(message)
    return await socket.send(encoded)

async def socket_receive(socket, connection_uuid, _):
    return await socket.recv()

async def perform_socket_io(connection_uuid, socket, io, args = None):
    try:
        return await io(socket, connection_uuid, args)

    except Exception as e:
        logger.debug("connection closed: {} ({}) - {}", connection_uuid, socket.remote_address, e)
        logger.debug("removed connection {}", connection_uuid)
        connections.pop(connection_uuid)

        return None


async def handle_connection(socket):
    connection_uuid = uuid.uuid4()
    connections[connection_uuid] = socket
    logger.debug("new viewer connection: {} ({})", connection_uuid, socket.remote_address)

    while True:
        message = await perform_socket_io(connection_uuid, socket, socket_receive)
        if message is None:
            break

        logger.debug("received: {}", message)

def publish(message):
    for connection_uuid, socket in connections.items():
        asyncio.ensure_future(perform_socket_io(connection_uuid, socket, socket_send, message))

async def accept_connections():
    logger.info("starting publisher on port 8001")
    async with websockets.server.serve(handle_connection, "", 8001):
        await asyncio.Future() # run forever

def cpu4_model(args, x):
    return args[0] * np.cos(np.pi + 2*x + args[1])

def tb1_model(args, x):
    return 0.5 + np.cos(np.pi + x + args[0]) * 0.5

def fit_cpu4(data):
    xs = np.linspace(0, 2*np.pi, central_complex.N_CPU4, endpoint = False)
    error = lambda args: cpu4_model(args, xs) - data
    params, _ = scipy.optimize.leastsq(error, np.array([0.1, 0.01]))
    return params

def fit_tb1(data):
    xs = np.linspace(0, 2*np.pi, central_complex.N_TB1, endpoint = False)
    error = lambda args: tb1_model(args, xs) - data
    params, _ = scipy.optimize.leastsq(error, np.array([0.1]))
    return params

def decode_cpu4(cpu4_output):
    cpu4_normalized = (cpu4_output - 0.5)*2.0
    params = fit_cpu4(cpu4_normalized)
    return params

def decode_tb1(tb1_output):
    heading = fit_tb1(tb1_output)[0]
    return heading

def serialize_state(
    position,
    heading,
    decoded_position,
    decoded_heading,
    tl2,
    cl1,
    tb1,
    tn1,
    tn2,
    memory,
    cpu4,
    cpu1,
    motor,
):
    return {
        "position": position.tolist(),
        "heading": heading,
        "decoded_position": decoded_position.tolist(),
        "decoded_heading": decoded_heading,
        "tl2": tl2.tolist(),
        "cl1": cl1.tolist(),
        "tb1": tb1.tolist(),
        "tn1": tn1.tolist(),
        "tn2": tn2.tolist(),
        "memory": memory.tolist(),
        "cpu4": cpu4.tolist(),
        "cpu1": cpu1.tolist(),
        "motor": motor,
    }

async def run_simulation():
    logger.info("starting simulation")

    # Physics state
    position = np.array([0.0, 0.0])
    heading = 0.0
    angular_velocity = 0.0

    # CX state
    cx = cx_rate.CXRate(0.1)
    tb1 = np.zeros(central_complex.N_TB1)
    memory = 0.5 * np.ones(central_complex.N_CPU4)
    motor = 0
    
    auto = False
    homing = True

    last_decoded = []

    last_frame = datetime.datetime.now()
    while True:
        # await new frame and calculate delta time
        await asyncio.sleep(TIME_STEP)
        now = datetime.datetime.now()
        dt = (now - last_frame).microseconds / 1e6
        last_frame = now

        # simulate
        speed = 1.0
        if homing:
            angular_velocity = motor
        else:
            angular_velocity = angular_velocity

        heading = bee_simulator.rotate(heading, angular_velocity)
        direction = np.array([np.cos(heading), np.sin(heading)])
        velocity = direction * speed * MAX_SPEED * dt

        if speed > 0.0001:
            # apply velocity
            position += velocity

        # update CX
        h = heading #(2.0 * np.pi - (heading + np.pi)) % (2.0 * np.pi)
        v = np.array([np.sin(h), np.cos(h)]) * speed * MAX_SPEED * dt
        tl2, cl1, tb1, tn1, tn2, memory, cpu4, cpu1, motor = trials.update_cells(h, v, tb1, memory, cx)

        decoded_polar = decode_cpu4(cpu4) #decode_position(cpu4_mem.reshape(2, -1), cpu4_mem_gain)
        last_decoded = (last_decoded + [np.array([
            np.cos(decoded_polar[1] + np.pi),
            -np.sin(decoded_polar[1] + np.pi),
        ]) * decoded_polar[0] * 300.0])[-16:]

        decoded_position = np.mean(np.array(last_decoded), 0)
        decoded_heading = decode_tb1(tb1)

        publish(serialize_state(
            position,
            heading,
            decoded_position,
            decoded_heading,
            tl2,
            cl1,
            tb1,
            tn1,
            tn2,
            memory,
            cpu4,
            cpu1,
            motor,
        ))


async def main():
    await asyncio.gather(
        run_simulation(),
        accept_connections(),
    )


if __name__ == "__main__":
    asyncio.run(main())
