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

#from pim.models.stone import bee_simulator, central_complex, cx_basic, cx_rate, trials
from pim.models.new.stone import bee_simulator
from pim.models.new import stone

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
        if connection_uuid in connections:
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


def serialize_state(
    position,
    heading,
    estimated_polar,
    estimated_position,
    estimated_heading,
    outputs,
):
    return {
        "position": position.tolist(),
        "heading": heading,
        "estimated_polar": estimated_polar.tolist(),
        "estimated_position": estimated_position.tolist(),
        "estimated_heading": estimated_heading.tolist(),
        "layers": dict([(name, output.tolist()) for (name, output) in outputs.items()])
    }

async def run_simulation():
    logger.info("starting simulation")

    # Physics state
    position = np.array([0.0, 0.0])
    heading = 0.0
    angular_velocity = 0.0

    # CX state
    #cx = stone.rate.CXRatePontin()
    cx = stone.basic.CXBasic()
    cx.setup()

    motor = 0
    last_estimates = []
    estimate_scaling = 600.0
    
    auto = False
    homing = True

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

        heading = bee_simulator.rotate(heading, angular_velocity)

        # Velocity is represented as sin, cos in bee_simulator.py / thrust. Mistake? We flip it when inputting to update_cells
        direction = np.array([np.cos(heading), np.sin(heading)])
        velocity = direction * speed * MAX_SPEED * dt

        if speed > 0.00001:
            position += velocity
            # Is this where we went wrong? trials.py line 138, 139

        h = heading #(2.0 * np.pi - (heading + np.pi)) % (2.0 * np.pi)
        v = np.array([np.sin(h), np.cos(h)]) * speed * MAX_SPEED

        motor = cx.update(dt, h, v)

        estimated_polar = cx.estimate_position()

        last_estimates = (last_estimates + [cx.to_cartesian(estimated_polar) * estimate_scaling])[-16:]
        estimated_position = np.mean(np.array(last_estimates), 0)
        estimated_heading = cx.estimate_heading()

        publish(serialize_state(
            position,
            heading,
            estimated_polar,
            estimated_position,
            estimated_heading,
            {
                "motor": np.array([motor]),
                "cpu4": cx.cpu4,
                "tb1": cx.tb1,
            }
        ))


async def main():
    await asyncio.gather(
        run_simulation(),
        accept_connections(),
    )


if __name__ == "__main__":
    asyncio.run(main())
