import asyncio
import json
import datetime
from typing import Any, Union
import uuid
import websockets.server
import numpy as np
from websockets.exceptions import ConnectionClosedOK
from loguru import logger

TIME_STEP = 0.016

connections = {}

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

def serialize_state(position):
    return {
        "position": position.tolist(),
    }

async def run_simulation():
    logger.info("starting simulation")

    position = np.array([0.0, 0.0])
    theta = 0.0

    last_frame = datetime.datetime.now()
    while True:
        # await new frame and calculate delta time
        await asyncio.sleep(TIME_STEP)
        now = datetime.datetime.now()
        dt = (now - last_frame).microseconds / 1e6
        last_frame = now

        # simulate
        theta += dt
        position = np.array([
            np.cos(theta),
            np.sin(theta),
        ])

        publish(serialize_state(
            position
        ))


async def main():
    await asyncio.gather(
        run_simulation(),
        accept_connections(),
    )


if __name__ == "__main__":
    asyncio.run(main())
