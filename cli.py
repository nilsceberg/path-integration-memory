#!/usr/bin/env python3

from loguru import logger

import argparse
import json
import os
import pathlib
import sys

import lib
import lib.setup

parser = argparse.ArgumentParser()
parser.add_argument("setup")
logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>   <level>{message}</level>",
    filter=lambda record: "experiment" not in record["extra"]
)

if __name__ == "__main__":
    args = parser.parse_args()

    try:
        with open(args.setup, "r") as f:
            setup_file = open(args.setup, "r")
            setup = json.load(setup_file)
            logger.debug(f"setup: {args.setup}")

        setup_name = pathlib.Path(args.setup).stem
        lib.setup.run(setup_name, setup, lib.models)

    except FileNotFoundError:
        logger.error(f"error: setup file '{args.setup}' not found")

    except Exception:
        logger.exception("unhandled exception")
