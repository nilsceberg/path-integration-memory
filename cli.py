#!/usr/bin/env python3

from loguru import logger

import argparse
import json
import os
import pathlib
import sys

import pim
import pim.setup

parser = argparse.ArgumentParser()
parser.add_argument("setup")
parser.add_argument("--threads", type=int)

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

        # override setup config with command-line arguments
        if args.threads:
            setup["threads"] = args.threads

        setup_name = pathlib.Path(args.setup).stem
        pim.setup.run(setup_name, setup, pim.models)

    except FileNotFoundError:
        logger.error(f"error: setup file '{args.setup}' not found")

    except Exception:
        logger.exception("unhandled exception")
