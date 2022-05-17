#!/usr/bin/env python3

from loguru import logger

import argparse
import json
import os
import pathlib

import lib
import lib.setup

parser = argparse.ArgumentParser()
parser.add_argument("setup")

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
