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
parser.add_argument("--threads", type=int, help="run THREADS experiments in parallel")

parser.add_argument("--report", action="store_true", help="generate report")
parser.add_argument("--throw", dest="save", action="store_false", help="don't save results")
parser.add_argument("--override", action="append", help="override experiment parameter, e.g. --override stone.noise=0.5", default=[])

logger.remove()
logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>   <level>{message}</level>",
    filter=lambda record: "experiment" not in record["extra"]
)

def deep_update(obj: dict, path: str, value: str):
    keys = path.split(".")
    for key in keys[:-1]:
        obj = obj[key]
    key = keys[-1]
    if key not in obj:
        obj[key] = value
    else:
        t = type(obj[key])
        obj[key] = t(value)

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

        # override arbitrary model parameters
        for override in args.override:
            path, value = override.split("=")
            obj = setup["experiments"]
            deep_update(obj, path, value)

        setup_name = pathlib.Path(args.setup).stem
        pim.setup.run(setup_name, setup, pim.models, save = args.save, report = args.report)

    except FileNotFoundError:
        logger.error(f"error: setup file '{args.setup}' not found")

    except Exception:
        logger.exception("unhandled exception")
