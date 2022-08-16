#!/usr/bin/env python3

from loguru import logger

import argparse
import json
import pathlib
import sys
import collections
from tqdm import tqdm

import pim.setup
import pim.analysis


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

def run_experiment(args):
    if args.progress:
        logger.remove()

    try:
        with open(args.setup, "r") as f:
            setup_file = open(args.setup, "r")
            setup = json.load(setup_file)
            logger.debug(f"setup: {args.setup}")

        # override setup config with command-line arguments
        if args.threads:
            setup["threads"] = args.threads

        # if one or more --only, remove experiments that weren't specified
        if args.only != []:
            setup["experiments"] = { name: experiment for (name, experiment) in setup["experiments"].items() if name in args.only }
        
        # add any command-line --record entries to experiments
        for experiment in setup["experiments"].values():
            if "record" in experiment:
                experiment["record"] += args.record
            else:
                experiment["record"] = args.record

        # override arbitrary model parameters
        for override in args.override:
            path, value = override.split("=")
            obj = setup["experiments"]
            deep_update(obj, path, value)

        setup_name = pathlib.Path(args.setup).stem

        # consume iterator
        results, total = pim.setup.run(setup_name, setup, save = args.save, report = args.report, experiment_loggers = not args.progress)

        if args.progress:
            results = tqdm(
                results,
                total=total,
                colour="blue",
            )

        collections.deque(
            results,
            maxlen = 0,
        )
        

    except FileNotFoundError:
        logger.error(f"error: setup file '{args.setup}' not found")

    except Exception:
        logger.exception("unhandled exception")


def analyze_results(args):
    # paths is a list of paths, resolved from arguments,
    # load_results gives us a lazy generator for actually loading them.
    # Once we've started to go through them once to determine whether they contain
    # several configs, we need to restart the generator, hence the multiple load_results calls.
    paths = pim.setup.enumerate_results(args.results)

    # TODO: maybe we could use a metadata file with information about config number etc.,
    # so that we don't have to load all files twice in the worst case (when there is only one config).
    several = pim.analysis.has_several_configs(pim.setup.load_results(paths))

    results = pim.setup.load_results(tqdm(paths, colour="green", delay=1))
    if several:
        pim.analysis.save_analysis(results)
    else:
        pim.analysis.print_analysis(results, individual = (len(paths) == 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(help="subcommand")
    experiment_parser = subparsers.add_parser("experiment",  aliases=["ex"], help="run experiment(s)")
    experiment_parser.set_defaults(func=run_experiment)
    experiment_parser.add_argument("setup")
    experiment_parser.add_argument("--threads", type=int, help="run THREADS experiments in parallel")
    
    experiment_parser.add_argument("--report", action="store_true", help="generate report")
    experiment_parser.add_argument("--throw", dest="save", action="store_false", help="don't save results")
    experiment_parser.add_argument("--override", action="append", help="override experiment parameter, e.g. --override stone.noise=0.5", default=[])
    experiment_parser.add_argument("--record", action="append", help="additional elements to record (for every experiment)", default=[])
    experiment_parser.add_argument("--only", action="append", help="run only specified experiments", default=[])
    
    experiment_parser.add_argument("--progress", action="store_true", help="show progress bar instead of log output")

    analyze_parser = subparsers.add_parser("analyze", aliases=["an"], help="analyze results")
    analyze_parser.add_argument("results", help="result file/directory", nargs='+')
    analyze_parser.set_defaults(func=analyze_results)

    args = parser.parse_args()
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>   <level>{message}</level>",
        filter=lambda record: "experiment" not in record["extra"]
    )

    if "func" in args:
        args.func(args)
    else:
        parser.print_help()
