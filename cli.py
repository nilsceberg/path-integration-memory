#!/usr/bin/env python3

import argparse
import json

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
            print(f"setup: {args.setup}")

        lib.setup.run(setup, lib.models)

    except FileNotFoundError:
        print(f"error: setup file '{args.setup}' not found")
    
    except Exception as e:
        print(f"error: {e}")
