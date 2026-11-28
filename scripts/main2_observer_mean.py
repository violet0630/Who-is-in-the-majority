"""Separate, opt-in observer-mean experiment; importing never calls an API."""

import argparse
from pathlib import Path

from scripts.majority_factorial import PROJECT, load_prepared, prepare
from scripts.main2_v2 import run_workflow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(PROJECT / "configs/main2_observer_mean.json", args.prepared)
    if args.allow_api:
        manifest = load_prepared(args.prepared)
        if manifest["protocol"]["settings"].get("nonobserver_perception") != "observer_mean":
            raise ValueError("This entry point requires an observer-mean bundle")
        run_workflow(args.prepared)


if __name__ == "__main__":
    main()
