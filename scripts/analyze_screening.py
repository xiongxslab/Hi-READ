#!/usr/bin/env python3

import argparse
import sys

from _common import REPO_ROOT  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Screening analysis entry point.")
    parser.add_argument("mode", choices=["whole", "fine"], help="Select whole-chromosome screening or fine-scale screening.")
    args, remainder = parser.parse_known_args()

    if args.mode == "whole":
        from hiread.whole_chromosome_screening import main as entrypoint
    else:
        from hiread.fine_scale_screening import main as entrypoint

    sys.argv = [sys.argv[0]] + remainder
    entrypoint()


if __name__ == "__main__":
    main()
