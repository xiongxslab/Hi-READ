#!/usr/bin/env python3

import argparse
import sys

from _common import REPO_ROOT  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Structural-variant analysis entry point.")
    parser.add_argument("mode", choices=["deletion", "duplication", "translocation"], help="Select the SV analysis workflow.")
    args, remainder = parser.parse_known_args()

    if args.mode == "deletion":
        from hiread.deletion import main as entrypoint
    elif args.mode == "duplication":
        from hiread.inference.duplication import main as entrypoint
    else:
        from hiread.inference.translocation import main as entrypoint

    sys.argv = [sys.argv[0]] + remainder
    entrypoint()


if __name__ == "__main__":
    main()
