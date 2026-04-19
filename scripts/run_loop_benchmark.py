#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from _common import REPO_ROOT


WORKFLOW_ROOT = REPO_ROOT / "workflows" / "loop_benchmark"

SUBCOMMAND_TO_SCRIPT = {
    "peak2peak-ntko-tko": Path("run_loop_benchmark.py"),
    "all2all-ntko-tko": Path("code/run_ntko_tko_chr15_multimethod_all2all.py"),
    "all2all-five": Path("code/run_five_datasets_chr15_multimethod_all2all.py"),
    "report": Path("code/build_final_peak2peak_all2all_report.py"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrapper for the packaged loop benchmark workflows.",
        usage="run_loop_benchmark.py <subcommand> [script args ...]",
    )
    parser.add_argument("subcommand", choices=sorted(SUBCOMMAND_TO_SCRIPT))
    parser.add_argument(
        "--workflow-root",
        default=str(WORKFLOW_ROOT),
        help="Override the copied loop benchmark workflow root.",
    )
    parser.add_argument(
        "--input-root",
        default=os.environ.get("HIREAD_LOOP_BENCH_INPUT_ROOT"),
        help="Optional root for loop benchmark input npy/peaks/results.",
    )
    args, remainder = parser.parse_known_args()

    workflow_root = Path(args.workflow_root)
    script_path = workflow_root / SUBCOMMAND_TO_SCRIPT[args.subcommand]
    if not script_path.exists():
        raise FileNotFoundError(f"Workflow entry point not found: {script_path}")
    env = os.environ.copy()
    if args.input_root:
        env["HIREAD_LOOP_BENCH_INPUT_ROOT"] = args.input_root

    cmd = [sys.executable, str(script_path), *remainder]
    raise SystemExit(subprocess.run(cmd, cwd=workflow_root, env=env).returncode)


if __name__ == "__main__":
    main()
