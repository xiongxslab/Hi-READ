#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _common import REPO_ROOT


WORKFLOW_ROOT = REPO_ROOT / "workflows" / "loops_cluster"

SUBCOMMAND_TO_SCRIPT = {
    "cluster-only": Path("code/run_cluster_only_pipeline.py"),
    "umap": Path("code/run_batch_umap.py"),
    "sampled-clustermap": Path("code/run_sampled_cluster_clustermap.py"),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wrapper for retained loop clustering and UMAP workflows.",
        usage="run_loops_cluster.py <subcommand> [script args ...]",
    )
    parser.add_argument("subcommand", choices=sorted(SUBCOMMAND_TO_SCRIPT))
    parser.add_argument(
        "--workflow-root",
        default=str(WORKFLOW_ROOT),
        help="Override the copied loops_cluster workflow root.",
    )
    args, remainder = parser.parse_known_args()

    workflow_root = Path(args.workflow_root)
    script_path = workflow_root / SUBCOMMAND_TO_SCRIPT[args.subcommand]
    if not script_path.exists():
        raise FileNotFoundError(f"Workflow entry point not found: {script_path}")
    run_cwd = workflow_root
    cmd = [sys.executable, str(script_path), *remainder]
    raise SystemExit(subprocess.run(cmd, cwd=run_cwd).returncode)


if __name__ == "__main__":
    main()
