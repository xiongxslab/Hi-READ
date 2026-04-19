#!/usr/bin/env python3
"""
Run all-to-all FitHiChIP loop calling on h9_CTCF_NT / KLF4 / NANOG / OCT4 / Rad21
across four methods (real, corigami, hicdiffusion, RRTdiffusion) on chr15 npy files,
and compare prediction performance against real data.

Conventions:
- All tasks process only chr15 npy files
- All tasks use all2all (IntType=4)
- RRTdiffusion uses only chr15*_final.npy files
- Comparison parameters follow established evaluation criteria:
  tolerance=50000, min_distance=20000, top_percentage=100, filter_method=sumCC, chromosome=chr15
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = WORKFLOW_ROOT.parents[2]
INPUT_ROOT = Path(os.environ.get("HIREAD_LOOP_BENCH_INPUT_ROOT", RELEASE_ROOT / "release_assets" / "loop_benchmark" / "inputs"))
FIT_HICHIP_PYTHON = os.environ.get("HIREAD_FITHICHIP_PYTHON", sys.executable)
CALLER_SCRIPT = os.environ.get("HIREAD_LOOP_CALLER_SCRIPT", str(WORKFLOW_ROOT / "scripts" / "universal_fithichip_caller_TF.py"))
COMPARE_SCRIPT = os.environ.get("HIREAD_LOOP_COMPARE_SCRIPT", str(WORKFLOW_ROOT / "scripts" / "compare_peak2peak_loops_dirs.py"))
MACS2_OUTPUT_DIR = os.environ.get("HIREAD_MACS2_OUTPUT_DIR", str(INPUT_ROOT / "macs2_output"))
DEFAULT_EXPERIMENT_ROOT = str(
    Path(
        os.environ.get(
            "HIREAD_LOOP_BENCH_EXPERIMENT_ROOT",
            RELEASE_ROOT / "release_assets" / "loop_benchmark" / "all2all_five_datasets_chr15_20260325",
        )
    )
)

DATASETS = ["h9_CTCF_NT", "KLF4", "NANOG", "OCT4", "Rad21"]
REAL_ROOT = INPUT_ROOT / "real"
CORIGAMI_ROOT = INPUT_ROOT / "corigami"
HICDIFFUSION_ROOT = INPUT_ROOT / "hicdiffusion"
RRTDIFFUSION_ROOT = INPUT_ROOT / "RRTdiffusion"

METHOD_CONFIGS = {
    "real": {
        "datasets": {
            "h9_CTCF_NT": {
                "source_npy_dir": str(REAL_ROOT / "h9_CTCF_NT" / "chr15" / "CTCF" / "Real" / "npy"),
                "subset_mode": "all_chr15",
            },
            "KLF4": {
                "source_npy_dir": str(REAL_ROOT / "KLF4" / "chr15" / "H9_hESCs" / "Real" / "npy"),
                "subset_mode": "all_chr15",
            },
            "NANOG": {
                "source_npy_dir": str(REAL_ROOT / "NANOG" / "chr15" / "H9_hESCs" / "Real" / "npy"),
                "subset_mode": "all_chr15",
            },
            "OCT4": {
                "source_npy_dir": str(REAL_ROOT / "OCT4" / "chr15" / "H9_hESCs" / "Real" / "npy"),
                "subset_mode": "all_chr15",
            },
            "Rad21": {
                "source_npy_dir": str(REAL_ROOT / "Rad21" / "chr15" / "H9_hESCs" / "Real" / "npy"),
                "subset_mode": "all_chr15",
            },
        },
    },
    "corigami": {
        "datasets": {
            "h9_CTCF_NT": {
                "source_npy_dir": str(CORIGAMI_ROOT / "h9_CTCF_NT"),
                "subset_mode": "all_chr15",
            },
            "KLF4": {
                "source_npy_dir": str(CORIGAMI_ROOT / "KLF4"),
                "subset_mode": "all_chr15",
            },
            "NANOG": {
                "source_npy_dir": str(CORIGAMI_ROOT / "NANOG"),
                "subset_mode": "all_chr15",
            },
            "OCT4": {
                "source_npy_dir": str(CORIGAMI_ROOT / "OCT4"),
                "subset_mode": "all_chr15",
            },
            "Rad21": {
                "source_npy_dir": str(CORIGAMI_ROOT / "Rad21"),
                "subset_mode": "all_chr15",
            },
        },
    },
    "hicdiffusion": {
        "datasets": {
            "h9_CTCF_NT": {
                "source_npy_dir": str(HICDIFFUSION_ROOT / "h9_CTCF_NT"),
                "subset_mode": "all_chr15",
            },
            "KLF4": {
                "source_npy_dir": str(HICDIFFUSION_ROOT / "KLF4"),
                "subset_mode": "all_chr15",
            },
            "NANOG": {
                "source_npy_dir": str(HICDIFFUSION_ROOT / "NANOG"),
                "subset_mode": "all_chr15",
            },
            "OCT4": {
                "source_npy_dir": str(HICDIFFUSION_ROOT / "OCT4"),
                "subset_mode": "all_chr15",
            },
            "Rad21": {
                "source_npy_dir": str(HICDIFFUSION_ROOT / "Rad21"),
                "subset_mode": "all_chr15",
            },
        },
    },
    "RRTdiffusion": {
        "datasets": {
            "h9_CTCF_NT": {
                "source_npy_dir": str(RRTDIFFUSION_ROOT / "h9_CTCF_NT"),
                "subset_mode": "final_only",
            },
            "KLF4": {
                "source_npy_dir": str(RRTDIFFUSION_ROOT / "KLF4"),
                "subset_mode": "final_only",
            },
            "NANOG": {
                "source_npy_dir": str(RRTDIFFUSION_ROOT / "NANOG"),
                "subset_mode": "final_only",
            },
            "OCT4": {
                "source_npy_dir": str(RRTDIFFUSION_ROOT / "OCT4"),
                "subset_mode": "final_only",
            },
            "Rad21": {
                "source_npy_dir": str(RRTDIFFUSION_ROOT / "Rad21"),
                "subset_mode": "final_only",
            },
        },
    },
}

COMPARISON_LAYOUT = [
    ("h9_CTCF_NT", "h9_CTCF_NT_loops.bed", "h9_CTCF_loops.bed"),
    ("KLF4", "KLF4_loops.bed", "KLF4_loops.bed"),
    ("NANOG", "NANOG_loops.bed", "NANOG_loops.bed"),
    ("OCT4", "OCT4_loops.bed", "OCT4_loops.bed"),
    ("Rad21", "Rad21_loops.bed", "Rad21_loops.bed"),
]


class FiveDatasetsChr15MultiMethodAll2All:
    def __init__(self, experiment_root, parallel_jobs=4, npy_workers=4, force=False):
        self.experiment_root = Path(experiment_root)
        self.parallel_jobs = parallel_jobs
        self.npy_workers = npy_workers
        self.force = force

        self.subset_root = self.experiment_root / "chr15_npy_subsets"
        self.loops_root = self.experiment_root / "loops"
        self.comparison_root = self.experiment_root / "comparisons"
        self.summary_path = self.experiment_root / "experiment_summary.json"

        for path in [self.subset_root, self.loops_root, self.comparison_root]:
            path.mkdir(parents=True, exist_ok=True)

    def _subset_pattern(self, subset_mode):
        if subset_mode == "final_only":
            return "chr15*_final.npy"
        return "chr15*.npy"

    def _prepare_subset(self, method_name, dataset_name, source_dir, subset_mode):
        source_dir = Path(source_dir)
        subset_dir = self.subset_root / method_name / dataset_name

        if subset_dir.exists() and self.force:
            shutil.rmtree(subset_dir)

        subset_dir.mkdir(parents=True, exist_ok=True)
        pattern = self._subset_pattern(subset_mode)

        linked = 0
        for npy_path in sorted(source_dir.rglob(pattern)):
            target_path = subset_dir / npy_path.name
            if target_path.exists() or target_path.is_symlink():
                target_path.unlink()
            os.symlink(npy_path, target_path)
            linked += 1

        if linked == 0:
            raise RuntimeError(
                f"{method_name}/{dataset_name} 未找到 {pattern}: {source_dir}"
            )

        return str(subset_dir), linked

    def _extract_loops(self, method_name, dataset_name):
        method_output_root = self.loops_root / method_name
        results_dir = method_output_root / dataset_name / f"{dataset_name}_fithichip_results"
        target_name = f"FitHiChIP_{dataset_name}.interactions_FitHiC_Q0.01_MergeNearContacts.bed"
        matches = glob.glob(str(results_dir / "**" / target_name), recursive=True)
        if not matches:
            return None

        final_loops_dir = method_output_root / "final_loops"
        final_loops_dir.mkdir(parents=True, exist_ok=True)
        target_file = final_loops_dir / f"{dataset_name}_loops.bed"
        shutil.copy2(matches[0], target_file)
        return str(target_file)

    def _check_completed(self, method_name, dataset_name):
        method_output_root = self.loops_root / method_name
        dataset_dir = method_output_root / dataset_name
        loops_file = method_output_root / "final_loops" / f"{dataset_name}_loops.bed"
        return (
            (dataset_dir / f"{dataset_name}_genome_wide.bed").exists()
            and (dataset_dir / f"{dataset_name}_pipeline_stats.json").exists()
            and loops_file.exists()
        )

    def _run_single_task(self, method_name, dataset_name, config):
        method_output_root = self.loops_root / method_name
        output_dir = method_output_root / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"{dataset_name}_parallel_run.log"

        if not self.force and self._check_completed(method_name, dataset_name):
            return {
                "method_name": method_name,
                "dataset_name": dataset_name,
                "status": "skipped",
                "output_dir": str(output_dir),
                "log_file": str(log_file),
            }

        subset_dir, linked_count = self._prepare_subset(
            method_name=method_name,
            dataset_name=dataset_name,
            source_dir=config["source_npy_dir"],
            subset_mode=config["subset_mode"],
        )

        cmd = [
            FIT_HICHIP_PYTHON,
            CALLER_SCRIPT,
            "--npy-dir", subset_dir,
            "--output-dir", str(output_dir),
            "--dataset-name", dataset_name,
            "--interaction-type", "4",
            "--bin-size", "8192",
            "--scaling-factor", "10",
            "--low-dist-thr", "20000",
            "--upp-dist-thr", "2000000",
            "--macs2-output-dir", MACS2_OUTPUT_DIR,
            "--npy-workers", str(self.npy_workers),
        ]

        start_time = datetime.now()
        with open(log_file, "w") as log_f:
            log_f.write(f"开始时间: {start_time.isoformat()}\n")
            log_f.write(f"method: {method_name}\n")
            log_f.write(f"dataset: {dataset_name}\n")
            log_f.write(f"source_npy_dir: {config['source_npy_dir']}\n")
            log_f.write(f"subset_mode: {config['subset_mode']}\n")
            log_f.write(f"chr15 subset dir: {subset_dir}\n")
            log_f.write(f"linked npy count: {linked_count}\n")
            log_f.write("interaction_type: 4 (all2all)\n")
            log_f.write(f"命令: {' '.join(cmd)}\n\n")
            log_f.flush()

            completed = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        duration_minutes = round((datetime.now() - start_time).total_seconds() / 60.0, 2)
        loops_file = None
        if completed.returncode == 0:
            loops_file = self._extract_loops(method_name, dataset_name)

        return {
            "method_name": method_name,
            "dataset_name": dataset_name,
            "status": "success" if completed.returncode == 0 else "failed",
            "return_code": completed.returncode,
            "duration_minutes": duration_minutes,
            "output_dir": str(output_dir),
            "log_file": str(log_file),
            "loops_file": loops_file,
            "linked_chr15_npy_count": linked_count,
            "subset_dir": subset_dir,
            "subset_mode": config["subset_mode"],
            "interaction_type": 4,
        }

    def _build_tasks(self):
        tasks = []
        for method_name, method_config in METHOD_CONFIGS.items():
            for dataset_name, dataset_config in method_config["datasets"].items():
                tasks.append((method_name, dataset_name, dataset_config))
        return tasks

    def _run_all_loops(self):
        tasks = self._build_tasks()
        results = []

        print("=" * 80)
        print("5数据集 chr15 多方法 all2all FitHiChIP 开始")
        print("=" * 80)
        print(f"输出目录: {self.loops_root}")
        print(f"并行任务数: {self.parallel_jobs}")
        print(f"单任务npy并行数: {self.npy_workers}")

        with ThreadPoolExecutor(max_workers=self.parallel_jobs) as executor:
            future_to_task = {
                executor.submit(self._run_single_task, method_name, dataset_name, config): (method_name, dataset_name)
                for method_name, dataset_name, config in tasks
            }
            for future in as_completed(future_to_task):
                method_name, dataset_name = future_to_task[future]
                result = future.result()
                results.append(result)
                status = "✅" if result["status"] in {"success", "skipped"} else "❌"
                print(
                    f"{status} {method_name}/{dataset_name} "
                    f"({result.get('duration_minutes', 0)} 分钟)"
                )

        results.sort(key=lambda item: (item["method_name"], item["dataset_name"]))
        return results

    def _run_single_comparison(self, method_name):
        gt_dir = self.comparison_root / method_name / "gt"
        pred_dir = self.comparison_root / method_name / "pred"
        output_dir = self.comparison_root / method_name / "result"
        for path in [gt_dir, pred_dir, output_dir]:
            path.mkdir(parents=True, exist_ok=True)

        real_loops_root = self.loops_root / "real" / "final_loops"
        pred_loops_root = self.loops_root / method_name / "final_loops"

        for dataset_name, gt_name, pred_name in COMPARISON_LAYOUT:
            gt_link = gt_dir / gt_name
            pred_link = pred_dir / pred_name
            gt_target = real_loops_root / f"{dataset_name}_loops.bed"
            pred_target = pred_loops_root / f"{dataset_name}_loops.bed"

            if not gt_target.exists() or not pred_target.exists():
                raise RuntimeError(
                    f"comparison缺少loops文件: {method_name}/{dataset_name} "
                    f"(gt={gt_target.exists()}, pred={pred_target.exists()})"
                )

            if gt_link.exists() or gt_link.is_symlink():
                gt_link.unlink()
            if pred_link.exists() or pred_link.is_symlink():
                pred_link.unlink()

            os.symlink(gt_target, gt_link)
            os.symlink(pred_target, pred_link)

        cmd = [
            FIT_HICHIP_PYTHON,
            COMPARE_SCRIPT,
            "--ground-truth-dir", str(gt_dir),
            "--predicted-dir", str(pred_dir),
            "--output-dir", str(output_dir),
            "--tolerance", "50000",
            "--min-distance", "20000",
            "--top-percentage", "100",
            "--filter-method", "sumCC",
            "--chromosome", "chr15",
        ]
        subprocess.run(cmd, check=True)

        with open(output_dir / "detailed_performance_results.json") as f:
            summary = json.load(f)
        return {
            "method_name": method_name,
            "comparison_output_dir": str(output_dir),
            "overall_performance": summary["overall_performance"],
        }

    def run(self):
        summary = {
            "start_time": datetime.now().isoformat(),
            "experiment_root": str(self.experiment_root),
            "parameters": {
                "parallel_jobs": self.parallel_jobs,
                "npy_workers": self.npy_workers,
                "force": self.force,
                "note": "5 datasets chr15 only; all methods use all2all; RRTdiffusion uses final.npy only",
            },
            "loop_call_results": [],
            "comparison_results": [],
        }

        loop_results = self._run_all_loops()
        summary["loop_call_results"] = loop_results

        comparison_results = []
        for method_name in ["corigami", "hicdiffusion", "RRTdiffusion"]:
            comparison_results.append(self._run_single_comparison(method_name))
        summary["comparison_results"] = comparison_results

        summary["end_time"] = datetime.now().isoformat()
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(json.dumps({
            "experiment_root": str(self.experiment_root),
            "summary_path": str(self.summary_path),
        }, indent=2, ensure_ascii=False))
        return True


def main():
    parser = argparse.ArgumentParser(
        description="5数据集 chr15 多方法 all2all FitHiChIP + 对比"
    )
    parser.add_argument(
        "--experiment-root",
        default=DEFAULT_EXPERIMENT_ROOT,
        help="实验输出根目录",
    )
    parser.add_argument(
        "--parallel-jobs",
        type=int,
        default=4,
        help="同时运行的任务数",
    )
    parser.add_argument(
        "--npy-workers",
        type=int,
        default=4,
        help="每个任务内部 npy->bed 的并行进程数",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使已有结果也强制重跑",
    )
    args = parser.parse_args()

    runner = FiveDatasetsChr15MultiMethodAll2All(
        experiment_root=args.experiment_root,
        parallel_jobs=args.parallel_jobs,
        npy_workers=args.npy_workers,
        force=args.force,
    )
    success = runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
