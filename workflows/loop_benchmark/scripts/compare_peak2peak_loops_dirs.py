#!/usr/bin/env python3
"""
Compare performance of two peak2peak loops directories.
Parameters consistent with LoopsPerformanceEvaluator, but directly processes .bed loops files.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FILE_MAP = {
    "h9_CTCF_NT_loops.bed": "h9_CTCF_loops.bed",
    "KLF4_loops.bed": "KLF4_loops.bed",
    "NANOG_loops.bed": "NANOG_loops.bed",
    "OCT4_loops.bed": "OCT4_loops.bed",
    "Rad21_loops.bed": "Rad21_loops.bed",
    "NTKO_loops.bed": "NTKO_loops.bed",
    "TKO_loops.bed": "TKO_loops.bed",
}


class Peak2PeakLoopsEvaluator:
    def __init__(
        self,
        ground_truth_dir,
        predicted_dir,
        output_dir,
        tolerance=50000,
        min_distance=20000,
        top_percentage=100.0,
        filter_method="sumCC",
        chromosome=None,
        file_map=None,
    ):
        self.ground_truth_dir = Path(ground_truth_dir)
        self.predicted_dir = Path(predicted_dir)
        self.output_dir = Path(output_dir)
        self.tolerance = tolerance
        self.min_distance = min_distance
        self.top_percentage = top_percentage
        self.filter_method = filter_method
        self.chromosome = chromosome
        self.file_map = file_map or DEFAULT_FILE_MAP
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_loops_file(self, file_path):
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        sample_df = pd.read_csv(file_path, sep="\t", nrows=5, header=None)
        has_header = False
        if sample_df.shape[0] > 0:
            first_row = sample_df.iloc[0]
            if any(
                isinstance(val, str) and not str(val).startswith("chr")
                for val in first_row.iloc[1:6]
            ):
                has_header = True

        if has_header:
            loops_df = pd.read_csv(file_path, sep="\t", header=0)
            col_mapping = {}
            for i, col in enumerate(loops_df.columns[:6]):
                col_mapping[col] = ["chr1", "s1", "e1", "chr2", "s2", "e2"][i]
            loops_df = loops_df.rename(columns=col_mapping)
        else:
            loops_df = pd.read_csv(file_path, sep="\t", header=None)
            loops_df.columns = ["chr1", "s1", "e1", "chr2", "s2", "e2"] + [
                f"col_{i}" for i in range(6, loops_df.shape[1])
            ]

        for col in ["s1", "e1", "s2", "e2"]:
            loops_df[col] = pd.to_numeric(loops_df[col], errors="coerce")
        loops_df = loops_df.dropna(subset=["s1", "e1", "s2", "e2"]).copy()
        loops_df = loops_df[loops_df["chr1"] == loops_df["chr2"]].copy()

        if self.chromosome is not None:
            loops_df = loops_df[loops_df["chr1"] == self.chromosome].copy()

        loops_df["distance"] = (loops_df["s2"] - loops_df["s1"]).abs()
        loops_df = loops_df[loops_df["distance"] >= self.min_distance].copy()

        if self.top_percentage < 100.0 and len(loops_df) > 0:
            if self.filter_method == "sumCC" and "sumCC" in loops_df.columns:
                loops_df = loops_df.sort_values("sumCC", ascending=False)
                keep_count = max(1, int(len(loops_df) * self.top_percentage / 100.0))
                loops_df = loops_df.head(keep_count).copy()
            elif self.filter_method == "qvalue":
                qvalue_col = None
                for col in loops_df.columns:
                    lower = str(col).lower()
                    if "q-value" in lower or "qvalue" in lower:
                        qvalue_col = col
                        break
                if qvalue_col:
                    loops_df = loops_df.sort_values(qvalue_col, ascending=True)
                    keep_count = max(1, int(len(loops_df) * self.top_percentage / 100.0))
                    loops_df = loops_df.head(keep_count).copy()

        if len(loops_df) == 0:
            return pd.DataFrame(columns=["chr1", "center1", "center2", "distance"])

        loops_df["center1"] = ((loops_df["s1"] + loops_df["e1"]) // 2).astype(int)
        loops_df["center2"] = ((loops_df["s2"] + loops_df["e2"]) // 2).astype(int)
        coords = np.sort(loops_df[["center1", "center2"]].to_numpy(dtype=np.int64), axis=1)
        loops_df["center1"] = coords[:, 0]
        loops_df["center2"] = coords[:, 1]
        return loops_df[["chr1", "center1", "center2", "distance"]].reset_index(drop=True)

    def _match_loops(self, gt_loops, pred_loops):
        if len(gt_loops) == 0 or len(pred_loops) == 0:
            return {
                "true_positives": 0,
                "false_positives": len(pred_loops),
                "false_negatives": len(gt_loops),
            }

        gt_matched = set()
        pred_matched = set()
        bucket_size = max(1, self.tolerance)

        for chrom in sorted(set(gt_loops["chr1"]).intersection(set(pred_loops["chr1"]))):
            gt_chr = gt_loops[gt_loops["chr1"] == chrom].copy().reset_index()
            pred_chr = pred_loops[pred_loops["chr1"] == chrom].copy().reset_index()

            buckets = defaultdict(list)
            pred_rows = pred_chr.to_dict("records")
            for local_idx, row in enumerate(pred_rows):
                bx = row["center1"] // bucket_size
                by = row["center2"] // bucket_size
                buckets[(bx, by)].append(local_idx)

            for gt_row in gt_chr.to_dict("records"):
                gx = gt_row["center1"] // bucket_size
                gy = gt_row["center2"] // bucket_size
                best_local_idx = None
                best_distance = None

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for local_idx in buckets.get((gx + dx, gy + dy), []):
                            pred_global_idx = int(pred_rows[local_idx]["index"])
                            if pred_global_idx in pred_matched:
                                continue

                            distance = max(
                                abs(int(gt_row["center1"]) - int(pred_rows[local_idx]["center1"])),
                                abs(int(gt_row["center2"]) - int(pred_rows[local_idx]["center2"])),
                            )
                            if distance > self.tolerance:
                                continue

                            if best_distance is None or distance < best_distance:
                                best_distance = distance
                                best_local_idx = local_idx

                if best_local_idx is not None:
                    gt_matched.add(int(gt_row["index"]))
                    pred_matched.add(int(pred_rows[best_local_idx]["index"]))

        tp = len(gt_matched)
        fp = len(pred_loops) - len(pred_matched)
        fn = len(gt_loops) - len(gt_matched)
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        }

    @staticmethod
    def _calculate_metrics(overlap_result):
        tp = overlap_result["true_positives"]
        fp = overlap_result["false_positives"]
        fn = overlap_result["false_negatives"]
        total_gt = tp + fn
        total_pred = tp + fp
        precision = tp / total_pred if total_pred > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_ground_truth": total_gt,
            "total_predicted": total_pred,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "jaccard_index": jaccard,
        }

    def _evaluate_single_pair(self, gt_filename, pred_filename):
        gt_loops = self._load_loops_file(self.ground_truth_dir / gt_filename)
        pred_loops = self._load_loops_file(self.predicted_dir / pred_filename)
        overlap = self._match_loops(gt_loops, pred_loops)
        metrics = self._calculate_metrics(overlap)
        metrics["ground_truth_file"] = gt_filename
        metrics["predicted_file"] = pred_filename
        return metrics

    def run(self):
        per_file_metrics = []
        for gt_filename, pred_filename in self.file_map.items():
            gt_path = self.ground_truth_dir / gt_filename
            pred_path = self.predicted_dir / pred_filename
            if not gt_path.exists() or not pred_path.exists():
                continue
            per_file_metrics.append(self._evaluate_single_pair(gt_filename, pred_filename))

        total_tp = sum(m["true_positives"] for m in per_file_metrics)
        total_fp = sum(m["false_positives"] for m in per_file_metrics)
        total_fn = sum(m["false_negatives"] for m in per_file_metrics)
        total_gt = sum(m["total_ground_truth"] for m in per_file_metrics)
        total_pred = sum(m["total_predicted"] for m in per_file_metrics)

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
        micro_jaccard = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

        macro_precision = float(np.mean([m["precision"] for m in per_file_metrics])) if per_file_metrics else 0.0
        macro_recall = float(np.mean([m["recall"] for m in per_file_metrics])) if per_file_metrics else 0.0
        macro_f1 = float(np.mean([m["f1_score"] for m in per_file_metrics])) if per_file_metrics else 0.0
        macro_jaccard = float(np.mean([m["jaccard_index"] for m in per_file_metrics])) if per_file_metrics else 0.0

        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "parameters": {
                "ground_truth_dir": str(self.ground_truth_dir),
                "predicted_dir": str(self.predicted_dir),
                "output_dir": str(self.output_dir),
                "tolerance_bp": self.tolerance,
                "min_distance_bp": self.min_distance,
                "top_percentage": self.top_percentage,
                "filter_method": self.filter_method,
                "chromosome": self.chromosome,
            },
            "overall_performance": {
                "total_files_evaluated": len(per_file_metrics),
                "total_ground_truth_loops": int(total_gt),
                "total_predicted_loops": int(total_pred),
                "total_true_positives": int(total_tp),
                "total_false_positives": int(total_fp),
                "total_false_negatives": int(total_fn),
                "micro_average": {
                    "precision": float(micro_precision),
                    "recall": float(micro_recall),
                    "f1_score": float(micro_f1),
                    "jaccard_index": float(micro_jaccard),
                },
                "macro_average": {
                    "precision": macro_precision,
                    "recall": macro_recall,
                    "f1_score": macro_f1,
                    "jaccard_index": macro_jaccard,
                },
            },
            "per_file_metrics": per_file_metrics,
        }

        with open(self.output_dir / "detailed_performance_results.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        pd.DataFrame(per_file_metrics).to_csv(self.output_dir / "per_file_performance.csv", index=False)
        return summary


def main():
    parser = argparse.ArgumentParser(description="比较peak2peak loops目录")
    parser.add_argument("--ground-truth-dir", required=True)
    parser.add_argument("--predicted-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tolerance", type=int, default=50000)
    parser.add_argument("--min-distance", type=int, default=20000)
    parser.add_argument("--top-percentage", type=float, default=100.0)
    parser.add_argument("--filter-method", choices=["sumCC", "qvalue"], default="sumCC")
    parser.add_argument("--chromosome", default=None)
    args = parser.parse_args()

    evaluator = Peak2PeakLoopsEvaluator(
        ground_truth_dir=args.ground_truth_dir,
        predicted_dir=args.predicted_dir,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        min_distance=args.min_distance,
        top_percentage=args.top_percentage,
        filter_method=args.filter_method,
        chromosome=args.chromosome,
    )
    summary = evaluator.run()
    print(json.dumps(summary["overall_performance"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
