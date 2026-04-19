#!/usr/bin/env python3
"""
Build final peak2peak / all2all result summary, parameter tables and Nature Methods style methods documentation.

Data sources:
1. Historical peak2peak results (7 datasets; NTKO/TKO are actually fallback all2all, explicitly noted in documentation)
2. New 5-dataset chr15 all2all results
3. Reusable NTKO/TKO all2all results from old directory, to replace incomplete new batch
"""

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
RELEASE_ROOT = WORKFLOW_ROOT.parents[2]
FIT_HICHIP_INPUT_ROOT = Path(
    os.environ.get(
        "HIREAD_FITHICHIP_INPUT_ROOT",
        RELEASE_ROOT / "release_assets" / "loop_benchmark" / "fithichip_inputs",
    )
)
FIVE_ALL2ALL_ROOT = Path(
    os.environ.get(
        "HIREAD_FIVE_DATASETS_ALL2ALL_ROOT",
        RELEASE_ROOT / "release_assets" / "loop_benchmark" / "all2all_five_datasets_chr15_20260325",
    )
)
COMPARE_SCRIPT = Path(
    os.environ.get(
        "HIREAD_LOOP_COMPARE_SCRIPT",
        WORKFLOW_ROOT / "scripts" / "compare_peak2peak_loops_dirs.py",
    )
)
REPORT_ROOT = Path(
    os.environ.get(
        "HIREAD_LOOP_REPORT_ROOT",
        WORKFLOW_ROOT / "reports",
    )
)
SUBSTITUTE_ROOT = REPORT_ROOT / "substitute_ntko_tko_all2all_old_results"


PEAK2PEAK_CSVS = {
    "corigami": FIT_HICHIP_INPUT_ROOT / "corigami_loops" / "chr15" / "per_file_performance.csv",
    "hicdiffusion": FIT_HICHIP_INPUT_ROOT / "hicdiffusion_loops" / "chr15" / "per_file_performance.csv",
    "RRTdiffusion": FIT_HICHIP_INPUT_ROOT / "RRTdiffusion_loops" / "loops" / "chr15" / "per_file_performance.csv",
}

ALL2ALL_FIVE_CSVS = {
    "corigami": FIVE_ALL2ALL_ROOT / "comparisons" / "corigami" / "result" / "per_file_performance.csv",
    "hicdiffusion": FIVE_ALL2ALL_ROOT / "comparisons" / "hicdiffusion" / "result" / "per_file_performance.csv",
    "RRTdiffusion": FIVE_ALL2ALL_ROOT / "comparisons" / "RRTdiffusion" / "result" / "per_file_performance.csv",
}

SUBSTITUTE_LOOP_SOURCES = {
    "real": {
        "NTKO": FIT_HICHIP_INPUT_ROOT / "real_NTKO_all2all" / "NTKO_fithichip_results" / "FitHiChIP_ALL2ALL_b8192_L20000_U2000000" / "Coverage_Bias" / "FitHiC_BiasCorr" / "Merge_Nearby_Interactions" / "FitHiChIP_NTKO.interactions_FitHiC_Q0.01_MergeNearContacts.bed",
        "TKO": FIT_HICHIP_INPUT_ROOT / "real_TKO_all2all" / "TKO_fithichip_results" / "FitHiChIP_ALL2ALL_b8192_L20000_U2000000" / "Coverage_Bias" / "FitHiC_BiasCorr" / "Merge_Nearby_Interactions" / "FitHiChIP_TKO.interactions_FitHiC_Q0.01_MergeNearContacts.bed",
    },
    "corigami": {
        "NTKO": FIT_HICHIP_INPUT_ROOT / "corigami_loops" / "corigami_NTKO_peak2peak" / "NTKO_fithichip_results" / "FitHiChIP_ALL2ALL_b8192_L20000_U2000000" / "Coverage_Bias" / "FitHiC_BiasCorr" / "Merge_Nearby_Interactions" / "FitHiChIP_NTKO.interactions_FitHiC_Q0.01_MergeNearContacts.bed",
        "TKO": FIT_HICHIP_INPUT_ROOT / "corigami_loops" / "corigami_TKO_peak2peak" / "TKO_fithichip_results" / "FitHiChIP_ALL2ALL_b8192_L20000_U2000000" / "Coverage_Bias" / "FitHiC_BiasCorr" / "Merge_Nearby_Interactions" / "FitHiChIP_TKO.interactions_FitHiC_Q0.01_MergeNearContacts.bed",
    },
    "hicdiffusion": {
        "NTKO": FIT_HICHIP_INPUT_ROOT / "hicdiffusion_loops" / "hicdiffusion_NTKO_peak2peak" / "NTKO_fithichip_results" / "FitHiChIP_ALL2ALL_b8192_L20000_U2000000" / "Coverage_Bias" / "FitHiC_BiasCorr" / "Merge_Nearby_Interactions" / "FitHiChIP_NTKO.interactions_FitHiC_Q0.01_MergeNearContacts.bed",
        "TKO": FIT_HICHIP_INPUT_ROOT / "hicdiffusion_loops" / "hicdiffusion_TKO_peak2peak" / "TKO_fithichip_results" / "FitHiChIP_ALL2ALL_b8192_L20000_U2000000" / "Coverage_Bias" / "FitHiC_BiasCorr" / "Merge_Nearby_Interactions" / "FitHiChIP_TKO.interactions_FitHiC_Q0.01_MergeNearContacts.bed",
    },
    "RRTdiffusion": {
        "NTKO": FIT_HICHIP_INPUT_ROOT / "RRTdiffusion_NTKO_all2all" / "NTKO_fithichip_results" / "FitHiChIP_ALL2ALL_b8192_L50000_U2000000" / "Coverage_Bias" / "FitHiC_BiasCorr" / "Merge_Nearby_Interactions" / "FitHiChIP_NTKO.interactions_FitHiC_Q0.01_MergeNearContacts.bed",
        "TKO": FIT_HICHIP_INPUT_ROOT / "RRTdiffusion_TKO_all2all" / "TKO_fithichip_results" / "FitHiChIP_ALL2ALL_b8192_L50000_U2000000" / "Coverage_Bias" / "FitHiC_BiasCorr" / "Merge_Nearby_Interactions" / "FitHiChIP_TKO.interactions_FitHiC_Q0.01_MergeNearContacts.bed",
    },
}

DATASET_MAP = {
    "CTCF": "h9_CTCF_NT",
    "CTCF.bed": "h9_CTCF_NT",
    "h9_CTCF_NT_loops.bed": "h9_CTCF_NT",
    "h9_CTCF_loops.bed": "h9_CTCF_NT",
    "KLF4.bed": "KLF4",
    "KLF4_loops.bed": "KLF4",
    "NANOG.bed": "NANOG",
    "NANOG_loops.bed": "NANOG",
    "OCT4.bed": "OCT4",
    "OCT4_loops.bed": "OCT4",
    "Rad21.bed": "Rad21",
    "Rad21_loops.bed": "Rad21",
    "NTKO.bed": "NTKO",
    "NTKO_loops.bed": "NTKO",
    "TKO.bed": "TKO",
    "TKO_loops.bed": "TKO",
}

DATASET_ORDER = ["h9_CTCF_NT", "KLF4", "NANOG", "OCT4", "Rad21", "NTKO", "TKO"]
FIVE_DATASETS = ["h9_CTCF_NT", "KLF4", "NANOG", "OCT4", "Rad21"]


def load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_peak2peak_loops_dirs", COMPARE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_dataset(name):
    return DATASET_MAP.get(name, name.replace("_loops.bed", "").replace(".bed", ""))


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_substitute_comparisons():
    ensure_clean_dir(SUBSTITUTE_ROOT)
    loops_root = SUBSTITUTE_ROOT / "loops"
    compare_root = SUBSTITUTE_ROOT / "comparisons"

    for method_name, datasets in SUBSTITUTE_LOOP_SOURCES.items():
        final_loops_dir = loops_root / method_name / "final_loops"
        final_loops_dir.mkdir(parents=True, exist_ok=True)
        for dataset_name, src in datasets.items():
            target = final_loops_dir / f"{dataset_name}_loops.bed"
            if target.exists() or target.is_symlink():
                target.unlink()
            os.symlink(src, target)

    compare_module = load_compare_module()
    evaluator_cls = compare_module.Peak2PeakLoopsEvaluator

    file_map = {
        "NTKO_loops.bed": "NTKO_loops.bed",
        "TKO_loops.bed": "TKO_loops.bed",
    }

    results = []
    for method_name in ["corigami", "hicdiffusion", "RRTdiffusion"]:
        output_dir = compare_root / method_name / "result"
        output_dir.mkdir(parents=True, exist_ok=True)

        evaluator = evaluator_cls(
            ground_truth_dir=loops_root / "real" / "final_loops",
            predicted_dir=loops_root / method_name / "final_loops",
            output_dir=output_dir,
            tolerance=50000,
            min_distance=20000,
            top_percentage=100.0,
            filter_method="sumCC",
            chromosome="chr15",
            file_map=file_map,
        )
        summary = evaluator.run()
        results.append({
            "method_name": method_name,
            "summary": summary,
            "csv": output_dir / "per_file_performance.csv",
            "json": output_dir / "detailed_performance_results.json",
        })

    return results


def load_metrics_csv(csv_path: Path, mode_label: str, method_name: str):
    df = pd.read_csv(csv_path)
    dataset_col = "ground_truth_file" if "ground_truth_file" in df.columns else "filename"
    df["dataset"] = df[dataset_col].map(normalize_dataset)
    df["mode"] = mode_label
    df["method"] = method_name
    df["source_csv"] = str(csv_path)
    return df[
        [
            "mode",
            "method",
            "dataset",
            "true_positives",
            "false_positives",
            "false_negatives",
            "total_ground_truth",
            "total_predicted",
            "precision",
            "recall",
            "f1_score",
            "jaccard_index",
            "source_csv",
        ]
    ].copy()


def aggregate_overall(df: pd.DataFrame, mode_label: str):
    rows = []
    for method_name, method_df in df.groupby("method"):
        tp = int(method_df["true_positives"].sum())
        fp = int(method_df["false_positives"].sum())
        fn = int(method_df["false_negatives"].sum())
        total_gt = int(method_df["total_ground_truth"].sum())
        total_pred = int(method_df["total_predicted"].sum())
        micro_precision = tp / (tp + fp) if (tp + fp) else 0.0
        micro_recall = tp / (tp + fn) if (tp + fn) else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall)
            else 0.0
        )
        micro_jaccard = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

        rows.append({
            "mode": mode_label,
            "method": method_name,
            "datasets_included": int(method_df["dataset"].nunique()),
            "total_ground_truth_loops": total_gt,
            "total_predicted_loops": total_pred,
            "total_true_positives": tp,
            "total_false_positives": fp,
            "total_false_negatives": fn,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "micro_jaccard": micro_jaccard,
            "macro_precision": float(method_df["precision"].mean()),
            "macro_recall": float(method_df["recall"].mean()),
            "macro_f1": float(method_df["f1_score"].mean()),
            "macro_jaccard": float(method_df["jaccard_index"].mean()),
        })
    return pd.DataFrame(rows)


def build_parameter_table():
    rows = [
        {
            "analysis_mode": "peak2peak_true",
            "datasets": "h9_CTCF_NT,KLF4,NANOG,OCT4,Rad21",
            "methods": "real,corigami,hicdiffusion,RRTdiffusion",
            "peak_file_status": "provided",
            "interaction_type": 1,
            "bin_size": 8192,
            "low_dist_thr": 20000,
            "upp_dist_thr": 2000000,
            "qvalue": 0.01,
            "use_p2p_background": 0,
            "bias_type": 1,
            "merge_nearby_interactions": 1,
            "notes": "True peak-constrained loop calling using MACS2 peak files",
            "config_examples": "<fithichip_results>/real_CTCF_peak2peak/CTCF_fithichip_config.conf; <fithichip_results>/hicdiffusion_CTCF_peak2peak/CTCF_fithichip_config.conf; <fithichip_results>/RRTdiffusion_CTCF_peak2peak/CTCF_fithichip_config.conf",
        },
        {
            "analysis_mode": "peak2peak_legacy_label_but_all2all_fallback",
            "datasets": "NTKO,TKO",
            "methods": "corigami,hicdiffusion",
            "peak_file_status": "empty",
            "interaction_type": 4,
            "bin_size": 8192,
            "low_dist_thr": 20000,
            "upp_dist_thr": 2000000,
            "qvalue": 0.01,
            "use_p2p_background": 0,
            "bias_type": 1,
            "merge_nearby_interactions": 1,
            "notes": "Directories were named peak2peak, but configs show #PeakFile= and IntType=4; report as all2all in strict interpretation",
            "config_examples": "<fithichip_results>/corigami_NTKO_peak2peak/NTKO_fithichip_config.conf; <fithichip_results>/hicdiffusion_NTKO_peak2peak/NTKO_fithichip_config.conf",
        },
        {
            "analysis_mode": "all2all_current_harmonized",
            "datasets": "h9_CTCF_NT,KLF4,NANOG,OCT4,Rad21",
            "methods": "real,corigami,hicdiffusion,RRTdiffusion",
            "peak_file_status": "empty",
            "interaction_type": 4,
            "bin_size": 8192,
            "low_dist_thr": 20000,
            "upp_dist_thr": 2000000,
            "qvalue": 0.01,
            "use_p2p_background": 0,
            "bias_type": 1,
            "merge_nearby_interactions": 1,
            "notes": "Current chr15 all2all benchmarking run",
            "config_examples": "<all2all_run_root>/loops/real/KLF4/KLF4_fithichip_config.conf",
        },
        {
            "analysis_mode": "all2all_substitute_old",
            "datasets": "NTKO,TKO",
            "methods": "real,corigami,hicdiffusion",
            "peak_file_status": "empty",
            "interaction_type": 4,
            "bin_size": 8192,
            "low_dist_thr": 20000,
            "upp_dist_thr": 2000000,
            "qvalue": 0.01,
            "use_p2p_background": 0,
            "bias_type": 1,
            "merge_nearby_interactions": 1,
            "notes": "Archived all2all-compatible runs used to substitute the unfinished rerun",
            "config_examples": "<fithichip_results>/real_NTKO_all2all/NTKO_fithichip_config.conf; <fithichip_results>/corigami_NTKO_peak2peak/NTKO_fithichip_config.conf; <fithichip_results>/hicdiffusion_NTKO_peak2peak/NTKO_fithichip_config.conf",
        },
        {
            "analysis_mode": "all2all_substitute_old_rrtdiffusion",
            "datasets": "NTKO,TKO",
            "methods": "RRTdiffusion",
            "peak_file_status": "empty",
            "interaction_type": 4,
            "bin_size": 8192,
            "low_dist_thr": 50000,
            "upp_dist_thr": 2000000,
            "qvalue": 0.01,
            "use_p2p_background": 0,
            "bias_type": 1,
            "merge_nearby_interactions": 1,
            "notes": "Archived RRTdiffusion all2all NTKO/TKO runs used a 50-kb lower distance threshold",
            "config_examples": "<fithichip_results>/RRTdiffusion_NTKO_all2all/NTKO_fithichip_config.conf; <fithichip_results>/RRTdiffusion_TKO_all2all/TKO_fithichip_config.conf",
        },
        {
            "analysis_mode": "evaluation",
            "datasets": "all compared loops",
            "methods": "corigami,hicdiffusion,RRTdiffusion",
            "peak_file_status": "NA",
            "interaction_type": "NA",
            "bin_size": "NA",
            "low_dist_thr": 20000,
            "upp_dist_thr": "NA",
            "qvalue": "NA",
            "use_p2p_background": "NA",
            "bias_type": "NA",
            "merge_nearby_interactions": "NA",
            "notes": "Same-chromosome loop matching, center-based one-to-one greedy matching, tolerance 50000 bp, minimum loop distance 20000 bp",
            "config_examples": "<evaluation_script>/compare_peak2peak_loops_dirs.py",
        },
    ]
    return pd.DataFrame(rows)


def build_methods_markdown(overall_peak_legacy, overall_peak_strict, overall_all2all, parameter_df):
    def df_to_md(df):
        df = df.copy()
        headers = [str(col) for col in df.columns]
        rows = []
        for _, row in df.iterrows():
            values = []
            for value in row.tolist():
                if pd.isna(value):
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:.6g}")
                else:
                    values.append(str(value))
            rows.append(values)

        sep = ["---"] * len(headers)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(sep) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    text = f"""# Loop Calling and Evaluation Summary

## Overview
We organized chromosome 15 loop-calling benchmarks for three prediction methods (`corigami`, `hicdiffusion`, and `RRTdiffusion`) against matched real data. Two analysis modes were considered: peak-to-peak and all-to-all. NTKO/TKO all-to-all-compatible fallback results can be supplied when fully harmonized reruns are not available.

## Loop Calling
Predicted or real contact maps were converted to FitHiChIP input locus-pair BED files and processed with `universal_fithichip_caller_TF.py`. For true peak-to-peak analyses, `IntType=1` and MACS2 peak files were provided for `h9_CTCF_NT`, `KLF4`, `NANOG`, `OCT4`, and `Rad21`. Common settings were `BINSIZE=8192`, `LowDistThr=20000`, `UppDistThr=2000000`, `QVALUE=0.01`, `UseP2PBackgrnd=0`, `BiasType=1`, and `MergeInt=1`.

For all-to-all analyses, `IntType=4` and `PeakFile` was left empty. The current harmonized all-to-all analysis for the five peak-enabled datasets used a 20-kb lower distance threshold. Optional NTKO/TKO fallback all-to-all results from `real`, `corigami`, and `hicdiffusion` also used 20 kb in the reference analysis, whereas `RRTdiffusion` NTKO/TKO fallback runs used a 50-kb lower distance threshold.

Historical NTKO/TKO directories under `corigami_loops/*_peak2peak` and `hicdiffusion_loops/*_peak2peak` were inspected directly. Despite their directory names, their FitHiChIP configuration files contained `#PeakFile=` and `IntType=4`, indicating that these runs were executed as all-to-all rather than true peak-to-peak analyses.

## Performance Evaluation
Loop sets were compared after restricting to intrachromosomal loops and chromosome 15. Loop anchors were represented by interval centers, and a one-to-one greedy matching procedure was applied with a positional tolerance of 50 kb. Only loops with genomic separation of at least 20 kb were evaluated. Precision, recall, F1 score, and the Jaccard index were computed from true positives, false positives, and false negatives.

## Parameter Summary
{df_to_md(parameter_df)}

## Overall Performance
### Historical Peak-to-Peak Labeling (7 datasets; NTKO/TKO are fallback all-to-all)
{df_to_md(overall_peak_legacy)}

### Strict Peak-to-Peak Interpretation (5 true peak-constrained datasets)
{df_to_md(overall_peak_strict)}

### All-to-All (7 datasets; NTKO/TKO substituted from fallback runs)
{df_to_md(overall_all2all)}

## Reporting Notes
1. The strict methodologically correct peak-to-peak set contains five datasets: `h9_CTCF_NT`, `KLF4`, `NANOG`, `OCT4`, and `Rad21`.
2. The legacy seven-dataset peak-to-peak table is preserved for continuity with earlier result directories, but `NTKO` and `TKO` should be interpreted as all-to-all fallback runs.
3. The all-to-all seven-dataset summary can be completed using fallback substitute results for `NTKO/TKO`. These values should be refreshed once fully harmonized reruns are available.
4. `RRTdiffusion` all-to-all NTKO/TKO substitute runs were generated with `LowDistThr=50000`, whereas the other substitute all-to-all runs used `LowDistThr=20000`.
"""
    return text


def main():
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    substitute_results = build_substitute_comparisons()

    peak_legacy_frames = []
    for method_name, csv_path in PEAK2PEAK_CSVS.items():
        peak_legacy_frames.append(load_metrics_csv(csv_path, "peak2peak_legacy", method_name))
    peak_legacy_df = pd.concat(peak_legacy_frames, ignore_index=True)
    peak_legacy_df["dataset"] = pd.Categorical(peak_legacy_df["dataset"], categories=DATASET_ORDER, ordered=True)
    peak_legacy_df = peak_legacy_df.sort_values(["method", "dataset"]).reset_index(drop=True)

    peak_strict_df = peak_legacy_df[peak_legacy_df["dataset"].isin(FIVE_DATASETS)].copy()

    all2all_five_frames = []
    for method_name, csv_path in ALL2ALL_FIVE_CSVS.items():
        all2all_five_frames.append(load_metrics_csv(csv_path, "all2all_substituted", method_name))
    all2all_five_df = pd.concat(all2all_five_frames, ignore_index=True)

    substitute_frames = []
    for item in substitute_results:
        substitute_frames.append(load_metrics_csv(item["csv"], "all2all_substituted", item["method_name"]))
    all2all_sub_df = pd.concat([all2all_five_df, *substitute_frames], ignore_index=True)
    all2all_sub_df["dataset"] = pd.Categorical(all2all_sub_df["dataset"], categories=DATASET_ORDER, ordered=True)
    all2all_sub_df = all2all_sub_df.sort_values(["method", "dataset"]).reset_index(drop=True)

    overall_peak_legacy = aggregate_overall(peak_legacy_df, "peak2peak_legacy_7datasets")
    overall_peak_strict = aggregate_overall(peak_strict_df, "peak2peak_true_5datasets")
    overall_all2all = aggregate_overall(all2all_sub_df, "all2all_substituted_7datasets")
    overall_all = pd.concat([overall_peak_legacy, overall_peak_strict, overall_all2all], ignore_index=True)

    parameter_df = build_parameter_table()

    peak_legacy_df.to_csv(REPORT_ROOT / "Table_S1_peak2peak_legacy_7datasets_per_dataset.csv", index=False)
    peak_strict_df.to_csv(REPORT_ROOT / "Table_S2_peak2peak_true_5datasets_per_dataset.csv", index=False)
    all2all_sub_df.to_csv(REPORT_ROOT / "Table_S3_all2all_substituted_7datasets_per_dataset.csv", index=False)
    overall_all.to_csv(REPORT_ROOT / "Table_S4_overall_metrics.csv", index=False)
    parameter_df.to_csv(REPORT_ROOT / "Table_S5_loop_calling_parameters.csv", index=False)

    loop_counts = all2all_sub_df[["method", "dataset", "total_ground_truth", "total_predicted"]].copy()
    loop_counts = loop_counts.rename(columns={
        "total_ground_truth": "ground_truth_loop_count",
        "total_predicted": "predicted_loop_count",
    })
    loop_counts.to_csv(REPORT_ROOT / "Table_S6_all2all_loop_counts.csv", index=False)

    methods_md = build_methods_markdown(
        overall_peak_legacy,
        overall_peak_strict,
        overall_all2all,
        parameter_df,
    )
    with open(REPORT_ROOT / "Nature_Methods_style_methods.md", "w") as f:
        f.write(methods_md)

    summary = {
        "report_root": str(REPORT_ROOT),
        "substitute_root": str(SUBSTITUTE_ROOT),
        "files": {
            "peak2peak_legacy_per_dataset": str(REPORT_ROOT / "Table_S1_peak2peak_legacy_7datasets_per_dataset.csv"),
            "peak2peak_true5_per_dataset": str(REPORT_ROOT / "Table_S2_peak2peak_true_5datasets_per_dataset.csv"),
            "all2all_substituted_per_dataset": str(REPORT_ROOT / "Table_S3_all2all_substituted_7datasets_per_dataset.csv"),
            "overall_metrics": str(REPORT_ROOT / "Table_S4_overall_metrics.csv"),
            "parameters": str(REPORT_ROOT / "Table_S5_loop_calling_parameters.csv"),
            "loop_counts": str(REPORT_ROOT / "Table_S6_all2all_loop_counts.csv"),
            "methods_doc": str(REPORT_ROOT / "Nature_Methods_style_methods.md"),
        },
    }
    with open(REPORT_ROOT / "report_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
