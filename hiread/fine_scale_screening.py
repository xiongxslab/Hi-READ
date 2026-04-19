#!/usr/bin/env python3

import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from hiread.inference import screening
from hiread.inference.utils import inference_utils as infer
from hiread.inference.utils import model_utils


WINDOW_SIZE = 2_097_152


def parse_region_spec(spec, tfs="NA"):
    chromosome, coordinates = spec.split(":")
    start, end = coordinates.split("-")
    tfs = tfs or "NA"
    tf_list = [item.strip() for item in tfs.split(",") if item.strip()]
    return {
        "chr": chromosome,
        "start": int(start),
        "end": int(end),
        "tf_count": len(tf_list),
        "tfs": ",".join(tf_list) if tf_list else "NA",
    }


class FineScaleScreening:
    def __init__(self, model_path, seq_path, chip_path, celltype, output_dir, deletion_size=128, step_size=128):
        self.model_path = model_path
        self.seq_path = seq_path
        self.chip_path = chip_path
        self.celltype = celltype
        self.output_dir = output_dir
        self.deletion_size = deletion_size
        self.step_size = step_size
        self.model = None
        self.seq = None
        self.chip = None
        self.loaded_chr = None
        self.results = []
        os.makedirs(self.output_dir, exist_ok=True)

    def load_model_and_data(self, chr_name):
        if self.model is None:
            self.model = model_utils.load_default(self.model_path)
            self.model.eval()

        if self.loaded_chr == chr_name:
            return

        seq_file_path = os.path.join(self.seq_path, f"{chr_name}.fa.gz")
        self.seq, self.chip = infer.load_data_default(chr_name, seq_file_path, self.chip_path)
        self.loaded_chr = chr_name

    def _get_original_prediction(self, chr_name, pred_start):
        end = pred_start + WINDOW_SIZE
        seq_region, chip_region = infer.get_data_at_interval(chr_name, pred_start, end, self.seq, self.chip)
        inputs = screening.preprocess_prediction(chr_name, pred_start, seq_region, chip_region)
        with torch.no_grad():
            return self.model(inputs)[0].detach().cpu().numpy()

    def _get_prediction_with_deletion(self, chr_name, pred_start, deletion_start):
        _, pred_deletion, _ = screening.predict_difference(
            chr_name,
            pred_start,
            deletion_start,
            self.deletion_size,
            self.model,
            self.seq,
            self.chip,
        )
        return pred_deletion

    @staticmethod
    def _calculate_impact_score(original, perturbed):
        return float(np.mean(np.abs(original - perturbed)))

    def screen_region_fine(self, chr_name, region_start, region_end, tf_info):
        self.load_model_and_data(chr_name)
        pred_start = max(0, region_start + self.deletion_size // 2 - WINDOW_SIZE // 2)
        original_prediction = self._get_original_prediction(chr_name, pred_start)

        region_results = []
        positions = range(region_start, region_end - self.deletion_size + 1, self.step_size)
        for deletion_start in tqdm(list(positions), desc=f"Fine screening {chr_name}:{region_start}-{region_end}"):
            perturbed_prediction = self._get_prediction_with_deletion(chr_name, pred_start, deletion_start)
            impact_score = self._calculate_impact_score(original_prediction, perturbed_prediction)
            region_results.append(
                {
                    "region_id": f"{chr_name}:{region_start}-{region_end}",
                    "chr": chr_name,
                    "region_start": region_start,
                    "region_end": region_end,
                    "deletion_start": deletion_start,
                    "deletion_end": deletion_start + self.deletion_size,
                    "deletion_center": deletion_start + self.deletion_size // 2,
                    "relative_position": deletion_start - region_start,
                    "impact_score": impact_score,
                    "tf_count": tf_info["tf_count"],
                    "tfs": tf_info["tfs"],
                    "deletion_size": self.deletion_size,
                }
            )
        return region_results

    def run_fine_screening(self, target_regions):
        all_results = []
        for region in target_regions:
            region_results = self.screen_region_fine(
                chr_name=region["chr"],
                region_start=region["start"],
                region_end=region["end"],
                tf_info={"tf_count": region["tf_count"], "tfs": region["tfs"]},
            )
            all_results.extend(region_results)
        self.results = all_results
        if not self.results:
            raise RuntimeError("No fine-screening results were generated.")
        self.save_results()
        self.generate_analysis_report()
        self.create_visualizations()
        return self.results

    def save_results(self):
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(os.path.join(self.output_dir, "fine_screening_results.csv"), index=False)
        with open(os.path.join(self.output_dir, "fine_screening_results.json"), "w", encoding="utf-8") as handle:
            json.dump(self.results, handle, indent=2)

        for region_id in results_df["region_id"].unique():
            region_data = results_df[results_df["region_id"] == region_id]
            top_positions = region_data.nlargest(10, "impact_score")
            region_key = region_id.replace(":", "_").replace("-", "_")
            top_positions.to_csv(os.path.join(self.output_dir, f"top10_positions_{region_key}.csv"), index=False)

    def generate_analysis_report(self):
        results_df = pd.DataFrame(self.results)
        report_path = os.path.join(self.output_dir, "fine_screening_report.txt")
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write("Fine-scale screening report\n")
            handle.write("=" * 80 + "\n")
            handle.write(f"Generated: {datetime.now().isoformat()}\n")
            handle.write(f"Regions: {results_df['region_id'].nunique()}\n")
            handle.write(f"Sites evaluated: {len(results_df)}\n")
            handle.write(f"Deletion size: {self.deletion_size} bp\n")
            handle.write(f"Step size: {self.step_size} bp\n\n")
            for region_id in results_df["region_id"].unique():
                region_data = results_df[results_df["region_id"] == region_id]
                top_site = region_data.nlargest(1, "impact_score").iloc[0]
                handle.write(f"{region_id}\n")
                handle.write(f"  TFs: {region_data['tfs'].iloc[0]}\n")
                handle.write(f"  Mean impact score: {region_data['impact_score'].mean():.6f}\n")
                handle.write(f"  Max impact score: {region_data['impact_score'].max():.6f}\n")
                handle.write(
                    f"  Top site: {int(top_site['deletion_start'])}-{int(top_site['deletion_end'])} "
                    f"(score={top_site['impact_score']:.6f})\n\n"
                )

    def create_visualizations(self):
        results_df = pd.DataFrame(self.results)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes[0, 0].hist(results_df["impact_score"], bins=50, color="steelblue", alpha=0.8)
        axes[0, 0].set_title("Impact Score Distribution")
        axes[0, 0].set_xlabel("Impact score")
        axes[0, 0].set_ylabel("Count")

        grouped = [results_df[results_df["region_id"] == region_id]["impact_score"] for region_id in results_df["region_id"].unique()]
        labels = [region_id.split(":")[1] for region_id in results_df["region_id"].unique()]
        axes[0, 1].boxplot(grouped, labels=labels, patch_artist=True)
        axes[0, 1].set_title("Impact Scores by Region")
        axes[0, 1].tick_params(axis="x", rotation=45)

        for region_id in results_df["region_id"].unique():
            region_data = results_df[results_df["region_id"] == region_id]
            axes[1, 0].plot(region_data["relative_position"], region_data["impact_score"], label=region_id)
        axes[1, 0].set_title("Position-wise Impact Scores")
        axes[1, 0].set_xlabel("Relative position (bp)")
        axes[1, 0].set_ylabel("Impact score")
        axes[1, 0].legend(fontsize=8)

        top_df = results_df.sort_values(["region_id", "impact_score"], ascending=[True, False]).groupby("region_id").head(10)
        pivot = top_df.pivot_table(index="region_id", columns=top_df.groupby("region_id").cumcount(), values="impact_score", fill_value=0.0)
        heatmap = axes[1, 1].imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrRd")
        axes[1, 1].set_title("Top 10 Impact Scores per Region")
        axes[1, 1].set_yticks(range(len(pivot.index)))
        axes[1, 1].set_yticklabels(list(pivot.index))
        axes[1, 1].set_xticks(range(pivot.shape[1]))
        axes[1, 1].set_xticklabels([f"Top {idx + 1}" for idx in range(pivot.shape[1])], rotation=45)
        fig.colorbar(heatmap, ax=axes[1, 1], fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "fine_screening_analysis.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)


def load_regions(args):
    if args.regions_csv:
        df = pd.read_csv(args.regions_csv)
        required = {"chr", "start", "end"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in regions CSV: {sorted(missing)}")
        if "tfs" not in df.columns:
            df["tfs"] = "NA"
        if "tf_count" not in df.columns:
            df["tf_count"] = df["tfs"].fillna("").apply(lambda value: len([item for item in str(value).split(",") if item.strip()]))
        return df.to_dict("records")

    if not args.region:
        raise ValueError("Provide either --regions-csv or at least one --region value.")

    labels = args.tfs or []
    while len(labels) < len(args.region):
        labels.append("NA")
    return [parse_region_spec(region_spec, labels[idx]) for idx, region_spec in enumerate(args.region)]


def main():
    parser = argparse.ArgumentParser(description="Run fine-scale screening on one or more candidate regions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seq", dest="seq_path", required=True, help="Directory containing chromosome FASTA files (*.fa.gz).")
    parser.add_argument("--chip", dest="chip_path", required=True)
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--out", dest="output_dir", required=True)
    parser.add_argument("--regions-csv", default=None, help="CSV with columns chr,start,end and optional tf_count,tfs.")
    parser.add_argument("--region", action="append", default=None, help="Inline region spec: chr:start-end")
    parser.add_argument("--tfs", action="append", default=None, help="Optional TF label matching each --region entry.")
    parser.add_argument("--deletion-size", type=int, default=128)
    parser.add_argument("--step-size", type=int, default=128)
    args = parser.parse_args()

    target_regions = load_regions(args)
    screener = FineScaleScreening(
        model_path=args.model,
        seq_path=args.seq_path,
        chip_path=args.chip_path,
        celltype=args.celltype,
        output_dir=args.output_dir,
        deletion_size=args.deletion_size,
        step_size=args.step_size,
    )
    screener.run_fine_screening(target_regions)


if __name__ == "__main__":
    main()
