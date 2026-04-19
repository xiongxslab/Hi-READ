#!/usr/bin/env python3

import argparse

from _common import REPO_ROOT
from hiread.inference.prediction import single_prediction


def main():
    parser = argparse.ArgumentParser(description="Run stage-1 prediction for a single genomic window.")
    parser.add_argument("--model", required=True, help="Stage-1 checkpoint path.")
    parser.add_argument("--seq-dir", required=True, help="Directory containing chromosome FASTA files (*.fa.gz).")
    parser.add_argument("--chip-bw", required=True, help="ChIP-seq or other genomic feature bigWig used by the stage-1 model.")
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--chr", dest="chr_name", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "stage1_prediction"))
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the input sequence before prediction.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for sequence shuffling.")
    args = parser.parse_args()

    single_prediction(
        output_path=args.out,
        celltype=args.celltype,
        chr_name=args.chr_name,
        start=args.start,
        model_path=args.model,
        seq_path=args.seq_dir,
        chip_path=args.chip_bw,
        shuffle_sequence_flag=args.shuffle,
        shuffle_seed=args.seed,
    )


if __name__ == "__main__":
    main()
