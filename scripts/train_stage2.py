#!/usr/bin/env python3

import argparse
from pathlib import Path

from _common import REPO_ROOT, default_stage2_paths, parse_chromosomes, parse_devices
from hiread_diffusion.stage2 import train_stage2


def main():
    parser = argparse.ArgumentParser(description="Train the stage-2 diffusion model from a stage-1 checkpoint.")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="Stage-2 dataset root containing dna_sequence/ and <celltype>/ subdirectories.")
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--seq-dir", default=None)
    parser.add_argument("--hic-dir", default=None)
    parser.add_argument("--bed-exclude", default=str(REPO_ROOT / "hiread_diffusion" / "exclude_regions.bed"))
    parser.add_argument("--train-chromosomes", default=None)
    parser.add_argument("--val-chromosomes", default="chr10")
    parser.add_argument("--test-chromosomes", default="chr15")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--slide-size", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--val-check-interval", type=float, default=0.25)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    inferred_paths = default_stage2_paths(args.data_root, args.celltype)
    seq_dir = args.seq_dir or inferred_paths["seq_dir"]
    hic_dir = args.hic_dir or inferred_paths["hic_dir"]

    chip_path = Path(inferred_paths["chip_bw"])
    feature_specs = [("CHIP", str(chip_path), None)] if chip_path.exists() else []

    best_checkpoint = train_stage2(
        stage1_checkpoint_path=args.stage1_checkpoint,
        seq_dir=seq_dir,
        hic_dir=hic_dir,
        bed_exclude=args.bed_exclude,
        feature_specs=feature_specs,
        output_dir=args.output_dir,
        train_chromosomes=parse_chromosomes(args.train_chromosomes),
        val_chromosomes=parse_chromosomes(args.val_chromosomes),
        test_chromosomes=parse_chromosomes(args.test_chromosomes),
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        slide_size=args.slide_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        min_learning_rate=args.min_learning_rate,
        gradient_clip_val=args.gradient_clip_val,
        val_check_interval=args.val_check_interval,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        checkpoint_path=args.resume_from,
    )
    print(best_checkpoint)


if __name__ == "__main__":
    main()
