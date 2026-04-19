#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

from _common import REPO_ROOT, default_stage2_paths, parse_chromosomes, parse_devices, parse_feature_spec
from hiread.training.stage1 import train_stage1
from hiread_diffusion.stage2 import train_stage2


def main():
    parser = argparse.ArgumentParser(description="Run one-click two-stage training: stage 1 first, then stage 2 using the best stage-1 checkpoint.")
    parser.add_argument("--stage1-data-root", dest="stage1_data_roots", nargs="+", required=True, help="One or more stage-1 dataset roots.")
    parser.add_argument("--stage2-data-root", default=None, help="Stage-2 dataset root. Defaults to the first stage-1 root.")
    parser.add_argument("--assembly", default="hg38")
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--stage1-feature", dest="stage1_features", action="append", default=None, help="Stage-1 feature spec in NAME=FILE[:NORM] format. Default: CHIP=chip.bw:none")
    parser.add_argument("--train-chromosomes", default=None)
    parser.add_argument("--val-chromosomes", default="chr10")
    parser.add_argument("--test-chromosomes", default="chr15")
    parser.add_argument("--bed-exclude", default=str(REPO_ROOT / "hiread_diffusion" / "exclude_regions.bed"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage1-batch-size", type=int, default=3)
    parser.add_argument("--stage2-batch-size", type=int, default=3)
    parser.add_argument("--stage2-val-batch-size", type=int, default=1)
    parser.add_argument("--stage1-num-workers", type=int, default=16)
    parser.add_argument("--stage1-max-epochs", type=int, default=100)
    parser.add_argument("--stage2-max-epochs", type=int, default=100)
    parser.add_argument("--stage1-patience", type=int, default=50)
    parser.add_argument("--stage2-patience", type=int, default=50)
    parser.add_argument("--stage2-slide-size", type=int, default=500000)
    parser.add_argument("--stage1-learning-rate", type=float, default=1e-3)
    parser.add_argument("--stage1-weight-decay", type=float, default=0.1)
    parser.add_argument("--stage1-pos-embedding", default="EPEG", choices=["EPEG", "PEG", "PPEG", "NONE"])
    parser.add_argument("--stage1-ssim-start-epoch", type=int, default=10)
    parser.add_argument("--stage1-ssim-max-epoch", type=int, default=30)
    parser.add_argument("--stage1-ssim-max-weight", type=float, default=0.15)
    parser.add_argument("--stage1-warmup-epochs", type=int, default=10)
    parser.add_argument("--stage1-min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--stage2-learning-rate", type=float, default=1e-4)
    parser.add_argument("--stage2-weight-decay", type=float, default=0.01)
    parser.add_argument("--stage2-min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--stage2-gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--stage2-val-check-interval", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    args = parser.parse_args()

    stage1_features = [parse_feature_spec(item) for item in (args.stage1_features or ["CHIP=chip.bw:none"])]
    stage2_data_root = args.stage2_data_root or args.stage1_data_roots[0]
    inferred_stage2 = default_stage2_paths(stage2_data_root, args.celltype)

    default_chip = Path(inferred_stage2["chip_bw"])
    stage2_features = [("CHIP", str(default_chip), None)] if default_chip.exists() else []

    train_chromosomes = parse_chromosomes(args.train_chromosomes)
    val_chromosomes = parse_chromosomes(args.val_chromosomes)
    test_chromosomes = parse_chromosomes(args.test_chromosomes)
    devices = parse_devices(args.devices)

    stage1_output_dir = os.path.join(args.output_dir, "stage1")
    stage2_output_dir = os.path.join(args.output_dir, "stage2")
    os.makedirs(args.output_dir, exist_ok=True)

    stage1_checkpoint = train_stage1(
        dataset_roots=args.stage1_data_roots,
        assembly=args.assembly,
        celltype=args.celltype,
        feature_specs=stage1_features,
        output_dir=stage1_output_dir,
        train_chromosomes=train_chromosomes,
        val_chromosomes=val_chromosomes,
        test_chromosomes=test_chromosomes,
        batch_size=args.stage1_batch_size,
        num_workers=args.stage1_num_workers,
        max_epochs=args.stage1_max_epochs,
        patience=args.stage1_patience,
        seed=args.seed,
        pos_embedding=args.stage1_pos_embedding,
        learning_rate=args.stage1_learning_rate,
        weight_decay=args.stage1_weight_decay,
        ssim_start_epoch=args.stage1_ssim_start_epoch,
        ssim_max_epoch=args.stage1_ssim_max_epoch,
        ssim_max_weight=args.stage1_ssim_max_weight,
        warmup_epochs=args.stage1_warmup_epochs,
        min_learning_rate=args.stage1_min_learning_rate,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
    )

    stage2_checkpoint = train_stage2(
        stage1_checkpoint_path=stage1_checkpoint,
        seq_dir=inferred_stage2["seq_dir"],
        hic_dir=inferred_stage2["hic_dir"],
        bed_exclude=args.bed_exclude,
        feature_specs=stage2_features,
        output_dir=stage2_output_dir,
        train_chromosomes=train_chromosomes,
        val_chromosomes=val_chromosomes,
        test_chromosomes=test_chromosomes,
        batch_size=args.stage2_batch_size,
        val_batch_size=args.stage2_val_batch_size,
        max_epochs=args.stage2_max_epochs,
        patience=args.stage2_patience,
        slide_size=args.stage2_slide_size,
        seed=args.seed,
        learning_rate=args.stage2_learning_rate,
        weight_decay=args.stage2_weight_decay,
        min_learning_rate=args.stage2_min_learning_rate,
        gradient_clip_val=args.stage2_gradient_clip_val,
        val_check_interval=args.stage2_val_check_interval,
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
    )

    summary = {
        "stage1_checkpoint": stage1_checkpoint,
        "stage2_checkpoint": stage2_checkpoint,
        "stage1_data_roots": args.stage1_data_roots,
        "stage2_data_root": stage2_data_root,
        "celltype": args.celltype,
        "assembly": args.assembly,
        "train_chromosomes": train_chromosomes,
        "val_chromosomes": val_chromosomes,
        "test_chromosomes": test_chromosomes,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
