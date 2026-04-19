#!/usr/bin/env python3

import argparse

from _common import parse_chromosomes, parse_devices, parse_feature_spec
from hiread.training.stage1 import train_stage1


def main():
    parser = argparse.ArgumentParser(description="Train the stage-1 Hi-READ model.")
    parser.add_argument("--data-root", dest="data_roots", nargs="+", required=True, help="One or more stage-1 dataset roots. Each root should contain dna_sequence/, centrotelo.bed and <celltype>/ subdirectories.")
    parser.add_argument("--assembly", default="hg38")
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--feature", dest="features", action="append", default=None, help="Genomic feature spec in NAME=FILE[:NORM] format, relative to <data-root>/<celltype>/genomic_features. Default: CHIP=chip.bw:none")
    parser.add_argument("--train-chromosomes", default=None, help="Comma-separated chromosome list for training.")
    parser.add_argument("--val-chromosomes", default="chr10")
    parser.add_argument("--test-chromosomes", default="chr15")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1023)
    parser.add_argument("--model-type", default="ConvTransModel")
    parser.add_argument("--num-genomic-features", type=int, default=1)
    parser.add_argument("--mid-hidden", type=int, default=256)
    parser.add_argument("--pos-embedding", default="EPEG", choices=["EPEG", "PEG", "PPEG", "NONE"])
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--ssim-start-epoch", type=int, default=10)
    parser.add_argument("--ssim-max-epoch", type=int, default=30)
    parser.add_argument("--ssim-max-weight", type=float, default=0.15)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    feature_specs = [parse_feature_spec(item) for item in (args.features or ["CHIP=chip.bw:none"])]
    best_checkpoint = train_stage1(
        dataset_roots=args.data_roots,
        assembly=args.assembly,
        celltype=args.celltype,
        feature_specs=feature_specs,
        output_dir=args.output_dir,
        train_chromosomes=parse_chromosomes(args.train_chromosomes),
        val_chromosomes=parse_chromosomes(args.val_chromosomes),
        test_chromosomes=parse_chromosomes(args.test_chromosomes),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        model_type=args.model_type,
        num_genomic_features=args.num_genomic_features,
        mid_hidden=args.mid_hidden,
        pos_embedding=args.pos_embedding,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ssim_start_epoch=args.ssim_start_epoch,
        ssim_max_epoch=args.ssim_max_epoch,
        ssim_max_weight=args.ssim_max_weight,
        warmup_epochs=args.warmup_epochs,
        min_learning_rate=args.min_learning_rate,
        accelerator=args.accelerator,
        devices=parse_devices(args.devices),
        precision=args.precision,
        checkpoint_path=args.resume_from,
    )
    print(best_checkpoint)


if __name__ == "__main__":
    main()
