#!/usr/bin/env python3

import argparse
import tempfile
from pathlib import Path

import torch

from _common import REPO_ROOT
from hiread.training.stage1 import Stage1LightningModule
from hiread_diffusion import MultiModalDiffusion, MultiModalDiffusionEncoderDecoder


def count_params(module):
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def make_temporary_stage1_checkpoint():
    model = Stage1LightningModule()
    state_dict = {f"model.{key}": value for key, value in model.model.state_dict().items()}
    temp_dir = Path(tempfile.mkdtemp(prefix="hiread_stage1_"))
    checkpoint_path = temp_dir / "stage1.ckpt"
    torch.save({"state_dict": state_dict}, checkpoint_path)
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Collect parameter counts for the default open-source models.")
    parser.add_argument("--output", default=str(REPO_ROOT / "docs" / "PARAMETER_TABLE.md"))
    args = parser.parse_args()

    stage1_model = Stage1LightningModule()
    encoder_decoder = MultiModalDiffusionEncoderDecoder(
        validation_folder=str(REPO_ROOT / "outputs" / "tmp_validation"),
        val_chr="chr10",
        test_chr="chr15",
        num_epi_features=1,
    )

    temp_stage1_ckpt = make_temporary_stage1_checkpoint()
    stage2_model = MultiModalDiffusion(
        hic_filename="hiread_stage2",
        validation_folder=str(REPO_ROOT / "outputs" / "tmp_validation"),
        encoder_decoder_model=str(temp_stage1_ckpt),
        encoder_decoder_kind="hiread_stage1",
        val_chr="chr10",
        test_chr="chr15",
        num_epi_features=1,
    )

    rows = [
        ("Stage 1 ConvTransModel",) + count_params(stage1_model.model),
        ("Stage 2 encoder-decoder",) + count_params(encoder_decoder),
        ("Stage 2 diffusion UNet",) + count_params(stage2_model.model),
        ("Stage 2 full training module",) + count_params(stage2_model),
    ]

    lines = [
        "# Parameter Table",
        "",
        "| Component | Total parameters | Trainable parameters |",
        "| --- | ---: | ---: |",
    ]
    for name, total, trainable in rows:
        lines.append(f"| {name} | {total:,} | {trainable:,} |")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
