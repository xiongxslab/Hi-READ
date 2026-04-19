#!/usr/bin/env python3

import argparse
import compileall
import json
import os
import tempfile
from pathlib import Path

import torch

from _common import REPO_ROOT


EXAMPLE_DATA_ROOT = Path(os.environ["HIREAD_VALIDATION_DATA_ROOT"]) if os.environ.get("HIREAD_VALIDATION_DATA_ROOT") else None


def compile_python_files():
    return compileall.compile_dir(str(REPO_ROOT), quiet=1)


def import_core_modules():
    import hiread.training.stage1  # noqa: F401
    import hiread.inference.utils.inference_utils  # noqa: F401
    import hiread.inference.utils.model_utils  # noqa: F401
    import hiread_diffusion.adapters  # noqa: F401
    import hiread_diffusion.diffusion_multimodal_model  # noqa: F401
    import hiread_diffusion.multimodal_datasets  # noqa: F401
    import hiread_diffusion.stage2  # noqa: F401


def build_temp_stage1_checkpoint():
    from hiread.training.stage1 import Stage1LightningModule

    module = Stage1LightningModule()
    state_dict = {f"model.{key}": value for key, value in module.model.state_dict().items()}
    temp_dir = Path(tempfile.mkdtemp(prefix="hiread_validate_"))
    checkpoint_path = temp_dir / "stage1.ckpt"
    torch.save({"state_dict": state_dict}, checkpoint_path)
    return checkpoint_path


def instantiate_models():
    from hiread.training.stage1 import Stage1LightningModule
    from hiread_diffusion import MultiModalDiffusion, MultiModalDiffusionEncoderDecoder

    temp_stage1_ckpt = build_temp_stage1_checkpoint()
    stage1_module = Stage1LightningModule()
    encoder_decoder = MultiModalDiffusionEncoderDecoder(
        validation_folder=str(REPO_ROOT / "outputs" / "tmp_validation"),
        val_chr="chr10",
        test_chr="chr15",
        num_epi_features=1,
    )
    stage2_model = MultiModalDiffusion(
        hic_filename="hiread_stage2",
        validation_folder=str(REPO_ROOT / "outputs" / "tmp_validation"),
        encoder_decoder_model=str(temp_stage1_ckpt),
        encoder_decoder_kind="hiread_stage1",
        val_chr="chr10",
        test_chr="chr15",
        num_epi_features=1,
    )
    return {
        "stage1_total_params": sum(param.numel() for param in stage1_module.model.parameters()),
        "encoder_decoder_total_params": sum(param.numel() for param in encoder_decoder.parameters()),
        "stage2_trainable_params": sum(param.numel() for param in stage2_model.parameters() if param.requires_grad),
    }


def smoke_test_datasets():
    if EXAMPLE_DATA_ROOT is None or not EXAMPLE_DATA_ROOT.exists():
        return {"example_data_available": False}

    from hiread.data.genome_dataset import GenomeDataset
    from hiread_diffusion.multimodal_datasets import MultiModalGenomicDataSet

    feature_dict = {"CHIP": {"file_name": "chip.bw", "norm": None}}
    stage1_dataset = GenomeDataset(
        celltype_root=str(EXAMPLE_DATA_ROOT / "H9_hESCs"),
        genome_assembly="hg38",
        feat_dicts=feature_dict,
        mode="val",
        include_sequence=True,
        include_genomic_features=True,
        use_aug=False,
        val_chromosomes=["chr10"],
        test_chromosomes=["chr15"],
        centrotelo_path=str(EXAMPLE_DATA_ROOT / "centrotelo.bed"),
    )
    stage1_sample = stage1_dataset[0]

    stage2_dataset = MultiModalGenomicDataSet(
        reference_genome_file=str(EXAMPLE_DATA_ROOT / "dna_sequence"),
        bed_exclude=str(REPO_ROOT / "hiread_diffusion" / "exclude_regions.bed"),
        chromosomes=["chr10"],
        slide_size=500_000,
        normal_chromosomes=[f"chr{i}" for i in range(1, 23)],
        hic_file_name="",
        epi_features_config={"CHIP": {"file_path": str(EXAMPLE_DATA_ROOT / "H9_hESCs" / "genomic_features" / "chip.bw"), "norm": None}},
        seq_dir=str(EXAMPLE_DATA_ROOT / "dna_sequence"),
        hic_dir=str(EXAMPLE_DATA_ROOT / "H9_hESCs" / "hic_matrix"),
    )
    stage2_sample = stage2_dataset[0]

    return {
        "example_data_available": True,
        "stage1_sample_seq_shape": list(stage1_sample[0].shape),
        "stage1_sample_feature_count": len(stage1_sample[1]),
        "stage1_sample_matrix_shape": list(stage1_sample[2].shape),
        "stage2_sample_feature_shape": list(stage2_sample[0].shape),
        "stage2_sample_matrix_shape": list(stage2_sample[1].shape),
    }


def forward_smoke_test():
    if EXAMPLE_DATA_ROOT is None or not EXAMPLE_DATA_ROOT.exists():
        return {"example_data_available": False}

    from hiread.data.genome_dataset import GenomeDataset
    from hiread.training.stage1 import Stage1LightningModule
    from hiread_diffusion import MultiModalDiffusion
    from hiread_diffusion.multimodal_datasets import MultiModalGenomicDataSet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stage1_dataset = GenomeDataset(
        celltype_root=str(EXAMPLE_DATA_ROOT / "H9_hESCs"),
        genome_assembly="hg38",
        feat_dicts={"CHIP": {"file_name": "chip.bw", "norm": None}},
        mode="val",
        include_sequence=True,
        include_genomic_features=True,
        use_aug=False,
        val_chromosomes=["chr10"],
        test_chromosomes=["chr15"],
        centrotelo_path=str(EXAMPLE_DATA_ROOT / "centrotelo.bed"),
    )
    seq, features, _, *_ = stage1_dataset[0]
    stage1_inputs = torch.cat(
        [
            torch.tensor(seq).unsqueeze(0),
            torch.cat([torch.tensor(item).unsqueeze(0).unsqueeze(2) for item in features], dim=2),
        ],
        dim=2,
    ).to(device).float()

    stage1_module = Stage1LightningModule().to(device).eval()
    with torch.no_grad():
        stage1_output = stage1_module(stage1_inputs)

    temp_stage1_ckpt = build_temp_stage1_checkpoint()
    stage2_dataset = MultiModalGenomicDataSet(
        reference_genome_file=str(EXAMPLE_DATA_ROOT / "dna_sequence"),
        bed_exclude=str(REPO_ROOT / "hiread_diffusion" / "exclude_regions.bed"),
        chromosomes=["chr10"],
        slide_size=500_000,
        normal_chromosomes=[f"chr{i}" for i in range(1, 23)],
        hic_file_name="",
        epi_features_config={"CHIP": {"file_path": str(EXAMPLE_DATA_ROOT / "H9_hESCs" / "genomic_features" / "chip.bw"), "norm": None}},
        seq_dir=str(EXAMPLE_DATA_ROOT / "dna_sequence"),
        hic_dir=str(EXAMPLE_DATA_ROOT / "H9_hESCs" / "hic_matrix"),
    )
    stage2_inputs, _, _ = stage2_dataset[0]
    stage2_inputs = stage2_inputs.unsqueeze(0).to(device)
    stage2_model = MultiModalDiffusion(
        hic_filename="hiread_stage2",
        validation_folder=str(REPO_ROOT / "outputs" / "tmp_validation"),
        encoder_decoder_model=str(temp_stage1_ckpt),
        encoder_decoder_kind="hiread_stage1",
        val_chr="chr10",
        test_chr="chr15",
        num_epi_features=1,
    ).to(device).eval()
    with torch.no_grad():
        stage2_output = stage2_model(stage2_inputs)

    return {
        "device": str(device),
        "stage1_output_shape": list(stage1_output.shape),
        "stage1_output_min": float(stage1_output.min()),
        "stage1_output_max": float(stage1_output.max()),
        "stage2_output_shape": list(stage2_output.shape),
        "stage2_output_min": float(stage2_output.min()),
        "stage2_output_max": float(stage2_output.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Run static and smoke validation for the cleaned repository.")
    parser.add_argument("--output", default=str(REPO_ROOT / "demo_outputs" / "validation_report.json"))
    args = parser.parse_args()

    report = {
        "compile_ok": compile_python_files(),
    }
    import_core_modules()
    report["import_ok"] = True
    report["model_instantiation"] = instantiate_models()
    report["dataset_smoke_test"] = smoke_test_datasets()
    report["forward_smoke_test"] = forward_smoke_test()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
