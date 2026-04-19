#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _common import REPO_ROOT, default_stage2_paths
from hiread.data.data_feature import GenomicFeature, SequenceFeature
from hiread_diffusion import MultiModalDiffusion


WINDOW_SIZE = 2_097_152


def load_inputs(chr_name, start, seq_dir, feature_specs):
    sequence_feature = SequenceFeature(path=str(Path(seq_dir) / f"{chr_name}.fa.gz"))
    sequence = sequence_feature.get(start, start + WINDOW_SIZE).T

    epi_arrays = []
    for _, file_path, norm in feature_specs:
        feature = GenomicFeature(path=file_path, norm=norm)
        values = feature.get(chr_name, start, start + WINDOW_SIZE)
        if len(values) != sequence.shape[1]:
            values = np.interp(
                np.linspace(0, 1, sequence.shape[1]),
                np.linspace(0, 1, len(values)),
                values,
            )
        epi_arrays.append(values)

    if epi_arrays:
        epi_features = np.stack(epi_arrays, axis=0)
    else:
        epi_features = np.zeros((1, sequence.shape[1]), dtype=np.float32)

    stacked = np.concatenate([sequence, epi_features], axis=0)
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0)


def save_plot(out_dir, chr_name, start, stage1_prior, stage2_prediction):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, matrix, title in [
        (axes[0], stage1_prior, "Stage 1 Prior"),
        (axes[1], stage2_prediction, "Stage 2 Prediction"),
    ]:
        image = ax.imshow(matrix, cmap="Reds", vmin=0, vmax=5)
        ax.set_title(title)
        ax.set_xlabel(f"{chr_name}:{start:,}-{start + WINDOW_SIZE:,}")
        ax.set_ylabel("Bins")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    figure_path = Path(out_dir) / f"{chr_name}_{start}_prediction.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return figure_path


def load_stage2_model(checkpoint_path, device, output_dir, num_epi_features, stage1_checkpoint=None):
    checkpoint_path = str(checkpoint_path)
    try:
        model = MultiModalDiffusion.load_from_checkpoint(checkpoint_path, map_location=device)
        model.eval()
        model.to(device)
        return model
    except (TypeError, RuntimeError) as exc:
        if stage1_checkpoint is None:
            raise RuntimeError(
                "Failed to load the stage-2 checkpoint directly. "
                "If this is an older checkpoint, rerun with --stage1-checkpoint "
                "so the stage-1 conditioning branch can be restored explicitly."
            ) from exc

        compatibility_model = MultiModalDiffusion(
            hic_filename="hiread_stage2_predict",
            validation_folder=str(Path(output_dir) / "_compat_validation"),
            encoder_decoder_model=str(stage1_checkpoint),
            encoder_decoder_kind="hiread_stage1",
            val_chr="chr10",
            test_chr="chr15",
            num_epi_features=num_epi_features,
        )

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        filtered_state_dict = {
            key: value
            for key, value in state_dict.items()
            if not key.startswith("encoder_decoder.")
        }
        compatibility_model.load_state_dict(filtered_state_dict, strict=False)
        compatibility_model.eval()
        compatibility_model.to(device)
        return compatibility_model


def main():
    parser = argparse.ArgumentParser(description="Run stage-2 prediction and save a plot of the stage-1 prior and final stage-2 prediction.")
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--stage1-checkpoint", default=None, help="Optional stage-1 checkpoint used to restore compatibility with older stage-2 checkpoints.")
    parser.add_argument("--data-root", required=True, help="Dataset root containing dna_sequence/ and <celltype>/.")
    parser.add_argument("--celltype", required=True)
    parser.add_argument("--seq-dir", default=None)
    parser.add_argument("--chr", dest="chr_name", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "stage2_prediction"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    inferred_paths = default_stage2_paths(args.data_root, args.celltype)
    seq_dir = args.seq_dir or inferred_paths["seq_dir"]

    default_chip = Path(inferred_paths["chip_bw"])
    feature_specs = [("CHIP", str(default_chip), None)] if default_chip.exists() else []

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    input_tensor = load_inputs(args.chr_name, args.start, seq_dir, feature_specs).to(device)
    model = load_stage2_model(
        checkpoint_path=args.stage2_checkpoint,
        device=device,
        output_dir=output_dir,
        num_epi_features=max(len(feature_specs), 1),
        stage1_checkpoint=args.stage1_checkpoint,
    )

    with torch.no_grad():
        _, stage1_prior = model._prepare_condition_tensors(input_tensor)
        stage2_prediction = model(input_tensor).view(256, 256)

    stage1_prior_np = stage1_prior[0].detach().cpu().numpy()
    stage2_prediction_np = stage2_prediction.detach().cpu().numpy()

    np.save(output_dir / f"{args.chr_name}_{args.start}_stage1_prior.npy", stage1_prior_np)
    np.save(output_dir / f"{args.chr_name}_{args.start}_stage2_prediction.npy", stage2_prediction_np)
    figure_path = save_plot(output_dir, args.chr_name, args.start, stage1_prior_np, stage2_prediction_np)

    print(str(figure_path))


if __name__ == "__main__":
    main()
