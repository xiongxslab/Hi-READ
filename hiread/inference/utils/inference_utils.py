import os
import numpy as np
import torch

from hiread.data.data_feature import GenomicFeature, SequenceFeature
from hiread.inference.utils import model_utils

def preprocess_default(seq, chip):
    seq = torch.tensor(seq).unsqueeze(0)
    chip = torch.tensor(np.nan_to_num(chip, 0))
    features = [chip]
    features = torch.cat([feat.unsqueeze(0).unsqueeze(2) for feat in features], dim=2)
    inputs = torch.cat([seq, features], dim=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)
    return inputs

def resolve_sequence_path(chr_name, seq_path):
    if seq_path.endswith(".fa.gz"):
        return seq_path
    return os.path.join(seq_path, f"{chr_name}.fa.gz")

def load_region(chr_name, start, seq_path, chip_path, window=2097152):
    """Load one genomic region."""
    end = start + window
    seq, chip = load_data_default(chr_name, seq_path, chip_path)
    seq_region, chip_region = get_data_at_interval(chr_name, start, end, seq, chip)
    return seq_region, chip_region

def load_data_default(chr_name, seq_path, chip_path):
    seq_chr_path = resolve_sequence_path(chr_name, seq_path)
    seq = SequenceFeature(path=seq_chr_path)
    chip = GenomicFeature(path=chip_path, norm=None)
    return seq, chip

def get_data_at_interval(chr_name, start, end, seq, chip):
    """Slice data from arrays with transformations."""
    seq_region = seq.get(start, end)
    chip_region = chip.get(chr_name, start, end)
    return seq_region, chip_region

def prediction(seq_region, chip_region, model_path):
    model = model_utils.load_default(model_path)
    inputs = preprocess_default(seq_region, chip_region)
    pred = model(inputs)[0].detach().cpu().numpy()
    return pred
