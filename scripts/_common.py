import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_feature_spec(spec, require_existing_path=False):
    if "=" not in spec:
        raise ValueError(f"Invalid feature spec: {spec}. Expected NAME=PATH[:NORM].")
    name, value = spec.split("=", 1)
    if ":" in value:
        path, norm = value.rsplit(":", 1)
        norm = None if norm.lower() == "none" else norm
    else:
        path, norm = value, None
    if require_existing_path:
        feature_path = Path(path)
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature path does not exist: {path}")
    return name, path, norm


def parse_chromosomes(value):
    if value is None or value.strip() == "":
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_devices(value):
    if value in (None, "auto"):
        return "auto"
    if isinstance(value, int):
        return value
    if str(value).isdigit():
        return int(value)
    return value


def default_stage2_paths(stage2_data_root, celltype):
    data_root = Path(stage2_data_root)
    return {
        "seq_dir": str(data_root / "dna_sequence"),
        "hic_dir": str(data_root / celltype / "hic_matrix"),
        "chip_bw": str(data_root / celltype / "genomic_features" / "chip.bw"),
        "centrotelo_bed": str(data_root / "centrotelo.bed"),
    }
