import torch

from hiread.model import hiread_models

def load_default(model_path, pos_embedding="EPEG"):
    model = get_model("ConvTransModel", 256, pos_embedding=pos_embedding)
    load_checkpoint(model, model_path)
    return model

def get_model(model_name, mid_hidden, num_genomic_features=1, pos_embedding="EPEG"):
    ModelClass = getattr(hiread_models, model_name)
    model = ModelClass(num_genomic_features, mid_hidden=mid_hidden, pos_embedding=pos_embedding)
    return model

def load_checkpoint(model, model_path):
    print(f"Loading weights from {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model_weights = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        model_weights = checkpoint
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")

    cleaned_weights = {}
    for key, value in model_weights.items():
        clean_key = key
        for prefix in ("model.", "net."):
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix):]
        cleaned_weights[clean_key] = value

    # Zero-init any missing keys so that absent PPEG weights don't
    # introduce random noise (these checkpoints predate the PPEG addition).
    model_sd = model.state_dict()
    missing = set(model_sd.keys()) - set(cleaned_weights.keys())
    if missing:
        print(f"  Zero-initializing {len(missing)} missing keys (e.g. PPEG)")
        for k in missing:
            cleaned_weights[k] = torch.zeros_like(model_sd[k])

    model.load_state_dict(cleaned_weights, strict=False)
    model.eval()
    return model

if __name__ == '__main__':
    raise SystemExit("Import this module from inference code.")
