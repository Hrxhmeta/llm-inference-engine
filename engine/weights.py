from safetensors import safe_open

import config


def load_weights():
    """Return {tensor name: float32 NumPy array} for the learned weights."""
    weights = {}
    # framework="numpy" reads straight into NumPy, so no torch is involved.
    with safe_open(config.WEIGHTS_FILE, framework="numpy") as f:
        for name in f.keys():
            if name.endswith(config.SKIP_SUFFIXES):
                continue
            # Some copies of the file prefix every name with "transformer.".
            clean = name.removeprefix("transformer.")
            weights[clean] = f.get_tensor(name).astype("float32")
    return weights