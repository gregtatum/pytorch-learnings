import json
from typing import Dict, Optional
import torch


def naive_hash(obj: object) -> str:
    """
    Naively hashes an object and turns it into a string. This is not robust but should
    work well enough for simple JSON-serializable objects.
    """
    import hashlib

    if not isinstance(obj, Dict):
        # Get the dict value of a class.
        obj = obj.__dict__

    string = json.dumps(obj, sort_keys=True)
    return hex(int(hashlib.sha256(string.encode("utf-8")).hexdigest(), 16))[3:12]


def output_plot(path):
    from imgcat import imgcat
    import matplotlib.pyplot as plt

    plt.savefig(path)
    print()
    print(f"Plot saved to: {path}")
    print()
    imgcat(open(path))
    print()


def save_json(path_str: str, obj):
    if (
        not isinstance(obj, Dict)
        and not isinstance(obj, list)
        and not isinstance(obj, str)
        and not isinstance(obj, int)
        and not isinstance(obj, float)
    ):
        # Get the dict value of a class.
        obj = obj.__dict__

    with open(path_str, "w") as f:
        f.write(json.dumps(obj, indent=2, sort_keys=True))


device: Optional[torch.device] = None


def get_device():
    global device
    if not device:
        if torch.backends.mps.is_available():
            print("PyTorch Device: mps")
            device = torch.device("mps")
        elif torch.cuda.is_available():
            print("PyTorch Device: cuda")
            device = torch.device("cuda")
        else:
            print("PyTorch Device: cpu")
            device = torch.device("cpu")

    return device
