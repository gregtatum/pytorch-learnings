import json


def naive_hash(obj: object) -> str:
    """
    Naively hashes an object and turns it into a string. This is not robust but should
    work well enough for simple JSON-serializable objects.
    """
    import hashlib

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
