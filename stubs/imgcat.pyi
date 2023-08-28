from io import TextIOWrapper

def imgcat(
    data: str | TextIOWrapper,
    filename=None,
    width=None,
    height=None,
    preserve_aspect_ratio=True,
    pixels_per_line=24,
    fp=None,
): ...
