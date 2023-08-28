from io import TextIOWrapper
from typing import Any, Optional

def imgcat(
    data: str | TextIOWrapper,
    filename: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    preserve_aspect_ratio: bool = True,
    pixels_per_line: int = 24,
    fp: Any = None,
) -> None: ...
