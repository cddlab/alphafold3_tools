import re
import string
from pathlib import Path


def sanitised_name(name) -> str:
    """Returns sanitised version of the name that can be used as a filename."""
    lower_spaceless_name = name.lower().replace(" ", "_")
    allowed_chars = set(string.ascii_lowercase + string.digits + "_-.")
    return "".join(char for char in lower_spaceless_name if char in allowed_chars)


def int_id_to_str_id(num: int) -> str:
    """Encodes a number as a string, using reverse spreadsheet style naming.
    This block is cited from
    https://github.com/google-deepmind/alphafold3/blob/main/src/alphafold3/structure/mmcif.py#L40

    Args:
      num: A positive integer.

    Returns:
      A string that encodes the positive integer using reverse spreadsheet style,
      naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
      usual way to encode chain IDs in mmCIF files.
    """
    if num <= 0:
        raise ValueError(f"Only positive integers allowed, got {num}.")

    num = num - 1  # 1-based indexing.
    output = []
    while num >= 0:
        output.append(chr(num % 26 + ord("A")))
        num = num // 26 - 1
    return "".join(output)


def get_seednumbers(dir: str | Path) -> list[int]:
    """Get the seed numbers from the directory names.

    Args:
        dir (str | Path): Directory containing the subdirectories

    Returns:
        list[int]: List of seed numbers
    """
    pattern = re.compile(r"seed-(\d+)_sample-[0-4]")
    subdirs = [d for d in Path(dir).iterdir() if d.is_dir() and pattern.match(d.name)]
    seednumbers: list[int] = list(
        {int(m.group(1)) for d in subdirs if (m := pattern.match(d.name))}
    )
    return sorted(seednumbers)
