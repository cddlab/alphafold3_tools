import json
import re
import string
from argparse import ArgumentParser
from pathlib import Path
from typing import Final

from loguru import logger

from alphafold3tools import __version__

DEFAULT_MAX_SUBSEQUENCE_RATIO: Final[float] = 0.95
# Custom templates are often the query structure itself; keep them.
CUSTOM_TEMPLATE_MAX_SUBSEQUENCE_RATIO: Final[float] = 1.0


def resolve_max_subsequence_ratio(max_subsequence_ratio: float | None, *, custom_template: bool) -> float | None:
    """Resolves the --max_subsequence_ratio option of the template search.

    Args:
        max_subsequence_ratio: The value given on the command line, or None if
            the option was not given.
        custom_template: Whether --custom_template is used, which changes the
            default from 0.95 to 1.0.
    Returns:
        The ratio for the template search, or None (no subsequence-ratio
        filtering) when it is 1.0.
    """
    if max_subsequence_ratio is None:
        if custom_template:
            logger.warning(
                "--custom_template is specified, so --max_subsequence_ratio is set to "
                f"{CUSTOM_TEMPLATE_MAX_SUBSEQUENCE_RATIO} instead of the default {DEFAULT_MAX_SUBSEQUENCE_RATIO}."
            )
            max_subsequence_ratio = CUSTOM_TEMPLATE_MAX_SUBSEQUENCE_RATIO
        else:
            max_subsequence_ratio = DEFAULT_MAX_SUBSEQUENCE_RATIO
    if max_subsequence_ratio == 1.0:
        logger.success(
            "No templates will be excluded based on subsequence ratio since max_subsequence_ratio is set to 1.0."
        )
        return None
    return max_subsequence_ratio


def add_version_option(parser: ArgumentParser) -> None:
    """
    Add -v/--version option to an existing ArgumentParser.
    """
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )


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


def to_json(content: dict) -> str:
    """Converts Input to an AlphaFold JSON."""
    alphafold_json = json.dumps(content, indent=2)
    # Collapse the per-element newlines inside flat list values for
    # queryIndices/templateIndices/modelSeeds (numbers) and id (strings), e.g.
    # "queryIndices": [12, 13, 14, 15] instead of one element per line. We
    # match the entire region between the square brackets by excluding
    # brackets themselves, since these fields never contain nested arrays.
    return re.sub(
        r'("(?:queryIndices|templateIndices|modelSeeds|id)": \[)([^\[\]]*)(\])',
        lambda mtch: mtch[1] + re.sub(r"\s+", " ", mtch[2]).strip() + mtch[3],
        alphafold_json,
    )
