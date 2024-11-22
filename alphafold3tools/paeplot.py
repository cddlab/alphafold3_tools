import json
import re
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from alphafold3tools.log import log_setup

matplotlib.use("Agg")


def get_chain_ids_and_lengths(token_chain_ids: list[str]) -> dict:
    """Get the chain IDs and lengths from the token chain IDs.
    Args:
        token_chain_ids (list[str]): List of token chain IDs
    Returns:
        count_dict (dict): Dictionary containing the chain IDs and lengths
    """
    unique_chain_ids = list(dict.fromkeys(token_chain_ids))
    count_dict = {
        chain_id: token_chain_ids.count(chain_id) for chain_id in unique_chain_ids
    }
    return count_dict


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


def map_with_colorbar(
    fig,
    ax,
    model_name,
    data,
    chain_ids_and_lengths,
    cmap="Greens_r",
    vmin=0,
    vmax=31.75,
    **kwargs,
):
    """Add a colorbar to the plot.

    Args:
        mappable (plt.cm.ScalarMappable): ScalarMappable object
        ax: Axes object
    """
    ax.set_title(model_name)
    ax.set_xlabel("Scored Residue")
    ax.set_ylabel("Aligned Residue")
    mappable: plt.cm.ScalarMappable = ax.imshow(
        data["pae"],
        label=model_name,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    # add black trace between the boundaries of the chains
    pos = 0
    positions = []
    labels = []
    for chain_id, chain_len in chain_ids_and_lengths.items():
        pos += chain_len
        ax.axvline(x=pos, color="black", linewidth=0.5)
        ax.axhline(y=pos, color="black", linewidth=0.5)
        labels.append(chain_id)
        positions.append(pos)
    ax.set_xlim(0, pos)
    ax.set_ylim(pos, 0)
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = fig.colorbar(mappable, cax=cax, ax=ax, **kwargs)
    return cbar


def plot_all_paes(
    dir: str | Path, out: str, title: str = "", cmap="Greens_r", dpi=300
) -> None:
    """Plot all predicted alignment error (PAE) of the models
    in the seed-(\\d+)_sample-[0-4] directories.

    Args:
        dir (str | Path): Directory containing the subdirectories
    """
    seednumbers = get_seednumbers(dir)
    dir = Path(dir).resolve()
    basename = Path(dir).name
    png_name = f"{dir}/{basename}_{out}.png"
    logger.info(
        f"The input directory is {dir}. The output plot will be saved as {png_name}."
    )
    jsonfile = Path(dir) / f"{basename}_confidences.json"
    if not jsonfile.exists():
        FileNotFoundError(f"{jsonfile} does not exist.")
    with open(jsonfile, "r") as f:
        data = json.load(f)
    # data[pae]をnumpyに変換
    data["pae"] = np.array(data["pae"])
    chain_ids_and_lengths = get_chain_ids_and_lengths(data["token_chain_ids"])
    fig, axes = plt.subplots(
        len(seednumbers), 5, figsize=(3.6 * 5, 4.2 * len(seednumbers)), dpi=300
    )
    if title != "":
        fig.suptitle(title, fontsize=16)
    for i, seed in enumerate(seednumbers):
        for j in range(5):
            subdir = Path(dir) / f"seed-{seed}_sample-{j}"
            jsonfile = subdir / "confidences.json"
            if not jsonfile.exists():
                FileNotFoundError(f"{jsonfile} does not exist.")
            with open(jsonfile, "r") as f:
                data = json.load(f)
            model_name = f"seed-{seed}_sample-{j}"
            if len(seednumbers) == 1:
                map_with_colorbar(
                    fig,
                    axes[j],
                    model_name,
                    data,
                    chain_ids_and_lengths,
                    cmap=cmap,
                    vmin=0,
                    vmax=31.75,
                    orientation="horizontal",
                    pad=0.2,
                    label="Expected Position Error (Ångströms)",
                )
            else:
                map_with_colorbar(
                    fig,
                    axes[i, j],
                    model_name,
                    data,
                    chain_ids_and_lengths,
                    cmap=cmap,
                    vmin=0,
                    vmax=31.75,
                    orientation="horizontal",
                    pad=0.2,
                    label="Expected Position Error (Ångströms)",
                )
    plt.tight_layout()
    png_name = f"{dir}/{basename}_{out}.png"
    plt.savefig(png_name, dpi=dpi)


def plot_best_pae(
    dir: str | Path,
    out: str,
    title: str = "",
    cmap="Greens_r",
    dpi=300,
) -> None:
    """Plot only the best predicted alignment error (PAE).
    The plot will be generated from f"{dir}/*_confidence.json".

    Args:
        dir (str | Path): Directory containing the subdirectories
    """
    dir = Path(dir).resolve()
    basename = Path(dir).name
    png_name = f"{dir}/{basename}_{out}.png"
    logger.info(
        f"The input directory is {dir}. The output plot will be saved as {png_name}."
    )
    jsonfile = Path(dir) / f"{basename}_confidences.json"
    if not jsonfile.exists():
        FileNotFoundError(f"{jsonfile} does not exist.")
    with open(jsonfile, "r") as f:
        data = json.load(f)
    # data[pae]をnumpyに変換
    data["pae"] = np.array(data["pae"])
    chain_ids_and_lengths = get_chain_ids_and_lengths(data["token_chain_ids"])
    fig, ax = plt.subplots(figsize=(3.6, 4.2), dpi=300)
    if title != "":
        fig.suptitle(title, fontsize=16)
    map_with_colorbar(
        fig,
        ax,
        "Best",
        data,
        chain_ids_and_lengths,
        cmap=cmap,
        vmin=0,
        vmax=31.75,
        orientation="horizontal",
        pad=0.2,
        label="Expected Position Error (Ångströms)",
    )
    plt.tight_layout()
    plt.savefig(png_name, dpi=dpi)


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Plot the predicted alignment error (PAE) of the models "
        "in the input directory. The input directory should contain "
        "**_confidence.json files. To plot all PAEs, the input directory "
        "should contain seed-(\\d+)_sample-[0-4] directories.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input directory containing the predicted models.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Suffix of the output plot file. Default is 'pae'. e.g. foo_pae.png",
        default="af3pae",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--cmap",
        help="Colormap for the plot. "
        "Currently only 'bwr' and 'Greens_r' are supported.",
        choices=["bwr", "Greens_r"],
        default="bwr",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--title",
        help="Title of the plot.",
        default="",
        type=str,
    )
    parser.add_argument(
        "-a",
        "--all",
        help="Plot all PAEs. Default is to plot only the best PAE.",
        action="store_true",
    )
    parser.add_argument(
        "--dpi",
        help="DPI of the output plot. Default is 100, but 300 is recommended "
        "for publication.",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        dest="loglevel",
        action="store_const",
        const="DEBUG",
        default="SUCCESS",
    )
    args = parser.parse_args()
    log_setup(args.loglevel)
    if not Path(args.input).is_dir():
        FileNotFoundError(
            f"{args.input} is not a directory. Specify a directory containing "
            "the AlphaFold3 predicted models."
        )
    if args.all:
        plot_all_paes(
            args.input,
            args.out,
            title=args.title,
            cmap=args.cmap,
            dpi=args.dpi,
        )
    else:
        plot_best_pae(
            args.input,
            args.out,
            title=args.title,
            cmap=args.cmap,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
