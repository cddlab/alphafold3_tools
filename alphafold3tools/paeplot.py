import json
import math
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import matplotlib
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt, rcParams
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from alphafold3tools.log import log_setup
from alphafold3tools.utils import add_version_option, get_seed_sample_numbers

matplotlib.use("Agg")
rcParams["font.family"] = "Arial"
rcParams["mathtext.fontset"] = "cm"
rcParams["font.size"] = 12
rcParams["lines.linewidth"] = 0.4
rcParams["svg.fonttype"] = "none"

# Number of PAE panels per row in `plot_all_paes`. AlphaFold3 produces 5 diffusion
# samples per seed by default; samples beyond this wrap onto the next row.
MAX_PANELS_PER_ROW = 5

pymol_chain_colors = [
    "#33FF33",
    "#00FFFF",
    "#FF33CC",
    "#FFFF00",
    "#FF9999",
    "#E5E5E5",
    "#7F7FFF",
    "#FF7F00",
    "#7FFF7F",
    "#199999",
    "#FF007F",
    "#FFDD5E",
]

unhcr_chain_colors = [
    "#0072B2",
    "#FFC740",
    "#32C189",
    "#6CD8FD",
    "#D25A45",
    "#A097E3",
    "#7C3C36",
    "#BFBFBF",
]


def get_chain_ids_and_lengths(token_chain_ids: list[str]) -> dict:
    """Get the chain IDs and lengths from the token chain IDs.
    Args:
        token_chain_ids (list[str]): List of token chain IDs
    Returns:
        count_dict (dict): Dictionary containing the chain IDs and lengths
    """
    unique_chain_ids = list(dict.fromkeys(token_chain_ids))
    count_dict = {chain_id: token_chain_ids.count(chain_id) for chain_id in unique_chain_ids}
    return count_dict


def map_with_colorbar(
    fig,
    ax,
    model_name,
    data,
    chain_ids_and_lengths,
    cmap="Greens_r",
    chain_cmap: str | list = pymol_chain_colors,
    vmin=0,
    vmax=31.75,
    **kwargs,
):
    """Add a colorbar to the plot.

    Args:
        mappable (plt.cm.ScalarMappable): ScalarMappable object
        ax: Axes object
    """
    ax.set_title(model_name, pad=22)
    ax.set_xlabel("Scored Residue")
    ax.set_ylabel("Aligned Residue")
    mappable: plt.cm.ScalarMappable = ax.imshow(
        data["pae"],
        label=model_name,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    # chain boundary lines + collect segments
    total_len = 0
    segments = []
    labels = []
    for chain_id, chain_len in chain_ids_and_lengths.items():
        start = total_len
        total_len += chain_len
        end = total_len
        segments.append((start, end))
        labels.append(chain_id)
        ax.axvline(x=end, color="black", linewidth=0.5, linestyle="--")
        ax.axhline(y=end, color="black", linewidth=0.5, linestyle="--")

    ax.set_xlim(0, total_len)
    ax.set_ylim(total_len, 0)
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5)
    cbar = fig.colorbar(mappable, cax=cax, ax=ax, **kwargs)
    cbar.set_label("Expected Position Error (Ångströms)")

    # build chain index arrays
    chain_index_1d = np.zeros(total_len, dtype=int)
    for idx, (start, end) in enumerate(segments):
        chain_index_1d[start:end] = idx
    # top bar: shape (1, L); right bar: shape (L, 1)
    top_array = chain_index_1d[np.newaxis, :]
    right_array = chain_index_1d[:, np.newaxis]
    logger.info(f"Chain segments: {segments}, labels: {labels}")
    # build a discrete colormap and a BoundaryNorm so each segment index maps to one color
    nseg = len(segments)
    # normalize any input into exactly nseg colors (tile or slice)
    if isinstance(chain_cmap, (list, tuple)):
        colors = list(chain_cmap)
    else:
        cm = plt.get_cmap(chain_cmap, nseg)
        colors = [cm(i) for i in range(nseg)]
    if len(colors) < nseg:
        times = (nseg + len(colors) - 1) // len(colors)
        colors = (colors * times)[:nseg]
    else:
        colors = colors[:nseg]
    chain_cmap_discrete = ListedColormap(colors, name="chain_discrete")
    boundaries = np.arange(nseg + 1) - 0.5
    norm = BoundaryNorm(boundaries, chain_cmap_discrete.N)

    # top chain bar
    ax_top = divider.append_axes("top", size="3%", pad=0.02)
    ax_top.imshow(
        top_array,
        aspect="auto",
        cmap=chain_cmap_discrete,
        norm=norm,
        interpolation="nearest",
    )
    ax_top.set_xticks([(s + e) / 2 for s, e in segments])
    ax_top.set_xticklabels(labels, rotation=0, fontsize=8)
    ax_top.xaxis.tick_top()
    ax_top.xaxis.set_label_position("top")
    ax_top.set_yticks([])
    ax_top.tick_params(axis="x", length=0)
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    # right chain bar
    ax_right = divider.append_axes("right", size="3%", pad=0.02)
    ax_right.imshow(
        right_array,
        aspect="auto",
        cmap=chain_cmap_discrete,
        norm=norm,
        interpolation="nearest",
    )
    ax_right.set_yticks([(s + e) / 2 for s, e in segments])
    ax_right.set_yticklabels(labels, fontsize=8)
    ax_right.set_xticks([])
    ax_right.yaxis.tick_right()
    ax_right.yaxis.set_label_position("right")
    ax_right.tick_params(axis="y", length=0)
    for spine in ax_right.spines.values():
        spine.set_visible(False)
    return cbar


def layout_pae_panels(
    seed_samples: dict[int, list[int]], max_cols: int = MAX_PANELS_PER_ROW
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """Decide the grid position of each seed/sample PAE panel.

    Each seed starts on a new row, and its samples wrap onto additional rows
    every `max_cols` panels.

    Args:
        seed_samples (dict[int, list[int]]): Mapping from seed number to its
            sample numbers, as returned by `get_seed_sample_numbers()`.
        max_cols (int): Maximum number of panels per row.

    Returns:
        tuple: (nrows, ncols, placements), where placements is a list of
            (row, col, seed, sample) in plotting order. ncols is
            min(max_cols, the largest number of samples of a seed), so that
            fewer than `max_cols` samples do not leave empty columns.
    """
    if not seed_samples or max_cols < 1:
        raise ValueError("seed_samples must not be empty and max_cols must be positive.")
    ncols = min(max_cols, max(len(samples) for samples in seed_samples.values()))
    placements: list[tuple[int, int, int, int]] = []
    first_row = 0
    for seed, samples in seed_samples.items():
        for k, sample in enumerate(samples):
            placements.append((first_row + k // ncols, k % ncols, seed, sample))
        first_row += math.ceil(len(samples) / ncols)
    return first_row, ncols, placements


def find_sample_confidence_file(subdir: Path, basename: str, model_name: str) -> Path:
    """Find the per-sample confidences JSON file in a seed-*_sample-* directory.

    Args:
        subdir (Path): The seed-*_sample-* directory.
        basename (str): Name of the AlphaFold3 output directory.
        model_name (str): "seed-<M>_sample-<N>".

    Returns:
        Path: Path to the confidences JSON file.
    """
    # before 2025-03-10, the name of jsonfile was "confidences.json"
    oldconfidencefile = subdir / "confidences.json"
    # after 2025-03-10, the json file is f"{basename}_{model_name}_confidences.json
    newconfidencefile = subdir / f"{basename}_{model_name}_confidences.json"
    if oldconfidencefile.exists():
        return oldconfidencefile
    if newconfidencefile.exists():
        return newconfidencefile
    raise FileNotFoundError(f"No {oldconfidencefile} or {newconfidencefile} found.")


def plot_all_paes(
    dir: str | Path,
    out: str,
    title: str = "",
    cmap="Greens_r",
    chain_cmap: str | list = pymol_chain_colors,
    dpi=300,
    format="png",
) -> None:
    """Plot all predicted alignment error (PAE) of the models
    in the seed-(\\d+)_sample-(\\d+) directories.

    Each seed starts on a new row. When a seed has more than
    `MAX_PANELS_PER_ROW` samples, its panels wrap onto additional rows.
    For example, 2 seeds x 12 samples gives 6 rows (5 + 5 + 2 per seed).

    Args:
        dir (str | Path): Directory containing the subdirectories
        out (str): Suffix of the output plot file. e.g. foo_pae.png
        title (str): Title of the plot.
        cmap (str): Colormap to use for the plot.
        dpi (int): Dots per inch for the output plot.
        format (str): Format of the output plot file.
    """
    seed_samples = get_seed_sample_numbers(dir)
    if not seed_samples:
        raise FileNotFoundError(f"No seed-*_sample-* subdirectories found in {dir}.")
    dir = Path(dir).resolve()
    basename = Path(dir).name
    output_name = f"{dir}/{basename}_{out}.{format}"
    logger.info(f"The input directory is {dir}. The output plot will be saved as {output_name}.")
    bestconfidencejsonfile = Path(dir) / f"{basename}_confidences.json"
    if not bestconfidencejsonfile.exists():
        raise FileNotFoundError(f"{bestconfidencejsonfile} does not exist.")
    with open(bestconfidencejsonfile, "r") as f:
        data = json.load(f)
    data["pae"] = np.array(data["pae"])
    chain_ids_and_lengths = get_chain_ids_and_lengths(data["token_chain_ids"])
    nrows, ncols, placements = layout_pae_panels(seed_samples)
    logger.info(f"Plotting {len(placements)} PAEs in {nrows} rows x {ncols} columns.")
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 4.2 * nrows), dpi=300, squeeze=False)
    if title != "":
        fig.suptitle(title, fontsize=16)
    used_cells = set()
    for row, col, seed, sample in placements:
        model_name = f"seed-{seed}_sample-{sample}"
        confidencejsonfile = find_sample_confidence_file(dir / model_name, basename, model_name)
        with open(confidencejsonfile, "r") as f:
            data = json.load(f)
        map_with_colorbar(
            fig,
            axes[row, col],
            model_name,
            data,
            chain_ids_and_lengths,
            cmap=cmap,
            chain_cmap=chain_cmap,
            vmin=0,
            vmax=31.75,
            orientation="horizontal",
        )
        used_cells.add((row, col))
    # hide the trailing empty panels of a wrapped row (e.g. the last 3 of "5 + 5 + 2")
    for row in range(nrows):
        for col in range(ncols):
            if (row, col) not in used_cells:
                axes[row, col].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_name, dpi=dpi, format=format)
    plt.clf()
    plt.close()


def plot_best_pae(
    dir: str | Path,
    out: str,
    title: str = "",
    cmap="Greens_r",
    chain_cmap: str | list = pymol_chain_colors,
    dpi=300,
    format="png",
) -> None:
    """Plot only the best predicted alignment error (PAE).
    The plot will be generated from f"{dir}/*_confidence.json".

    Args:
        dir (str | Path): Directory containing the subdirectories.
        out (str): Suffix of the output plot file. e.g. foo_pae.png
        title (str): Title of the plot.
        cmap (str): Colormap to use for the plot.
        dpi (int): Dots per inch for the output plot.
        format (str): Output format ('png' or 'svg') of the plot file.
    """
    dir = Path(dir).resolve()
    basename = Path(dir).name
    output_name = f"{dir}/{basename}_{out}.{format}"
    logger.info(f"The input directory is {dir}. The output plot will be saved as {output_name}.")
    jsonfile = Path(dir) / f"{basename}_confidences.json"
    if not jsonfile.exists():
        FileNotFoundError(f"{jsonfile} does not exist.")
    with open(jsonfile, "r") as f:
        data = json.load(f)
    data["pae"] = np.array(data["pae"])
    chain_ids_and_lengths = get_chain_ids_and_lengths(data["token_chain_ids"])
    fig, ax = plt.subplots(figsize=(3.6, 4.2), dpi=300)
    if title != "":
        fig.suptitle(title, fontsize=16)
    map_with_colorbar(
        fig,
        ax,
        "",
        data,
        chain_ids_and_lengths,
        cmap=cmap,
        chain_cmap=chain_cmap,
        vmin=0,
        vmax=31.75,
        orientation="horizontal",
    )
    plt.tight_layout()
    plt.savefig(output_name, dpi=dpi, format=format)
    plt.clf()
    plt.close()


def plot_pae_from_json(
    jsonfile: str | Path,
    out: str,
    title: str = "",
    cmap="Greens_r",
    chain_cmap: str | list = pymol_chain_colors,
    dpi=300,
    format="png",
) -> None:
    """Plot the predicted alignment error (PAE) from a given JSON file.
    This function is for a JSON file.
    The file should contain "predicted_aligned_error" key (if downloaded
    from the AlphaFold DB) or "pae" and "token_chain_ids" keys
    (if generated by AlphaFold3).

    Args:
        jsonfile (str | Path): Path to the json file.
        out (str): Suffix of the output plot file. e.g. foo_pae.png
        title (str): Title of the plot.
        cmap (str): Colormap to use for the plot.
        dpi (int): Dots per inch for the output plot.
        format (str): Output format ('png' or 'svg') of the plot file.
    """
    jsonfile = Path(jsonfile).resolve()
    if not jsonfile.exists():
        FileNotFoundError(f"{jsonfile} does not exist.")
    with open(jsonfile, "r") as f:
        data_ = json.load(f)
    # AlphaFold3 json file
    if "pae" in data_ and "token_chain_ids" in data_:
        logger.info(f"{jsonfile} seems an AlphaFold3 json file.")
        data = {
            "pae": np.array(data_["pae"]),
        }
        chain_ids_and_lengths = get_chain_ids_and_lengths(data_["token_chain_ids"])
    # AlphaFold DB json file
    elif "predicted_aligned_error" in data_[0]:
        logger.info(f"{jsonfile} seems a json file downloaded from the AlphaFold DB.")
        # convert to numpy array and make a new dict
        data = {"pae": np.array(data_[0]["predicted_aligned_error"])}
        length = data["pae"].shape[0]
        # assume single chain with chain ID "A"
        chain_ids_and_lengths = {"A": length}
    else:
        raise KeyError(f"{jsonfile} does not contain 'predicted_aligned_error' or 'pae' key.")
    fig, ax = plt.subplots(figsize=(3.6, 4.2), dpi=300)
    if title != "":
        fig.suptitle(title, fontsize=16)
    map_with_colorbar(
        fig,
        ax,
        title,
        data,
        chain_ids_and_lengths,
        cmap=cmap,
        chain_cmap=chain_cmap,
        vmin=0,
        vmax=31.75,
        orientation="horizontal",
    )
    plt.tight_layout()
    output_name = f"{jsonfile.parent}/{jsonfile.stem}_{out}.{format}"
    plt.savefig(output_name, dpi=dpi, format=format)
    plt.clf()
    plt.close()


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Plot the predicted alignment error (PAE) of the models "
        "in the input directory. The input directory should contain "
        "**_confidence.json files. To plot all PAEs, the input directory "
        "should contain seed-(\\d+)_sample-(\\d+) directories.",
    )
    add_version_option(parser)
    parser.add_argument(
        "-i",
        "--input",
        help="Input directory containing the predicted models or a json file.",
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
        help="Colormap for the plot. Currently only 'bwr' and 'Greens_r' are supported.",
        choices=["bwr", "Greens_r"],
        default="bwr",
        type=str,
    )
    parser.add_argument(
        "--chain-cmap",
        help="Colormap for the chain colors. If 'pymol', use pymol colors; "
        "if 'unhcr', use UNHCR colors; otherwise, use as a matplotlib colormap name.",
        default="pymol",
        type=str,
        dest="chain_cmap",
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
        help="DPI of the output plot. Default is 100, but 300 is recommended for publication.",
        default=100,
        type=int,
    )
    parser.add_argument(
        "-f",
        "--format",
        help="Output format for the plot. Default is 'png'.",
        choices=["png", "svg"],
        default="png",
        type=str,
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
    if args.chain_cmap == "pymol":
        chain_cmap = pymol_chain_colors
    elif args.chain_cmap == "unhcr":
        chain_cmap = unhcr_chain_colors
    else:
        chain_cmap = args.chain_cmap  # use as a matplotlib colormap name
    if Path(args.input).is_dir():
        logger.info(f"The input directory is {args.input}.")
        if args.all:
            plot_all_paes(
                args.input,
                args.out,
                title=args.title,
                cmap=args.cmap,
                chain_cmap=chain_cmap,
                dpi=args.dpi,
                format=args.format,
            )
        else:
            plot_best_pae(
                args.input,
                args.out,
                title=args.title,
                cmap=args.cmap,
                chain_cmap=chain_cmap,
                dpi=args.dpi,
                format=args.format,
            )
    elif Path(args.input).is_file():
        logger.info(f"The input file is {args.input}.")
        plot_pae_from_json(
            args.input,
            args.out,
            title=args.title,
            cmap=args.cmap,
            chain_cmap=chain_cmap,
            dpi=args.dpi,
            format=args.format,
        )


if __name__ == "__main__":
    main()
