import json
from pathlib import Path

import pytest
from matplotlib import image as mpimg

from alphafold3tools.paeplot import layout_pae_panels, plot_all_paes

PAE = [[0.5, 10.0, 20.0], [10.0, 0.5, 20.0], [20.0, 20.0, 0.5]]
TOKEN_CHAIN_IDS = ["A", "A", "B"]


def _write_confidences(path: Path) -> None:
    path.write_text(json.dumps({"pae": PAE, "token_chain_ids": TOKEN_CHAIN_IDS}))


@pytest.fixture
def make_af3_outdir(tmp_path: Path):
    """Build a minimal AlphaFold3 output directory with the given seeds and number of samples."""

    def _make(seeds: list[int], nsamples: int, name: str = "job") -> Path:
        outdir = tmp_path / name
        outdir.mkdir()
        _write_confidences(outdir / f"{name}_confidences.json")
        for seed in seeds:
            for sample in range(nsamples):
                model_name = f"seed-{seed}_sample-{sample}"
                subdir = outdir / model_name
                subdir.mkdir()
                _write_confidences(subdir / f"{name}_{model_name}_confidences.json")
        return outdir

    return _make


def test_layout_default_five_samples():
    nrows, ncols, placements = layout_pae_panels({1: [0, 1, 2, 3, 4]})
    assert (nrows, ncols) == (1, 5)
    assert placements == [(0, j, 1, j) for j in range(5)]


def test_layout_wraps_each_seed_every_five_samples():
    # 2 seeds x 12 samples: each seed takes 3 rows (5 + 5 + 2), so 6 rows in total.
    nrows, ncols, placements = layout_pae_panels({1: list(range(12)), 2: list(range(12))})
    assert (nrows, ncols) == (6, 5)
    assert len(placements) == 24
    positions = {(seed, sample): (row, col) for row, col, seed, sample in placements}
    assert positions[(1, 4)] == (0, 4)
    assert positions[(1, 5)] == (1, 0)
    assert positions[(1, 10)] == (2, 0)
    assert positions[(1, 11)] == (2, 1)
    # the second seed starts on a new row
    assert positions[(2, 0)] == (3, 0)
    assert positions[(2, 11)] == (5, 1)


def test_layout_new_row_for_each_seed_with_ten_samples():
    nrows, ncols, placements = layout_pae_panels({1: list(range(10)), 3: list(range(10))})
    assert (nrows, ncols) == (4, 5)
    assert [(row, col) for row, col, seed, sample in placements if seed == 3][0] == (2, 0)


def test_layout_fewer_than_five_samples_shrinks_columns():
    nrows, ncols, placements = layout_pae_panels({1: [0, 1, 2], 2: [0, 1, 2]})
    assert (nrows, ncols) == (2, 3)
    assert placements[-1] == (1, 2, 2, 2)


def test_layout_rejects_empty_input():
    with pytest.raises(ValueError):
        layout_pae_panels({})


def test_plot_all_paes_figure_has_one_block_of_rows_per_seed(make_af3_outdir):
    outdir = make_af3_outdir(seeds=[1, 2], nsamples=12)
    dpi = 20
    plot_all_paes(outdir, "af3pae", dpi=dpi)
    outfile = outdir / "job_af3pae.png"
    assert outfile.exists()
    height, width = mpimg.imread(outfile).shape[:2]
    # 6 rows x 5 columns of 3.6 x 4.2 inch panels
    assert height == round(4.2 * 6 * dpi)
    assert width == round(3.6 * 5 * dpi)


def test_plot_all_paes_with_default_five_samples(make_af3_outdir):
    outdir = make_af3_outdir(seeds=[1], nsamples=5)
    dpi = 20
    plot_all_paes(outdir, "af3pae", dpi=dpi)
    height, width = mpimg.imread(outdir / "job_af3pae.png").shape[:2]
    assert height == round(4.2 * dpi)
    assert width == round(3.6 * 5 * dpi)


def test_plot_all_paes_missing_sample_confidences_raises(make_af3_outdir):
    outdir = make_af3_outdir(seeds=[1], nsamples=6)
    (outdir / "seed-1_sample-5" / "job_seed-1_sample-5_confidences.json").unlink()
    with pytest.raises(FileNotFoundError):
        plot_all_paes(outdir, "af3pae", dpi=20)


def test_plot_all_paes_without_sample_dirs_raises(tmp_path: Path):
    outdir = tmp_path / "job"
    outdir.mkdir()
    _write_confidences(outdir / "job_confidences.json")
    with pytest.raises(FileNotFoundError):
        plot_all_paes(outdir, "af3pae", dpi=20)
