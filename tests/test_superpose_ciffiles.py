import shutil
from pathlib import Path

import gemmi
import pytest

from alphafold3tools.superpose_ciffiles import superpose_ciffiles

MODEL_CIF = Path(__file__).parent.parent / "testfiles" / "ipsae" / "af3" / "tlxi_tlxi_model.cif"


def _make_af3_outdir(outdir: Path, seeds: list[int], nsamples: int) -> Path:
    for seed in seeds:
        for j in range(nsamples):
            subdir = outdir / f"seed-{seed}_sample-{j}"
            subdir.mkdir(parents=True)
            shutil.copy(MODEL_CIF, subdir / f"{outdir.name}_seed-{seed}_sample-{j}_model.cif")
    return outdir


@pytest.fixture
def af3_outdir(tmp_path: Path) -> Path:
    """Build a minimal AlphaFold3 output directory with one seed and five samples."""
    return _make_af3_outdir(tmp_path / "OPCL1", seeds=[1], nsamples=5)


def _block_names(ciffile: Path) -> list[str]:
    return [block.name for block in gemmi.cif.read(str(ciffile))]


def test_block_names_default_to_input_dirname(af3_outdir: Path):
    superpose_ciffiles(str(af3_outdir))
    outfile = af3_outdir / "OPCL1_superposed.cif"
    assert _block_names(outfile) == [f"OPCL1_seed-1_sample-{j}" for j in range(5)]


def test_block_names_default_with_trailing_slash(af3_outdir: Path):
    superpose_ciffiles(str(af3_outdir) + "/")
    outfile = af3_outdir / "OPCL1_superposed.cif"
    assert _block_names(outfile)[0] == "OPCL1_seed-1_sample-0"


def test_block_names_use_objectname(af3_outdir: Path, tmp_path: Path):
    outfile = tmp_path / "out.cif"
    superpose_ciffiles(str(af3_outdir), str(outfile), objectname="objectname")
    assert _block_names(outfile) == [f"objectname_seed-1_sample-{j}" for j in range(5)]


def test_superposes_all_samples_beyond_five(tmp_path: Path):
    outdir = _make_af3_outdir(tmp_path / "OPCL1", seeds=[1, 2], nsamples=12)
    superpose_ciffiles(str(outdir))
    # every sample of every seed, in numerical order (sample-10 after sample-9)
    expected = [f"OPCL1_seed-{seed}_sample-{j}" for seed in (1, 2) for j in range(12)]
    assert _block_names(outdir / "OPCL1_superposed.cif") == expected


def test_superposes_fewer_than_five_samples(tmp_path: Path):
    outdir = _make_af3_outdir(tmp_path / "OPCL1", seeds=[1], nsamples=2)
    superpose_ciffiles(str(outdir))
    assert _block_names(outdir / "OPCL1_superposed.cif") == ["OPCL1_seed-1_sample-0", "OPCL1_seed-1_sample-1"]


def test_no_sample_dirs_raises(tmp_path: Path):
    outdir = tmp_path / "OPCL1"
    outdir.mkdir()
    with pytest.raises(FileNotFoundError):
        superpose_ciffiles(str(outdir))
