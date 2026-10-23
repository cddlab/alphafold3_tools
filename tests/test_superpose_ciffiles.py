import shutil
from pathlib import Path

import gemmi
import pytest

from alphafold3tools.superpose_ciffiles import superpose_ciffiles

MODEL_CIF = Path(__file__).parent.parent / "testfiles" / "ipsae" / "af3" / "tlxi_tlxi_model.cif"


@pytest.fixture
def af3_outdir(tmp_path: Path) -> Path:
    """Build a minimal AlphaFold3 output directory with one seed and five samples."""
    outdir = tmp_path / "OPCL1"
    for j in range(5):
        subdir = outdir / f"seed-1_sample-{j}"
        subdir.mkdir(parents=True)
        shutil.copy(MODEL_CIF, subdir / f"OPCL1_seed-1_sample-{j}_model.cif")
    return outdir


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
