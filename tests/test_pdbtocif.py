from pathlib import Path

import gemmi
import pytest

from alphafold3tools.pdbtocif import pdb_to_cif

TESTFILES = Path(__file__).parent.parent / "testfiles" / "pdbtocif"


def _read_seqres(path: Path) -> list[tuple[str, str]]:
    lines = path.read_text().splitlines()
    return [(lines[i], lines[i + 1]) for i in range(0, len(lines), 2)]


@pytest.fixture
def converted_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Convert the same PDB with and without SEQRES records."""
    pdb_text = (TESTFILES / "P9WIC5_v6.pdb").read_text()
    no_seqres_text = "\n".join(line for line in pdb_text.splitlines() if not line.startswith("SEQRES"))
    with_dir = tmp_path / "with_seqres"
    without_dir = tmp_path / "without_seqres"
    with_dir.mkdir()
    without_dir.mkdir()
    (with_dir / "test.pdb").write_text(pdb_text)
    (without_dir / "test.pdb").write_text(no_seqres_text + "\n")

    for d in (with_dir, without_dir):
        pdb_to_cif(str(d / "test.pdb"), str(d / "test.cif"), default_pdb_id="test")
    return with_dir, without_dir


def test_pdb_without_seqres_fills_entity_sequence(converted_pair):
    with_dir, without_dir = converted_pair
    expected = gemmi.cif.read(str(with_dir / "test.cif")).sole_block()
    block = gemmi.cif.read(str(without_dir / "test.cif")).sole_block()

    tag = "_entity_poly.pdbx_seq_one_letter_code_can"
    seq = gemmi.cif.as_string(block.find_value(tag)).replace("\n", "")
    expected_seq = gemmi.cif.as_string(expected.find_value(tag)).replace("\n", "")
    assert seq == expected_seq
    assert len(seq) > 0


def test_pdb_without_seqres_assigns_label_seq_id(converted_pair):
    _, without_dir = converted_pair
    block = gemmi.cif.read(str(without_dir / "test.cif")).sole_block()
    label_seq_ids = list(block.find_values("_atom_site.label_seq_id"))
    assert label_seq_ids
    assert "." not in label_seq_ids


def test_pdb_without_seqres_writes_seqres_txt(converted_pair):
    with_dir, without_dir = converted_pair
    records = _read_seqres(without_dir / "pdb_seqres.txt")
    expected = _read_seqres(with_dir / "pdb_seqres.txt")
    assert records == expected
    for header, seq in records:
        assert f"length:{len(seq)}" in header
        assert len(seq) > 0
