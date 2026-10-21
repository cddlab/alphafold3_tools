from pathlib import Path

import gemmi
import pytest
import rdkit.Chem

from alphafold3tools.sdftoccd import (
    candidate_charges,
    convert_sdf_to_ccd,
    mol2_element,
    neutralize_oxyanions,
    read_mol2_molecule,
    read_sdf_molecule,
)

TESTFILES = Path(__file__).parent.parent / "testfiles"


def _convert(sdffile: Path, mol_name: str) -> gemmi.cif.Block:
    return gemmi.cif.read_string(convert_sdf_to_ccd(str(sdffile), mol_name)).sole_block()


def _loop_rows(block: gemmi.cif.Block, category: str, tags: list[str]) -> list[tuple[str, ...]]:
    table = block.find(category, tags)
    return [tuple(row[i] for i in range(len(tags))) for row in table]


def test_candidate_charges_ordered_by_distance():
    assert candidate_charges(0, max_shift=2) == [0, -1, 1, -2, 2]
    assert candidate_charges(-1, max_shift=1) == [-1, -2, 0]


def test_convert_sanitizable_sdf(warnings_log):
    block = _convert(TESTFILES / "orobanchol.sdf", "ORO")

    assert block.find_value("_chem_comp.id") == "ORO"
    assert len(block.find_loop("_chem_comp_atom.atom_id")) == 25
    assert not any("re-assigned" in m for m in warnings_log)


def test_convert_gaussview_style_sdf(warnings_log):
    """Carboxylate drawn as C(=O)=O and non-ring aromatic bonds are repaired."""
    block = _convert(TESTFILES / "sorbate_gaussview.sdf", "SOR")

    # The carboxylate O(-1) is written as neutral OH
    assert block.find_value("_chem_comp.formula") == "C6H8O2"
    assert set(block.find_loop("_chem_comp_atom.charge")) == {"0"}
    orders = list(block.find_loop("_chem_comp_bond.value_order"))
    assert orders.count("DOUB") == 3
    assert "AROM" not in orders
    smiles = gemmi.cif.as_string(block.find_value("_pdbx_chem_comp_descriptor.descriptor"))
    assert smiles == "C/C=C/C=C/C(=O)O"
    assert any("total charge: -1" in m for m in warnings_log)


def test_read_sdf_molecule_keeps_hydrogens():
    mol = read_sdf_molecule(str(TESTFILES / "sorbate_gaussview.sdf"), removeHs=False)
    assert mol.GetNumAtoms() == 15


def _write_partial_sdf(tmp_path: Path, n_atoms: int) -> Path:
    """Write sorbate_gaussview.sdf truncated to its first n_atoms atoms."""
    lines = (TESTFILES / "sorbate_gaussview.sdf").read_text().splitlines()
    atoms = lines[4 : 4 + n_atoms]
    bonds = [line for line in lines[19:33] if int(line.split()[1]) <= n_atoms]
    counts = f"{len(atoms):3d}{len(bonds):3d}  0  0  0  0  0  0  0  0  0"
    sdffile = tmp_path / "partial.sdf"
    sdffile.write_text("\n".join([*lines[:3], counts, *atoms, *bonds, "M  END", "$$$$", ""]))
    return sdffile


def test_read_sdf_molecule_without_hydrogens(tmp_path: Path):
    """Bond orders cannot be determined when hydrogens are missing."""
    with pytest.raises(ValueError, match="no hydrogens"):
        read_sdf_molecule(str(_write_partial_sdf(tmp_path, 8)))


@pytest.mark.parametrize(
    ("atom_type", "element"),
    [
        ("C.ar", "C"),
        ("O.co2", "O"),
        ("N.am", "N"),
        ("H", "H"),
        ("Cl", "Cl"),
        ("c3", "C"),
        ("ca", "C"),
        ("na", "N"),
        ("ha", "H"),
        ("oh", "O"),
        ("cl", "Cl"),
        ("br", "Br"),
    ],
)
def test_mol2_element(atom_type: str, element: str):
    assert mol2_element(atom_type) == element


def test_mol2_element_unknown_type():
    with pytest.raises(ValueError, match="Unknown atom type"):
        mol2_element("Du")


def test_convert_gaff_mol2_keeps_names_coordinates_and_order(warnings_log):
    """Atom names, coordinates and the order of atoms and bonds follow the mol2 file."""
    block = _convert(TESTFILES / "sorbate_gaff.mol2", "SOR")

    atoms = _loop_rows(
        block,
        "_chem_comp_atom.",
        [
            "atom_id",
            "type_symbol",
            "charge",
            "pdbx_model_Cartn_x_ideal",
            "pdbx_model_Cartn_y_ideal",
            "pdbx_model_Cartn_z_ideal",
            "pdbx_ordinal",
        ],
    )
    assert atoms[:3] == [
        ("OXA", "O", "0", "4.267", "-0.316", "0.857", "1"),
        ("CAF", "C", "0", "3.290", "-0.245", "0.058", "2"),
        ("OXB", "O", "0", "3.555", "-0.274", "-1.305", "3"),
    ]
    assert [a[0] for a in atoms] == ["OXA", "CAF", "OXB", "CAE", "CAD", "CAC", "CAB", "CAM"]

    bonds = _loop_rows(block, "_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order", "pdbx_ordinal"])
    assert bonds == [
        ("OXA", "CAF", "DOUB", "1"),
        ("CAF", "OXB", "SING", "2"),
        ("CAF", "CAE", "SING", "3"),
        ("CAE", "CAD", "DOUB", "4"),
        ("CAC", "CAD", "SING", "5"),  # written as '6 5' in the mol2 file
        ("CAC", "CAB", "DOUB", "6"),
        ("CAB", "CAM", "SING", "7"),
    ]
    assert block.find_value("_chem_comp.formula") == "C6H8O2"
    # The total charge is guessed from the sum of the partial charges
    assert any("total charge: -1" in m for m in warnings_log)


def test_convert_sybyl_mol2_with_aromatic_ring():
    block = _convert(TESTFILES / "benzoate_sybyl.mol2", "BEZ")

    atoms = _loop_rows(block, "_chem_comp_atom.", ["atom_id", "pdbx_aromatic_flag"])
    assert atoms == [("O1", "N"), ("C7", "N"), ("O2", "N")] + [(f"C{i}", "Y") for i in range(1, 7)]
    bonds = _loop_rows(block, "_chem_comp_bond.", ["atom_id_1", "atom_id_2", "value_order"])
    assert bonds[:3] == [("O1", "C7", "DOUB"), ("C7", "O2", "SING"), ("C7", "C1", "SING")]
    assert {b[2] for b in bonds[3:]} == {"AROM"}
    assert block.find_value("_chem_comp.formula") == "C7H6O2"


def test_read_mol2_molecule_keeps_hydrogens():
    mol = read_mol2_molecule(str(TESTFILES / "sorbate_gaff.mol2"), removeHs=False)
    assert mol.GetNumAtoms() == 15
    assert mol.GetAtomWithIdx(14).GetProp("name") == "HM3"


def test_read_mol2_molecule_duplicate_names(tmp_path: Path):
    text = (TESTFILES / "sorbate_gaff.mol2").read_text().replace(" OXB ", " OXA ")
    mol2file = tmp_path / "dup.mol2"
    mol2file.write_text(text)
    with pytest.raises(ValueError, match="must be unique.*OXA"):
        read_mol2_molecule(str(mol2file))


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("CC(=O)[O-]", "CC(=O)O"),
        ("OP(=O)([O-])[O-]", "O=P(O)(O)O"),
        ("C[N+](=O)[O-]", "C[N+](=O)[O-]"),
        ("CC(=O)O", "CC(=O)O"),
    ],
)
def test_neutralize_oxyanions(smiles: str, expected: str):
    mol = neutralize_oxyanions(rdkit.Chem.MolFromSmiles(smiles))
    assert rdkit.Chem.MolToSmiles(mol) == expected
