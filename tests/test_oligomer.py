import textwrap
from pathlib import Path

import pytest

from alphafold3tools.structure.oligomer import (
    guess_homomer_count,
    guess_homomer_count_from_store,
)
from alphafold3tools.structure_stores import StructureStore

MMCIF_DIR = Path("testfiles/mmcif_files")

# Synthetic heteromeric structure: entity 1 -> chains A, B; entity 2 -> chain C.
# The single biological assembly contains all three chains (one identity
# operator). The homo-oligomer count of chain A must count only its own entity
# (A, B => 2), not the heteromeric partner C.
HETEROMER_CIF = textwrap.dedent("""\
    data_HET
    #
    loop_
    _entity.id
    _entity.type
    1 polymer
    2 polymer
    #
    loop_
    _struct_asym.id
    _struct_asym.entity_id
    A 1
    B 1
    C 2
    #
    loop_
    _pdbx_struct_assembly.id
    _pdbx_struct_assembly.details
    _pdbx_struct_assembly.method_details
    _pdbx_struct_assembly.oligomeric_details
    _pdbx_struct_assembly.oligomeric_count
    1 author_defined_assembly ? trimeric 3
    #
    loop_
    _pdbx_struct_assembly_gen.assembly_id
    _pdbx_struct_assembly_gen.oper_expression
    _pdbx_struct_assembly_gen.asym_id_list
    1 1 A,B,C
    #
    loop_
    _pdbx_struct_oper_list.id
    _pdbx_struct_oper_list.type
    _pdbx_struct_oper_list.name
    _pdbx_struct_oper_list.symmetry_operation
    _pdbx_struct_oper_list.matrix[1][1]
    _pdbx_struct_oper_list.matrix[1][2]
    _pdbx_struct_oper_list.matrix[1][3]
    _pdbx_struct_oper_list.vector[1]
    _pdbx_struct_oper_list.matrix[2][1]
    _pdbx_struct_oper_list.matrix[2][2]
    _pdbx_struct_oper_list.matrix[2][3]
    _pdbx_struct_oper_list.vector[2]
    _pdbx_struct_oper_list.matrix[3][1]
    _pdbx_struct_oper_list.matrix[3][2]
    _pdbx_struct_oper_list.matrix[3][3]
    _pdbx_struct_oper_list.vector[3]
    1 'identity operation' 1_555 x,y,z 1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
    #
    loop_
    _atom_site.group_PDB
    _atom_site.id
    _atom_site.type_symbol
    _atom_site.label_atom_id
    _atom_site.label_alt_id
    _atom_site.label_comp_id
    _atom_site.label_asym_id
    _atom_site.label_entity_id
    _atom_site.label_seq_id
    _atom_site.pdbx_PDB_ins_code
    _atom_site.Cartn_x
    _atom_site.Cartn_y
    _atom_site.Cartn_z
    _atom_site.occupancy
    _atom_site.B_iso_or_equiv
    _atom_site.auth_seq_id
    _atom_site.auth_comp_id
    _atom_site.auth_asym_id
    _atom_site.auth_atom_id
    _atom_site.pdbx_PDB_model_num
    ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.0 20.0 1 ALA A CA 1
    ATOM 2 C CA . ALA B 1 1 ? 1.0 0.0 0.0 1.0 20.0 1 ALA B CA 1
    ATOM 3 C CA . GLY C 2 1 ? 2.0 0.0 0.0 1.0 20.0 1 GLY C CA 1
    """)


def _read(pdb_id: str) -> str:
    return (MMCIF_DIR / f"{pdb_id}.cif").read_text()


@pytest.mark.skipif(
    not (MMCIF_DIR / "1BJP.cif").exists(), reason="test mmCIF files not available"
)
class TestGuessHomomerCountRealStructures:
    def test_homohexamer_1bjp(self):
        mmcif = _read("1BJP")
        # 1BJP is a homohexamer; every polymer chain (A-E) maps to a 6-copy
        # biological assembly.
        for chain_id in ["A", "B", "C", "D", "E"]:
            assert guess_homomer_count(mmcif, chain_id) == 6

    def test_homodimer_6w81(self):
        mmcif = _read("6W81")
        assert guess_homomer_count(mmcif, "A") == 2
        assert guess_homomer_count(mmcif, "B") == 2

    def test_monomers(self):
        assert guess_homomer_count(_read("4ZZ4"), "A") == 1
        assert guess_homomer_count(_read("7KYZ"), "A") == 1

    def test_from_store_lowercase_pdb_id(self):
        store = StructureStore(str(MMCIF_DIR))
        # Hits use lower-cased PDB IDs; the store reads "1bjp.cif"
        # (case-insensitive FS matches "1BJP.cif").
        assert guess_homomer_count_from_store(store, "1bjp", "A") == 6


class TestGuessHomomerCountSynthetic:
    def test_heteromer_counts_only_query_entity(self):
        # Chain A belongs to entity 1 (chains A, B) -> 2 copies.
        assert guess_homomer_count(HETEROMER_CIF, "A") == 2
        # Chain C belongs to entity 2 (chain C only) -> 1 copy.
        assert guess_homomer_count(HETEROMER_CIF, "C") == 1

    def test_missing_chain_returns_one(self):
        assert guess_homomer_count(HETEROMER_CIF, "Z") == 1

    def test_malformed_mmcif_returns_one(self):
        assert guess_homomer_count("not a real mmcif", "A") == 1

    def test_from_store_missing_structure_returns_one(self):
        store = StructureStore({"1bjp": HETEROMER_CIF})
        assert guess_homomer_count_from_store(store, "nope", "A") == 1
