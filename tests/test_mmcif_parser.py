"""Tests for the template mmCIF written by mmcif_parser.mmcifcontent."""

import re
from pathlib import Path

import gemmi
import pytest

from alphafold3tools.searchtemplates import _parse_hit_metadata_with_gemmi
from alphafold3tools.structure.mmcif_parser import mmcifcontent
from alphafold3tools.structure.mmcif_utils import check_no_water_in_non_water_entities
from alphafold3tools.structure_stores import StructureStore

MMCIF_DIR = Path(__file__).parent.parent / "testfiles" / "mmcif_files"
# Written by gemmi for a pandas NaN; the mmCIF null is "?".
_NAN_TOKEN = re.compile(r"(?<![\w'\"])nan(?![\w'\"])")


def _read(name: str) -> str:
    return (MMCIF_DIR / f"{name}.cif").read_text()


def _store(pdb_id: str, mmcif: str) -> StructureStore:
    """A store holding a single mmCIF under its lowercase PDB ID, as hits use."""
    return StructureStore({pdb_id: mmcif})


def _values(block: gemmi.cif.Block, tag: str) -> list[str | None]:
    """Values of ``tag`` with "?" as None, as gemmi reads mmCIF categories."""
    category, item = tag.rsplit(".", 1)
    return block.get_mmcif_category(category)[item]


class TestSwappedLabelAndAuthorChainIds:
    """7YF5 has label chain A = author chain B and label chain B = author chain A."""

    @pytest.fixture(scope="class")
    def store(self) -> StructureStore:
        return _store("7yf5", _read("7YF5"))

    @pytest.mark.parametrize(("auth_chain_id", "label_chain_id"), [("A", "B"), ("B", "A")])
    def test_all_categories_describe_the_same_chain(self, store, auth_chain_id, label_chain_id):
        block = mmcifcontent("7yf5", auth_chain_id, store)

        assert set(_values(block, "_struct_asym.id")) == {label_chain_id}
        assert set(_values(block, "_atom_site.label_asym_id")) == {label_chain_id}
        assert set(_values(block, "_atom_site.auth_asym_id")) == {auth_chain_id}
        assert set(_values(block, "_pdbx_poly_seq_scheme.asym_id")) == {label_chain_id}
        assert set(_values(block, "_pdbx_poly_seq_scheme.pdb_strand_id")) == {auth_chain_id}
        assert block.find_value("_entity_poly.pdbx_strand_id") == auth_chain_id

    @pytest.mark.parametrize("auth_chain_id", ["A", "B"])
    def test_resolved_residues_of_scheme_match_atom_site(self, store, auth_chain_id):
        block = mmcifcontent("7yf5", auth_chain_id, store)

        scheme_resolved = {
            seq_id
            for seq_id, auth_seq_num in zip(
                _values(block, "_pdbx_poly_seq_scheme.seq_id"),
                _values(block, "_pdbx_poly_seq_scheme.auth_seq_num"),
                strict=True,
            )
            if auth_seq_num is not None
        }
        assert scheme_resolved == set(_values(block, "_atom_site.label_seq_id"))

    def test_unresolved_residues_have_null_author_number(self, store):
        block = mmcifcontent("7yf5", "B", store)

        auth_seq_nums = _values(block, "_pdbx_poly_seq_scheme.auth_seq_num")
        assert auth_seq_nums.count(None) == 11  # Author chain B has 11 unresolved residues.
        assert _NAN_TOKEN.search(block.as_string()) is None


class TestMicroheterogeneity:
    """1CBN has PRO/SER at residue 22 and LEU/ILE at residue 25."""

    @pytest.fixture(scope="class")
    def block(self) -> gemmi.cif.Block:
        return mmcifcontent("1cbn", "A", _store("1cbn", _read("1CBN")))

    @pytest.mark.parametrize(
        ("tag_prefix", "seq_id_item"), [("_pdbx_poly_seq_scheme", "seq_id"), ("_entity_poly_seq", "num")]
    )
    def test_one_residue_per_residue_number(self, block, tag_prefix, seq_id_item):
        seq_ids = _values(block, f"{tag_prefix}.{seq_id_item}")
        assert len(seq_ids) == len(set(seq_ids)) == 46

    @pytest.mark.parametrize(
        ("tag_prefix", "seq_id_item"), [("_pdbx_poly_seq_scheme", "seq_id"), ("_entity_poly_seq", "num")]
    )
    def test_residue_names_match_atom_site(self, block, tag_prefix, seq_id_item):
        names = dict(
            zip(_values(block, f"{tag_prefix}.{seq_id_item}"), _values(block, f"{tag_prefix}.mon_id"), strict=True)
        )
        atom_site_names = dict(
            zip(_values(block, "_atom_site.label_seq_id"), _values(block, "_atom_site.label_comp_id"), strict=True)
        )

        assert {seq_id: names[seq_id] for seq_id in atom_site_names} == atom_site_names
        assert (names["22"], names["25"]) == ("PRO", "LEU")


def _with_second_model(mmcif: str, x_shift: float) -> str:
    """Appends a copy of model 1, shifted along x, as model 2."""
    block = gemmi.cif.read_string(mmcif).sole_block()
    atom_site = block.get_mmcif_category("_atom_site", raw=True)
    num_atoms = len(atom_site["id"])
    model2 = {
        "id": [str(num_atoms + i + 1) for i in range(num_atoms)],
        "Cartn_x": [f"{float(x) + x_shift:.3f}" for x in atom_site["Cartn_x"]],
        "pdbx_PDB_model_num": ["2"] * num_atoms,
    }
    combined = {tag: values + model2.get(tag, values) for tag, values in atom_site.items()}
    block.set_mmcif_category("_atom_site", combined, raw=True)
    return block.as_string()


def test_only_first_model_is_written():
    single = mmcifcontent("1bjp", "A", _store("1bjp", _read("1BJP")))
    multi = mmcifcontent("1bjp", "A", _store("1bjp", _with_second_model(_read("1BJP"), x_shift=100.0)))

    assert set(_values(multi, "_atom_site.pdbx_PDB_model_num")) == {"1"}
    assert _values(multi, "_atom_site.Cartn_x") == _values(single, "_atom_site.Cartn_x")


class TestWater:
    """1BJP has 78 water molecules in entity 3."""

    def test_water_is_not_written(self):
        block = mmcifcontent("1bjp", "A", _store("1bjp", _read("1BJP")))

        assert "HOH" not in _values(block, "_atom_site.label_comp_id")

    def test_water_entity_is_accepted(self):
        check_no_water_in_non_water_entities(gemmi.cif.read_string(_read("1BJP")).sole_block())

    @pytest.fixture
    def bad_store(self) -> StructureStore:
        """1BJP whose water entity is declared as non-polymer."""
        water_entity = "3 water       nat water"
        mmcif = _read("1BJP")
        assert water_entity in mmcif
        return _store("1bjp", mmcif.replace(water_entity, "3 non-polymer nat water"))

    def test_non_water_entity_with_water_cannot_be_a_template(self, bad_store):
        with pytest.raises(ValueError, match="non-water entity has water molecules"):
            mmcifcontent("1bjp", "A", bad_store)

    def test_hit_on_non_water_entity_with_water_is_skipped(self, bad_store, warnings_log):
        assert _parse_hit_metadata_with_gemmi(bad_store, "1bjp", "A") == (None, None, None)
        assert any("non-water entity has water molecules" in message for message in warnings_log)
