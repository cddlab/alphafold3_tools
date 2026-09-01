import json
import os

import pytest
from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.modjson import (
    _resolve_chain_a3m,
    add_ligand,
    add_templates,
    add_userccd,
    fix_sequence_ids,
    modify_name,
    purge_ligand,
    read_json_data,
    remove_ccdcodes,
    write_json_data,
)

log_setup()


def test_read_json_data():
    json_path = "testfiles/read_data.json"
    data = read_json_data(json_path)
    assert isinstance(data, dict)
    assert "dialect" in data
    assert data["dialect"] == "alphafold3"
    assert data["sequences"][0]["protein"]["id"] == ["A", "B"]


def test_write_json_data(tmp_path):
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [],
        "modelSeeds": [1],
        "bondedAtomPairs": None,
        "userCCD": None,
    }
    output_file = tmp_path / "output.json"
    write_json_data(output_file, data)

    with open(output_file, "r") as file:
        written_data = json.load(file)

    assert written_data == data


def test_remove_ccdcodes():
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "modifications": [],
                    "unpairedMsa": ">101\nMSNTNQGPVTVLGL",
                    "pairedMsa": "",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": ["E", "F"], "ccdCodes": ["NAP"]}},
            {"ligand": {"id": ["G", "H"], "ccdCodes": ["NAD"]}},
        ],
    }
    ligands_to_remove = ["PRD", "NAD"]
    removed_data = remove_ccdcodes(data, ligands_to_remove)
    expected_data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "modifications": [],
                    "unpairedMsa": ">101\nMSNTNQGPVTVLGL",
                    "pairedMsa": "",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["E", "F"], "ccdCodes": ["NAP"]}},
        ],
    }

    assert removed_data == expected_data


def test_add_ligand():
    data1 = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": "E", "ccdCodes": ["ATP"]}},
        ],
    }
    added_data1 = add_ligand(data1, "smiles", "CCO", 1)
    expected_data1 = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": "E", "ccdCodes": ["ATP"]}},
            {"ligand": {"id": ["A"], "smiles": "CCO"}},
        ],
    }
    assert added_data1 == expected_data1
    data2 = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": "E", "ccdCodes": ["ATP"]}},
        ],
    }
    added_data2 = add_ligand(data2, "ccdCodes", "PRD", 2)
    expected_data2 = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": "E", "ccdCodes": ["ATP"]}},
            {"ligand": {"id": ["A", "B"], "ccdCodes": ["PRD"]}},
        ],
    }
    assert added_data2 == expected_data2


def test_purge_ligands():
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": ["E", "F"], "smiles": "CCO"}},
        ],
    }
    purged_data = purge_ligand(data)
    expected_data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
        ],
    }
    assert purged_data == expected_data


def test_fix_sequence_ids():
    data1 = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["E", "F"], "ccdCodes": ["NAP"]}},
            {"ligand": {"id": "Z", "ccdCodes": ["NAD"]}},
        ],
    }
    fixed_data1 = fix_sequence_ids(data1)
    expected_data1 = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["NAP"]}},
            {"ligand": {"id": "E", "ccdCodes": ["NAD"]}},
        ],
    }
    assert fixed_data1 == expected_data1


def test_modify_name():
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "oldname",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["NAP"]}},
        ],
    }
    new_name = "newname"
    modified_data = modify_name(data, new_name)
    expected_data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "newname",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["NAP"]}},
        ],
    }
    assert modified_data == expected_data


def test_fix_sequence_ids_with_multiple_ligands():
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["E", "F"], "ccdCodes": ["NAP"]}},
            {"ligand": {"id": "Z", "ccdCodes": ["NAD"]}},
            {"ligand": {"id": ["G", "H"], "ccdCodes": ["ATP"]}},
        ],
    }
    fixed_data = fix_sequence_ids(data)
    expected_data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["NAP"]}},
            {"ligand": {"id": "E", "ccdCodes": ["NAD"]}},
            {"ligand": {"id": ["F", "G"], "ccdCodes": ["ATP"]}},
        ],
    }
    assert fixed_data == expected_data


def test_add_ligand_with_existing_ids():
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": "E", "ccdCodes": ["ATP"]}},
        ],
    }
    added_data = add_ligand(data, "smiles", "CCO", 1)
    expected_data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": "E", "ccdCodes": ["ATP"]}},
            {"ligand": {"id": ["A"], "smiles": "CCO"}},
        ],
    }
    assert added_data == expected_data


def test_remove_ccdcodes_with_no_match():
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "testprotein",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["C", "D"], "ccdCodes": ["PRD"]}},
            {"ligand": {"id": ["E", "F"], "ccdCodes": ["NAP"]}},
        ],
    }
    ligands_to_remove = ["XYZ"]
    removed_data = remove_ccdcodes(data, ligands_to_remove)
    expected_data = data  # No change expected
    assert removed_data == expected_data


def test_add_userccd(tmp_path):
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "test1",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["E", "F"], "ccdCodes": ["NAP"]}},
        ],
        "userCCD": "",
    }

    userccd_content = "data_MY-ORO\n# \n_chem_comp.id MY-ORO\n_chem_comp.name 'MY-ORO'"
    userccd_file = tmp_path / "userccd1.cif"
    userccd_file.write_text(userccd_content)

    new_data = add_userccd(data, [str(userccd_file)])

    expected_userccd = (
        "data_MY-ORO\n# \n_chem_comp.id MY-ORO\n_chem_comp.name 'MY-ORO'## \n"
    )
    assert new_data["userCCD"] == expected_userccd


def test_add_userccd_multiple_files(tmp_path):
    data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": "test1",
        "sequences": [
            {
                "protein": {
                    "id": ["A", "B"],
                    "sequence": "MSNTNQGPVTVLGL",
                    "templates": [],
                }
            },
            {"ligand": {"id": ["E", "F"], "ccdCodes": ["NAP"]}},
        ],
        "userCCD": "",
    }

    userccd_content1 = "data_MY-ORO\n# \n_chem_comp.id MY-ORO\n_chem_comp.name 'MY-ORO'"
    userccd_content2 = "data_MY-FOO\n# \n_chem_comp.id MY-FOO\n_chem_comp.name 'MY-FOO'"
    userccd_file1 = tmp_path / "userccd1.cif"
    userccd_file2 = tmp_path / "userccd2.cif"
    userccd_file1.write_text(userccd_content1)
    userccd_file2.write_text(userccd_content2)

    new_data = add_userccd(data, [str(userccd_file1), str(userccd_file2)])

    expected_userccd = (
        "data_MY-ORO\n# \n_chem_comp.id MY-ORO\n_chem_comp.name 'MY-ORO'## \n"
        "data_MY-FOO\n# \n_chem_comp.id MY-FOO\n_chem_comp.name 'MY-FOO'## \n"
    )
    assert new_data["userCCD"] == expected_userccd


class TestAddTemplates:
    """Template search added to modjson via add_templates().

    The search pipeline (HMMER + PDB databases) is fully monkeypatched, so these
    tests need neither a real database nor HMMER binaries. modjson imports the
    search helpers lazily, so the source modules are patched.
    """

    def _valid_search_paths(self, tmp_path):
        """Create dummy but existing pdb dir / seqres / hmmbuild paths so the
        early template-search path validation passes."""
        pdb_dir = tmp_path / "mmcif_files"
        pdb_dir.mkdir()
        seqres = tmp_path / "pdb_seqres.txt"
        seqres.write_text("")
        hmmbuild = tmp_path / "hmmbuild"
        hmmbuild.write_text("")
        return str(pdb_dir), str(seqres), str(hmmbuild)

    def _fake_templates(self):
        return [{"mmcif": "DUMMY", "queryIndices": [0], "templateIndices": [0]}]

    def _protein_data(self, templates, msa_key="unpairedMsa"):
        return {
            "dialect": "alphafold3",
            "version": 1,
            "name": "job",
            "sequences": [
                {
                    "protein": {
                        "id": ["A", "B"],
                        "sequence": "PIAQIHILEGRSDEQKE",
                        "modifications": [],
                        msa_key: ">query\nPIAQIHILEGRSDEQKE\n",
                        "pairedMsa": "",
                        "templates": templates,
                    }
                }
            ],
            "modelSeeds": [1],
            "bondedAtomPairs": None,
            "userCCD": None,
        }

    def _patch_search(self, monkeypatch, templates):
        import alphafold3tools.searchtemplates as st

        def fake_search(**kwargs):
            return templates

        monkeypatch.setattr(st, "search_templates", fake_search)

    def test_fills_empty_templates(self, monkeypatch, tmp_path):
        pdb, seqres, hmmbuild = self._valid_search_paths(tmp_path)
        templates = self._fake_templates()
        self._patch_search(monkeypatch, templates)
        data = self._protein_data([])

        new_data = add_templates(
            data,
            pdb_database_path=pdb,
            seqres_database_path=seqres,
            hmmbuild_binary_path=hmmbuild,
            hmmsearch_binary_path=hmmbuild,
        )
        assert new_data["sequences"][0]["protein"]["templates"] == templates
        # Original untouched (deepcopy semantics).
        assert data["sequences"][0]["protein"]["templates"] == []

    def test_preserves_existing_without_overwrite(self, monkeypatch, tmp_path):
        pdb, seqres, hmmbuild = self._valid_search_paths(tmp_path)
        existing = [{"mmcif": "OLD", "queryIndices": [1], "templateIndices": [2]}]
        self._patch_search(monkeypatch, self._fake_templates())
        data = self._protein_data(existing)

        new_data = add_templates(
            data,
            pdb_database_path=pdb,
            seqres_database_path=seqres,
            hmmbuild_binary_path=hmmbuild,
            hmmsearch_binary_path=hmmbuild,
            overwrite=False,
        )
        assert new_data["sequences"][0]["protein"]["templates"] == existing

    def test_overwrites_existing_with_overwrite(self, monkeypatch, tmp_path):
        pdb, seqres, hmmbuild = self._valid_search_paths(tmp_path)
        existing = [{"mmcif": "OLD", "queryIndices": [1], "templateIndices": [2]}]
        new_templates = self._fake_templates()
        self._patch_search(monkeypatch, new_templates)
        data = self._protein_data(existing)

        new_data = add_templates(
            data,
            pdb_database_path=pdb,
            seqres_database_path=seqres,
            hmmbuild_binary_path=hmmbuild,
            hmmsearch_binary_path=hmmbuild,
            overwrite=True,
        )
        assert new_data["sequences"][0]["protein"]["templates"] == new_templates

    def test_resolve_chain_a3m_prefers_inline(self):
        protein = {
            "sequence": "AAAA",
            "unpairedMsa": ">q\nAAAA\n",
            "unpairedMsaPath": "/nonexistent/path.a3m",
        }
        assert _resolve_chain_a3m(protein) == ">q\nAAAA\n"

    def test_resolve_chain_a3m_reads_path(self, tmp_path):
        a3m = tmp_path / "chain.a3m"
        a3m.write_text(">q\nCCCC\n>hit\nCCGC\n")
        protein = {"sequence": "CCCC", "unpairedMsaPath": str(a3m)}
        assert _resolve_chain_a3m(protein) == ">q\nCCCC\n>hit\nCCGC\n"

    def test_resolve_chain_a3m_falls_back_to_sequence(self):
        protein = {"sequence": "MKV", "unpairedMsa": "", "pairedMsa": ""}
        assert _resolve_chain_a3m(protein) == ">query\nMKV\n"

    def test_guess_copies_overrides_id_length(self, monkeypatch, tmp_path):
        pdb, seqres, hmmbuild = self._valid_search_paths(tmp_path)
        templates = self._fake_templates()

        import alphafold3tools.searchtemplates as st
        import alphafold3tools.structure.oligomer as oligomer

        def fake_search_with_hits(**kwargs):
            return templates, [("1bjp", "A")]

        def fake_guess(store, pdb_id, chain_id):
            assert (pdb_id, chain_id) == ("1bjp", "A")
            return 3

        monkeypatch.setattr(st, "search_templates_with_hits", fake_search_with_hits)
        monkeypatch.setattr(oligomer, "guess_homomer_count_from_store", fake_guess)

        data = self._protein_data([])  # starts as 2 copies (A, B)
        new_data = add_templates(
            data,
            pdb_database_path=pdb,
            seqres_database_path=seqres,
            hmmbuild_binary_path=hmmbuild,
            hmmsearch_binary_path=hmmbuild,
            guess_copies=True,
        )
        # Provisional id list has the guessed length; fix_sequence_ids renumbers.
        assert len(new_data["sequences"][0]["protein"]["id"]) == 3
        fixed = fix_sequence_ids(new_data)
        assert fixed["sequences"][0]["protein"]["id"] == ["A", "B", "C"]
        assert fixed["sequences"][0]["protein"]["templates"] == templates

    def test_missing_seqres_raises(self, tmp_path):
        pdb_dir = tmp_path / "mmcif_files"
        pdb_dir.mkdir()
        hmmbuild = tmp_path / "hmmbuild"
        hmmbuild.write_text("")
        data = self._protein_data([])
        with pytest.raises(FileNotFoundError):
            add_templates(
                data,
                pdb_database_path=str(pdb_dir),
                seqres_database_path=str(tmp_path / "missing_seqres.txt"),
                hmmbuild_binary_path=str(hmmbuild),
                hmmsearch_binary_path=str(hmmbuild),
            )


if __name__ == "__main__":
    pytest.main()
