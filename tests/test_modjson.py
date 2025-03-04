import json
import os

import pytest
from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.modjson import (
    add_ligand,
    add_userccd,
    fix_sequence_ids,
    modify_name,
    purge_ligands,
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
    purged_data = purge_ligands(data)
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


if __name__ == "__main__":
    pytest.main()
