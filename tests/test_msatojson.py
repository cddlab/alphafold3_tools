import json

import pytest

from alphafold3tools.msatojson import (
    generate_input_json_content,
    get_paired_and_unpaired_msa,
    get_residuelens_stoichiometries,
    int_id_to_str_id,
    split_a3msequences,
)


@pytest.fixture
def setup_lines():
    with open("./testfiles/testcomplexseqs.a3m", "r") as f:
        lines = f.readlines()
    yield lines


class TestMSA:
    def test_get_paired_and_unpaired_msa(self, setup_lines):
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_lines
        )
        cardinality = len(residue_lens)
        assert residue_lens == [139, 126]
        assert stoichiometries == [2, 3]
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            setup_lines, residue_lens, cardinality
        )
        assert len(unpairedmsas) == 2
        assert len(unpairedmsas[0]) == 8
        assert len(unpairedmsas[1]) == 10
        assert [len(v) for v in pairedmsas] == [6, 6]
        assert [len(v) for v in unpairedmsas] == [8, 10]
        assert pairedmsas[1][1].sequence.startswith("------------------FNAGDL")
        assert pairedmsas[1][5].sequence.startswith("-SHLSKTPHEHPLKFIEAFNSGDP")
        assert unpairedmsas[0][2].name.startswith(">UniRef100_N0CX87")
        assert unpairedmsas[0][3].name.startswith(">SRR5262245_37414285")
        assert unpairedmsas[1][0].name.startswith(">102\n")
        assert unpairedmsas[1][1].name.startswith(">UniRef100_UPI0005BB8534\t")

    def test_split_residues(self):
        residue_lens = [8, 7]
        line = "DEEPmINDDABCDEDaF"
        residues = split_a3msequences(residue_lens, line)
        assert residues[0] == "DEEPmINDD"
        assert residues[1] == "ABCDEDaF"

    def test_int_id_to_str_id(self):
        assert int_id_to_str_id(1) == "A"
        assert int_id_to_str_id(26) == "Z"
        assert int_id_to_str_id(27) == "AA"

    def test_generate_input_json_content(self, setup_lines):
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_lines
        )
        cardinality = len(residue_lens)
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            setup_lines, residue_lens, cardinality
        )
        content = generate_input_json_content(
            name="testcomplexseqs",
            cardinality=2,
            stoichiometries=stoichiometries,
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=False,
        )
        assert content["dialect"] == "alphafold3"
        assert content["sequences"][0]["protein"]["id"] == ["A", "B"]
        assert content["sequences"][1]["protein"]["id"] == ["C", "D", "E"]


@pytest.fixture
def setup_homomer_lines():
    with open("./testfiles/1bjp_6.a3m", "r") as f:
        lines = f.readlines()
    yield lines


class TestHomomerMSA:
    def test_get_paired_and_unpaired_msa(self, setup_homomer_lines):
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_homomer_lines
        )
        assert residue_lens == [62]
        assert stoichiometries == [6]
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            lines=setup_homomer_lines, residue_lens=residue_lens, cardinality=1
        )
        assert len(unpairedmsas) == 1
        assert len(unpairedmsas[0]) == 6
        assert [len(v) for v in unpairedmsas] == [6]
        assert pairedmsas == [[]]
        assert unpairedmsas[0][1].sequence.startswith(
            "PVVTIELWEGRTPEQKRELVRAVSSAISRVLGCPEEAVHVILHEVPKANWGIGGRLASEL--"
        )


@pytest.fixture
def setup_noheader_a3m():
    with open("./testfiles/1bjp_no_header.a3m", "r") as f:
        lines = f.readlines()
    yield lines


class TestNoHeaderMSA:
    def test_get_paired_and_unpaired_msa(self, setup_noheader_a3m):
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_noheader_a3m
        )
        assert residue_lens == [62]
        assert stoichiometries == [1]
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            lines=setup_noheader_a3m, residue_lens=residue_lens, cardinality=1
        )
        assert len(unpairedmsas) == 1
        assert len(unpairedmsas[0]) == 6
        assert [len(v) for v in unpairedmsas] == [6]
        assert pairedmsas == [[]]
        assert unpairedmsas[0][1].sequence.startswith(
            "PVVTIELWEGRTPEQKRELVRAVSSAISRVLGCPEEAVHVILHEVPKANWGIGGRLASEL--"
        )
