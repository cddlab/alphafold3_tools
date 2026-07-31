import json
from pathlib import Path

import pytest

import alphafold3tools.msatojson as msatojson
from alphafold3tools.msatojson import (
    Seq,
    convert_msas_to_str,
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
def setup_query_header_a3m():
    with open("./testfiles/Q9I1F6-F1-msa_v6.a3m", "r") as f:
        lines = f.readlines()
    yield lines


@pytest.fixture
def setup_noheader_a3m():
    with open("./testfiles/1bjp_no_header.a3m", "r") as f:
        lines = f.readlines()
    yield lines


class TestQueryHeaderMSA:
    """a3m files whose first sequence header is '>query' (e.g. MMseqs2 web server output)
    must be treated as unpaired-only MSA, identical to the '>101' code path."""

    def test_get_paired_and_unpaired_msa(self, setup_query_header_a3m):
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_query_header_a3m
        )
        assert stoichiometries == [1]
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            lines=setup_query_header_a3m, residue_lens=residue_lens, cardinality=1
        )
        assert pairedmsas == [[]]
        assert len(unpairedmsas) == 1
        assert len(unpairedmsas[0]) == 6
        assert unpairedmsas[0][0].name == ">query\n"

    def test_generate_json_has_unpaired_msa(self, setup_query_header_a3m):
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_query_header_a3m
        )
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            lines=setup_query_header_a3m, residue_lens=residue_lens, cardinality=1
        )
        content = generate_input_json_content(
            name="Q9I1F6",
            cardinality=1,
            stoichiometries=stoichiometries,
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=False,
        )
        prot = content["sequences"][0]["protein"]
        assert prot["pairedMsa"] == ""
        assert prot["unpairedMsa"].startswith(">query\n")
        assert prot["unpairedMsa"].count(">") == 6


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


class TestWriteMsaPath:
    """Tests for --write_msapath: MSA content is written to separate .a3m files
    and JSON uses unpairedMsaPath/pairedMsaPath keys instead of inline content."""

    def test_single_chain_uses_original_path(self, setup_homomer_lines, tmp_path):
        """Single-chain a3m (no paired MSA): unpairedMsaPath must equal the
        original input file path; pairedMsa must be an empty string (not
        pairedMsaPath, which AlphaFold3 rejects when empty)."""
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_homomer_lines
        )
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            lines=setup_homomer_lines, residue_lens=residue_lens, cardinality=1
        )
        inputmsafile = Path("./testfiles/1bjp_6.a3m")
        content = generate_input_json_content(
            name="1bjp_6",
            cardinality=1,
            stoichiometries=stoichiometries,
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=False,
            write_msapath=True,
            inputmsafile=inputmsafile,
            msa_output_dir=tmp_path,
        )
        prot = content["sequences"][0]["protein"]
        assert prot["unpairedMsaPath"] == str(inputmsafile)
        assert prot["pairedMsa"] == ""
        assert "unpairedMsa" not in prot
        assert "pairedMsaPath" not in prot

    def test_multi_chain_writes_msa_files(self, setup_lines, tmp_path):
        """Multi-chain a3m (paired MSA present): new .a3m files are written to
        msa_output_dir and JSON references them via *MsaPath keys."""
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_lines
        )
        cardinality = len(residue_lens)
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            setup_lines, residue_lens, cardinality
        )
        content = generate_input_json_content(
            name="testcomplexseqs",
            cardinality=cardinality,
            stoichiometries=stoichiometries,
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=False,
            write_msapath=True,
            inputmsafile=Path("./testfiles/testcomplexseqs.a3m"),
            msa_output_dir=tmp_path,
        )
        # chain 0 (stoichiometry 2, ids A/B) → suffix "A"
        prot0 = content["sequences"][0]["protein"]
        assert "unpairedMsa" not in prot0
        assert "pairedMsa" not in prot0
        unpaired0 = Path(prot0["unpairedMsaPath"])
        paired0 = Path(prot0["pairedMsaPath"])
        assert unpaired0.name == "testcomplexseqs_unpaired_A.a3m"
        assert paired0.name == "testcomplexseqs_paired_A.a3m"
        assert unpaired0.exists()
        assert paired0.exists()
        # chain 1 (stoichiometry 3, ids C/D/E) → suffix "C"
        prot1 = content["sequences"][1]["protein"]
        unpaired1 = Path(prot1["unpairedMsaPath"])
        paired1 = Path(prot1["pairedMsaPath"])
        assert unpaired1.name == "testcomplexseqs_unpaired_C.a3m"
        assert paired1.name == "testcomplexseqs_paired_C.a3m"
        assert unpaired1.exists()
        assert paired1.exists()

    def test_written_files_contain_correct_content(self, setup_lines, tmp_path):
        """Content of written .a3m files must match convert_msas_to_str output."""
        residue_lens, stoichiometries = get_residuelens_stoichiometries(
            lines=setup_lines
        )
        cardinality = len(residue_lens)
        pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
            setup_lines, residue_lens, cardinality
        )
        generate_input_json_content(
            name="testcomplexseqs",
            cardinality=cardinality,
            stoichiometries=stoichiometries,
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=False,
            write_msapath=True,
            inputmsafile=Path("./testfiles/testcomplexseqs.a3m"),
            msa_output_dir=tmp_path,
        )
        for i, chain_letter in enumerate(["A", "C"]):
            unpaired_file = tmp_path / f"testcomplexseqs_unpaired_{chain_letter}.a3m"
            paired_file = tmp_path / f"testcomplexseqs_paired_{chain_letter}.a3m"
            assert unpaired_file.read_text() == convert_msas_to_str(unpairedmsas[i])
            assert paired_file.read_text() == convert_msas_to_str(pairedmsas[i])


class TestGuessCopies:
    """--guess-copies overrides the stoichiometry with the homo-oligomer count
    guessed from the first template's biological assembly."""

    def _single_chain_msas(self):
        query = Seq(name=">101\n", sequence="PIAQIHILEGRSDEQKE")
        return [[]], [[query]]

    def _valid_search_paths(self, tmp_path):
        """Create dummy but existing seqres/hmmbuild paths so the early
        template-search path validation passes (search itself is monkeypatched)."""
        seqres = tmp_path / "pdb_seqres.txt"
        seqres.write_text("")
        hmmbuild = tmp_path / "hmmbuild"
        hmmbuild.write_text("")
        return str(seqres), str(hmmbuild)

    def test_overrides_stoichiometry_to_guessed_count(self, monkeypatch, tmp_path):
        pairedmsas, unpairedmsas = self._single_chain_msas()
        seqres, hmmbuild = self._valid_search_paths(tmp_path)

        def fake_search_with_hits(**kwargs):
            return [], [("1bjp", "A")]

        def fake_guess(store, pdb_id, chain_id):
            assert (pdb_id, chain_id) == ("1bjp", "A")
            return 6

        monkeypatch.setattr(
            msatojson, "search_templates_with_hits", fake_search_with_hits
        )
        monkeypatch.setattr(msatojson, "guess_homomer_count_from_store", fake_guess)

        content = generate_input_json_content(
            name="1bjp",
            cardinality=1,
            stoichiometries=[1],  # header says monomer; guessing overrides to 6.
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=True,
            guess_copies=True,
            pdb_database_path="testfiles/mmcif_files",
            seqres_database_path=seqres,
            hmmbuild_binary_path=hmmbuild,
        )
        assert content["sequences"][0]["protein"]["id"] == [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ]

    def test_multiple_chains_shift_alphabet_without_overlap(
        self, monkeypatch, tmp_path
    ):
        q1 = Seq(name=">101\n", sequence="AAAA")
        q2 = Seq(name=">102\n", sequence="CCCC")
        pairedmsas = [[], []]
        unpairedmsas = [[q1], [q2]]
        seqres, hmmbuild = self._valid_search_paths(tmp_path)

        def fake_search_with_hits(**kwargs):
            # First chain -> dimer (1abc), second chain -> trimer (2xyz).
            seq = kwargs["msa_a3m_string"]
            if seq.startswith(">101"):
                return [], [("1abc", "A")]
            return [], [("2xyz", "A")]

        def fake_guess(store, pdb_id, chain_id):
            return 2 if pdb_id == "1abc" else 3

        monkeypatch.setattr(
            msatojson, "search_templates_with_hits", fake_search_with_hits
        )
        monkeypatch.setattr(msatojson, "guess_homomer_count_from_store", fake_guess)

        content = generate_input_json_content(
            name="complex",
            cardinality=2,
            stoichiometries=[1, 1],
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=True,
            guess_copies=True,
            pdb_database_path="testfiles/mmcif_files",
            seqres_database_path=seqres,
            hmmbuild_binary_path=hmmbuild,
        )
        assert content["sequences"][0]["protein"]["id"] == ["A", "B"]
        assert content["sequences"][1]["protein"]["id"] == ["C", "D", "E"]

    def test_missing_seqres_path_errors_early(self, monkeypatch, tmp_path):
        # Template search is requested but seqres_database_path does not exist:
        # generate_input_json_content must fail fast before any search runs.
        pairedmsas, unpairedmsas = self._single_chain_msas()
        _, hmmbuild = self._valid_search_paths(tmp_path)

        def boom(**kwargs):  # would be called if validation did not fire first
            raise AssertionError("template search should not run")

        monkeypatch.setattr(msatojson, "search_templates_with_hits", boom)
        monkeypatch.setattr(msatojson, "search_templates", boom)

        with pytest.raises(FileNotFoundError, match="seqres_database_path"):
            generate_input_json_content(
                name="1bjp",
                cardinality=1,
                stoichiometries=[1],
                pairedmsas=pairedmsas,
                unpairedmsas=unpairedmsas,
                includetemplates=True,
                pdb_database_path="testfiles/mmcif_files",
                seqres_database_path=str(tmp_path / "does_not_exist.txt"),
                hmmbuild_binary_path=hmmbuild,
            )

    def test_missing_pdb_database_dir_errors_early(self, monkeypatch, tmp_path):
        pairedmsas, unpairedmsas = self._single_chain_msas()
        seqres, hmmbuild = self._valid_search_paths(tmp_path)

        def boom(**kwargs):
            raise AssertionError("template search should not run")

        monkeypatch.setattr(msatojson, "search_templates_with_hits", boom)
        monkeypatch.setattr(msatojson, "search_templates", boom)

        with pytest.raises(NotADirectoryError, match="pdb_database_path"):
            generate_input_json_content(
                name="1bjp",
                cardinality=1,
                stoichiometries=[1],
                pairedmsas=pairedmsas,
                unpairedmsas=unpairedmsas,
                includetemplates=True,
                pdb_database_path=str(tmp_path / "no_such_dir"),
                seqres_database_path=seqres,
                hmmbuild_binary_path=hmmbuild,
            )

    def test_no_validation_when_templates_disabled(self, tmp_path):
        # Without includetemplates, missing template-search paths are irrelevant.
        pairedmsas, unpairedmsas = self._single_chain_msas()
        content = generate_input_json_content(
            name="1bjp",
            cardinality=1,
            stoichiometries=[1],
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=False,
            pdb_database_path=str(tmp_path / "no_such_dir"),
            seqres_database_path=str(tmp_path / "no_such_file"),
        )
        assert content["sequences"][0]["protein"]["id"] == ["A"]

    def test_no_template_hits_falls_back_to_one(self, monkeypatch, tmp_path):
        pairedmsas, unpairedmsas = self._single_chain_msas()
        seqres, hmmbuild = self._valid_search_paths(tmp_path)

        def fake_search_with_hits(**kwargs):
            return [], []  # no template hits

        monkeypatch.setattr(
            msatojson, "search_templates_with_hits", fake_search_with_hits
        )

        content = generate_input_json_content(
            name="1bjp",
            cardinality=1,
            stoichiometries=[1],
            pairedmsas=pairedmsas,
            unpairedmsas=unpairedmsas,
            includetemplates=True,
            guess_copies=True,
            pdb_database_path="testfiles/mmcif_files",
            seqres_database_path=seqres,
            hmmbuild_binary_path=hmmbuild,
        )
        assert content["sequences"][0]["protein"]["id"] == ["A"]
