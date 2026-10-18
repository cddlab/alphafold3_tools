import gzip
import json
import shutil
import sys
from pathlib import Path

import gemmi
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
    validate_template_search_paths,
    write_input_json_file,
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


class TestTemplateDownloadMode:
    """When the database paths are omitted, msatojson downloads what it needs."""

    def _hmmbuild(self, tmp_path):
        hmmbuild = tmp_path / "hmmbuild"
        hmmbuild.write_text("")
        return str(hmmbuild)

    def test_validate_allows_omitted_database_paths(self, tmp_path):
        validate_template_search_paths(None, None, self._hmmbuild(tmp_path))

    def test_downloads_structures_into_templates_dir(self, monkeypatch, tmp_path):
        a3m = tmp_path / "in" / "foo.a3m"
        a3m.parent.mkdir()
        a3m.write_text(">101\nPIAQIHILEGRSDEQKE\n")
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        captured = {}

        def fake_search(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(msatojson, "search_templates", fake_search)
        monkeypatch.setattr(msatojson, "resolve_seqres_database_path", lambda path: "/cache/pdb_seqres.txt")

        write_input_json_file(
            inputmsafile=a3m,
            name="foo",
            outputjsonfile=out_dir / "foo.json",
            includetemplates=True,
            pdb_database_path=None,
            seqres_database_path=None,
            hmmbuild_binary_path=self._hmmbuild(tmp_path),
        )

        assert captured["pdb_database_path"] == out_dir / "templates" / "foo"
        assert captured["download_missing_structures"] is True
        assert captured["seqres_database_path"] == "/cache/pdb_seqres.txt"

    def test_local_database_is_used_as_is(self, monkeypatch, tmp_path):
        a3m = tmp_path / "foo.a3m"
        a3m.write_text(">101\nPIAQIHILEGRSDEQKE\n")
        seqres = tmp_path / "pdb_seqres.txt"
        seqres.write_text("")
        captured = {}

        def fake_search(**kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(msatojson, "search_templates", fake_search)

        write_input_json_file(
            inputmsafile=a3m,
            name="foo",
            outputjsonfile=tmp_path / "foo.json",
            includetemplates=True,
            pdb_database_path="testfiles/mmcif_files",
            seqres_database_path=str(seqres),
            hmmbuild_binary_path=self._hmmbuild(tmp_path),
        )

        assert captured["pdb_database_path"] == "testfiles/mmcif_files"
        assert captured["download_missing_structures"] is False
        assert captured["seqres_database_path"] == str(seqres)
        assert not (tmp_path / "templates").exists()

    def test_pdb_path_required_without_download(self, monkeypatch, tmp_path):
        def boom(**kwargs):
            raise AssertionError("template search should not run")

        monkeypatch.setattr(msatojson, "search_templates", boom)
        query = Seq(name=">101\n", sequence="PIAQIHILEGRSDEQKE")
        with pytest.raises(FileNotFoundError, match="pdb_database_path"):
            generate_input_json_content(
                name="foo",
                cardinality=1,
                stoichiometries=[1],
                pairedmsas=[[]],
                unpairedmsas=[[query]],
                includetemplates=True,
                pdb_database_path=None,
                seqres_database_path=None,
                hmmbuild_binary_path=self._hmmbuild(tmp_path),
            )


_HAS_HMMER = shutil.which("hmmbuild") is not None and shutil.which("hmmsearch") is not None


class TestCustomTemplateCli:
    """``--custom_template`` builds the template database from a directory of structures."""

    @pytest.fixture
    def custom_dir(self, tmp_path):
        directory = tmp_path / "custom"
        directory.mkdir()
        shutil.copy("testfiles/mmcif_files/1BJP.cif", directory / "1BJP.cif")
        return directory

    @pytest.fixture
    def searches(self, monkeypatch):
        """Replaces the template search and records the database each call sees."""
        calls: list[dict] = []

        def fake_search(**kwargs):
            database = Path(kwargs["pdb_database_path"])
            kwargs["database_files"] = sorted(path.name for path in database.iterdir())
            kwargs["seqres_text"] = Path(kwargs["seqres_database_path"]).read_text()
            calls.append(kwargs)
            return []

        monkeypatch.setattr(msatojson, "search_templates", fake_search)
        monkeypatch.setattr(msatojson, "search_templates_with_hits", lambda **kwargs: (fake_search(**kwargs), []))
        # main() calls log_setup(), which would remove the warnings_log handler.
        monkeypatch.setattr(msatojson, "log_setup", lambda level: None)
        return calls

    def _run(self, monkeypatch, tmp_path, *args, a3m_text=">101\nPIAQIHILEGRSDEQKE\n", hmmer=False):
        a3m = tmp_path / "foo.a3m"
        a3m.write_text(a3m_text)
        if hmmer:
            binaries = []
        else:
            hmmbuild = tmp_path / "hmmbuild"
            hmmbuild.write_text("")
            binaries = ["--hmmbuild_binary_path", str(hmmbuild), "--hmmsearch_binary_path", str(hmmbuild)]
        output = tmp_path / "foo.json"
        argv = ["msatojson", "-i", str(a3m), "-o", str(output), *binaries, *args]
        monkeypatch.setattr(sys, "argv", argv)
        msatojson.main()
        return output

    def test_custom_template_enables_template_search(self, monkeypatch, tmp_path, custom_dir, searches):
        output = self._run(monkeypatch, tmp_path, "--custom_template", str(custom_dir))

        assert len(searches) == 1
        call = searches[0]
        assert call["database_files"] == ["1bjp.cif", "pdb_seqres.txt"]
        assert call["seqres_text"].startswith(">1bjp_A mol:protein length:62")
        assert call["download_missing_structures"] is False
        assert call["max_subsequence_ratio"] is None
        assert json.loads(output.read_text())["sequences"][0]["protein"]["templates"] == []
        # The generated database is a temporary directory.
        assert not Path(call["pdb_database_path"]).exists()

    def test_default_ratio_change_is_warned(self, monkeypatch, tmp_path, custom_dir, searches, warnings_log):
        self._run(monkeypatch, tmp_path, "--custom_template", str(custom_dir))

        assert any("--custom_template is specified" in message for message in warnings_log)

    def test_explicit_ratio_is_kept(self, monkeypatch, tmp_path, custom_dir, searches, warnings_log):
        self._run(monkeypatch, tmp_path, "--custom_template", str(custom_dir), "--max_subsequence_ratio", "0.95")

        assert searches[0]["max_subsequence_ratio"] == 0.95
        assert not any("--custom_template is specified" in message for message in warnings_log)

    def test_default_ratio_without_custom_template(self, monkeypatch, tmp_path, searches):
        seqres = tmp_path / "pdb_seqres.txt"
        seqres.write_text("")
        self._run(
            monkeypatch,
            tmp_path,
            "--include_templates",
            "--pdb_database_path",
            "testfiles/mmcif_files",
            "--seqres_database_path",
            str(seqres),
        )

        assert searches[0]["max_subsequence_ratio"] == 0.95

    @pytest.mark.parametrize("option", ["--pdb_database_path", "--seqres_database_path"])
    def test_database_paths_are_rejected(self, monkeypatch, tmp_path, custom_dir, searches, option):
        with pytest.raises(SystemExit) as excinfo:
            self._run(monkeypatch, tmp_path, "--custom_template", str(custom_dir), option, str(tmp_path))

        assert excinfo.value.code == 2
        assert searches == []

    def test_empty_custom_template_is_rejected(self, monkeypatch, tmp_path, searches):
        with pytest.raises(SystemExit) as excinfo:
            self._run(monkeypatch, tmp_path, "--custom_template", "")

        assert excinfo.value.code == 2

    def test_database_is_removed_when_the_search_fails(self, monkeypatch, tmp_path, custom_dir):
        database_paths = []

        def failing_search(**kwargs):
            database_paths.append(Path(kwargs["pdb_database_path"]))
            raise RuntimeError("search failed")

        monkeypatch.setattr(msatojson, "search_templates", failing_search)
        monkeypatch.setattr(msatojson, "log_setup", lambda level: None)

        with pytest.raises(RuntimeError, match="search failed"):
            self._run(monkeypatch, tmp_path, "--custom_template", str(custom_dir))

        assert len(database_paths) == 1
        assert not database_paths[0].exists()

    def test_guess_copies_is_allowed(self, monkeypatch, tmp_path, custom_dir, searches):
        self._run(monkeypatch, tmp_path, "--custom_template", str(custom_dir), "--guess_copies")

        assert len(searches) == 1

    def test_directory_without_structures_exits_with_error(self, monkeypatch, tmp_path, searches):
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(SystemExit) as excinfo:
            self._run(monkeypatch, tmp_path, "--custom_template", str(empty))

        assert excinfo.value.code == 1
        assert searches == []

    def test_database_is_shared_in_directory_mode(self, monkeypatch, tmp_path, custom_dir, searches):
        input_dir = tmp_path / "a3m"
        input_dir.mkdir()
        for name in ("foo", "bar"):
            (input_dir / f"{name}.a3m").write_text(">101\nPIAQIHILEGRSDEQKE\n")
        hmmbuild = tmp_path / "hmmbuild"
        hmmbuild.write_text("")
        argv = ["msatojson", "-i", str(input_dir), "-o", str(tmp_path / "out"), "--hmmbuild_binary_path"]
        monkeypatch.setattr(sys, "argv", [*argv, str(hmmbuild), "--custom_template", str(custom_dir)])

        msatojson.main()

        assert len(searches) == 2
        assert len({call["pdb_database_path"] for call in searches}) == 1
        assert all(call["database_files"] == ["1bjp.cif", "pdb_seqres.txt"] for call in searches)
        assert sorted(path.name for path in (tmp_path / "out").glob("*.json")) == ["bar.json", "foo.json"]

    @pytest.mark.skipif(not _HAS_HMMER, reason="hmmbuild or hmmsearch not found in PATH")
    def test_predicted_model_is_added_as_template(self, monkeypatch, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        pdb = Path("testfiles/pdbtocif/P9WIC5_v6.pdb")
        (custom_dir / "P9WIC5_v6.pdb.gz").write_bytes(gzip.compress(pdb.read_bytes()))
        structure = gemmi.read_structure(str(pdb))
        query = gemmi.one_letter_code([residue.name for residue in structure[0][0]])

        output = self._run(
            monkeypatch, tmp_path, "--custom_template", str(custom_dir), a3m_text=f">101\n{query}\n", hmmer=True
        )

        templates = json.loads(output.read_text())["sequences"][0]["protein"]["templates"]
        assert len(templates) == 1
        assert templates[0]["mmcif"].startswith("data_p9wic5v6")
        assert (custom_dir / "P9WIC5_v6.cif").is_file()
