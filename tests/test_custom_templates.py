import gzip
import os
import re
import shutil
from pathlib import Path

import gemmi
import pytest

import alphafold3tools.custom_templates as custom_templates
from alphafold3tools.custom_templates import (
    PLACEHOLDER_RELEASE_DATE,
    CustomTemplateError,
    find_custom_template_files,
    prepare_custom_template_database,
)
from alphafold3tools.searchtemplates import _parse_hit_description, _parse_hit_metadata_with_gemmi, search_templates
from alphafold3tools.structure.mmcif_parser import mmcifcontent
from alphafold3tools.structure_stores import StructureStore
from alphafold3tools.utils import resolve_max_subsequence_ratio

TESTFILES = Path(__file__).parent.parent / "testfiles"
PREDICTED_PDB = TESTFILES / "pdbtocif" / "P9WIC5_v6.pdb"  # AlphaFold2 model without a PDB ID.
AF3_CIF = TESTFILES / "ipsae" / "af3" / "tlxi_tlxi_model.cif"  # AlphaFold3 output.
PDB_ENTRY_CIF = TESTFILES / "mmcif_files" / "1BJP.cif"
LIGAND_CIF = TESTFILES / "orobanchol.cif"  # A CCD ligand definition; no polymer.

needs_hmmer = pytest.mark.skipif(
    shutil.which("hmmbuild") is None or shutil.which("hmmsearch") is None,
    reason="hmmbuild or hmmsearch not found in PATH",
)


def _copy(src: Path, dest: Path, *, compress: bool = False) -> Path:
    data = src.read_bytes()
    dest.write_bytes(gzip.compress(data) if compress else data)
    return dest


def _touch(directory: Path, *names: str) -> None:
    for name in names:
        (directory / name).write_text("")


def _protein_chain_ids(cif_path: Path) -> list[str]:
    block = gemmi.cif.read(str(cif_path)).sole_block()
    chain_ids = []
    for row in block.find("_entity_poly.", ["type", "pdbx_strand_id"]):
        if row.str(0) == "polypeptide(L)":
            chain_ids.extend(row.str(1).split(","))
    return chain_ids


def _query_sequence_of(cif_path: Path) -> str:
    block = gemmi.cif.read(str(cif_path)).sole_block()
    return gemmi.cif.as_string(block.find_value("_entity_poly.pdbx_seq_one_letter_code_can")).replace("\n", "")


class TestFindCustomTemplateFiles:
    def test_finds_plain_and_gzip_files_case_insensitively(self, tmp_path):
        _touch(tmp_path, "a.pdb", "b.CIF", "c.pdb.gz", "d.CIF.GZ", "e.txt", "f.cif.bz2", "g.ent", ".cif")
        (tmp_path / "sub").mkdir()
        _touch(tmp_path / "sub", "h.cif")

        found = find_custom_template_files(tmp_path)

        assert [path.name for path in found] == ["a.pdb", "b.CIF", "c.pdb.gz", "d.CIF.GZ"]

    def test_prefers_mmcif_over_pdb_and_plain_over_gzip(self, tmp_path, warnings_log):
        _touch(tmp_path, "x.cif", "x.cif.gz", "x.pdb", "x.pdb.gz", "y.cif.gz", "y.pdb", "z.pdb", "z.pdb.gz")

        found = find_custom_template_files(tmp_path)

        assert [path.name for path in found] == ["x.cif", "y.cif.gz", "z.pdb"]
        ignored = ["x.cif.gz", "x.pdb", "x.pdb.gz", "y.pdb", "z.pdb.gz"]
        assert len(warnings_log) == len(ignored)
        for name in ignored:
            assert any(name in message for message in warnings_log)

    def test_names_differing_only_in_case_are_the_same_structure(self, tmp_path):
        _touch(tmp_path, "Model.pdb", "model.cif")

        assert [path.name for path in find_custom_template_files(tmp_path)] == ["model.cif"]

    def test_names_differing_only_in_unicode_normalization_are_the_same_structure(self, tmp_path):
        nfd_name, nfc_name = "éx", "éx"  # Both are "éx"; macOS treats them as one name.
        _touch(tmp_path, f"{nfd_name}.cif", f"{nfc_name}.pdb")

        assert [path.name for path in find_custom_template_files(tmp_path)] == [f"{nfd_name}.cif"]

    def test_warns_when_pdb_is_newer_than_mmcif(self, tmp_path, warnings_log):
        _touch(tmp_path, "x.cif", "x.pdb")
        os.utime(tmp_path / "x.cif", (1_000_000, 1_000_000))

        find_custom_template_files(tmp_path)

        assert len(warnings_log) == 1
        assert "x.pdb is newer; delete x.cif" in warnings_log[0]


@pytest.mark.parametrize(
    "names, expected",
    [
        (["P9WIC5_v6"], ["p9wic5v6"]),
        (["AAAAAA"], ["aaaaaa"]),
        (["ab"], ["ab00"]),
        (["___"], ["0000"]),
        (["my-model", "my_model", "MyModel"], ["mymodel", "mymodel1", "mymodel2"]),
    ],
)
def test_template_ids_are_valid_and_unique(names, expected):
    ids = custom_templates._template_ids(names)

    assert ids == expected
    assert all(re.fullmatch(r"[a-z0-9]{4,}", template_id) for template_id in ids)


class TestPdbConversion:
    def test_pdb_is_converted_next_to_it(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(PREDICTED_PDB, custom_dir / "P9WIC5_v6.pdb")

        database = prepare_custom_template_database(custom_dir, tmp_path / "work")

        assert (custom_dir / "P9WIC5_v6.cif").is_file()
        # pdbtocif must not write its pdb_seqres.txt into the user's directory.
        assert not (custom_dir / "pdb_seqres.txt").exists()
        assert database.sources == {"p9wic5v6": custom_dir / "P9WIC5_v6.pdb"}

    def test_gzip_pdb_is_converted_to_plain_mmcif(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(PREDICTED_PDB, custom_dir / "P9WIC5_v6.pdb.gz", compress=True)

        prepare_custom_template_database(custom_dir, tmp_path / "work")

        converted = custom_dir / "P9WIC5_v6.cif"
        assert converted.read_text().startswith("data_")
        assert _protein_chain_ids(converted) == ["A"]

    def test_pdb_is_not_converted_when_mmcif_exists(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(PREDICTED_PDB, custom_dir / "1BJP.pdb")
        _copy(PDB_ENTRY_CIF, custom_dir / "1BJP.cif.gz", compress=True)

        def fail(*args, **kwargs):
            raise AssertionError("pdb_to_cif must not be called when the mmCIF file exists")

        monkeypatch.setattr(custom_templates, "pdb_to_cif", fail)

        database = prepare_custom_template_database(custom_dir, tmp_path / "work")

        assert database.sources == {"1bjp": custom_dir / "1BJP.cif.gz"}
        assert not (custom_dir / "1BJP.cif").exists()

    def test_failed_conversion_leaves_no_partial_file(self, tmp_path, monkeypatch, warnings_log):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(PREDICTED_PDB, custom_dir / "broken.pdb")
        _copy(PDB_ENTRY_CIF, custom_dir / "1BJP.cif")

        def fail_midway(input_path, output_path, **kwargs):
            Path(output_path).write_text("data_partial\n")
            raise RuntimeError("conversion failed")

        monkeypatch.setattr(custom_templates, "pdb_to_cif", fail_midway)

        database = prepare_custom_template_database(custom_dir, tmp_path / "work")

        assert list(database.sources) == ["1bjp"]
        assert sorted(path.name for path in custom_dir.iterdir()) == ["1BJP.cif", "broken.pdb"]
        assert any("broken.pdb" in message for message in warnings_log)

    def test_pdb_failing_the_checks_is_not_saved_as_mmcif(self, tmp_path, monkeypatch):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(PREDICTED_PDB, custom_dir / "P9WIC5_v6.pdb")

        def reject(store, template_id):
            raise ValueError("rejected")

        monkeypatch.setattr(custom_templates, "_check_template", reject)

        with pytest.raises(CustomTemplateError):
            prepare_custom_template_database(custom_dir, tmp_path / "work")
        # Otherwise the saved X.cif would be used instead of X.pdb from then on.
        assert sorted(path.name for path in custom_dir.iterdir()) == ["P9WIC5_v6.pdb"]

    @pytest.mark.skipif(os.name != "posix", reason="needs symbolic links")
    def test_existing_entry_at_converted_path_is_never_replaced(self, tmp_path, warnings_log):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        for name in ("dangling", "directory"):
            _copy(PREDICTED_PDB, custom_dir / f"{name}.pdb")
        (custom_dir / "dangling.cif").symlink_to(elsewhere / "target.cif")
        (custom_dir / "directory.cif").mkdir()

        database = prepare_custom_template_database(custom_dir, tmp_path / "work")

        # The converted structures are still used as templates, but nothing is written over.
        assert sorted(database.sources) == ["dangling", "directory"]
        assert list(elsewhere.iterdir()) == []
        assert (custom_dir / "dangling.cif").is_symlink()
        assert (custom_dir / "directory.cif").is_dir()
        assert sum("already exists" in message for message in warnings_log) == 2
        assert not any(path.name.endswith(".part") for path in custom_dir.iterdir())

    @pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root POSIX user")
    def test_read_only_directory_with_pdb_raises(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(PREDICTED_PDB, custom_dir / "P9WIC5_v6.pdb")
        custom_dir.chmod(0o555)
        try:
            with pytest.raises(CustomTemplateError, match="not writable"):
                prepare_custom_template_database(custom_dir, tmp_path / "work")
        finally:
            custom_dir.chmod(0o755)


class TestPrepareCustomTemplateDatabase:
    @pytest.fixture
    def database(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(PREDICTED_PDB, custom_dir / "P9WIC5_v6.pdb.gz", compress=True)
        _copy(AF3_CIF, custom_dir / "tlxi_tlxi_model.cif.gz", compress=True)
        _copy(PDB_ENTRY_CIF, custom_dir / "1BJP.cif")
        return prepare_custom_template_database(custom_dir, tmp_path / "work")

    def test_sources_are_listed_by_template_id(self, database):
        assert {template_id: path.name for template_id, path in database.sources.items()} == {
            "1bjp": "1BJP.cif",
            "p9wic5v6": "P9WIC5_v6.pdb.gz",
            "tlxitlximodel": "tlxi_tlxi_model.cif.gz",
        }
        for template_id in database.sources:
            assert (database.pdb_database_path / f"{template_id}.cif").is_file()

    def test_seqres_records_are_parsed_as_template_hits(self, database):
        lines = database.seqres_database_path.read_text().splitlines()
        headers = lines[::2]
        assert headers
        found = set()
        for header in headers:
            # hmmsearch reports a record ">ID mol:..." as "ID/start-end [subseq from] mol:...".
            name, description = header.removeprefix(">").split(" ", 1)
            pdb_id, chain_id, *_ = _parse_hit_description(f"{name}/1-10 [subseq from] {description}")
            assert f"{pdb_id}_{chain_id}" == name
            found.add((pdb_id, chain_id))
        assert found == {
            *(("1bjp", chain_id) for chain_id in "ABCDE"),
            ("p9wic5v6", "A"),
            ("tlxitlximodel", "A"),
            ("tlxitlximodel", "B"),
        }

    def test_every_protein_chain_can_be_extracted_as_a_template(self, database):
        store = StructureStore(database.pdb_database_path)
        for template_id in database.sources:
            for chain_id in _protein_chain_ids(database.pdb_database_path / f"{template_id}.cif"):
                release_date, sequence, unresolved = _parse_hit_metadata_with_gemmi(store, template_id, chain_id)
                assert sequence
                assert unresolved is not None
                block = mmcifcontent(template_id, chain_id, store)
                assert block.name == template_id

    def test_missing_fields_of_alphafold3_model_are_filled(self, database):
        block = gemmi.cif.read(str(database.pdb_database_path / "tlxitlximodel.cif")).sole_block()

        assert block.name == "tlxitlximodel"
        assert block.find_value("_entry.id") == "tlxitlximodel"
        assert list(block.find_values("_pdbx_audit_revision_history.revision_date")) == [PLACEHOLDER_RELEASE_DATE]
        assert block.find_value("_pdbx_database_status.recvd_initial_deposition_date") == PLACEHOLDER_RELEASE_DATE
        original = gemmi.cif.read(str(AF3_CIF)).sole_block()
        poly_seq = original.find("_entity_poly_seq.", ["entity_id", "mon_id"])
        expected = {
            entity_id: gemmi.one_letter_code([row.str(1) for row in poly_seq if row.str(0) == entity_id])
            for entity_id in ("1", "2")
        }
        filled = {
            row.str(0): row.str(1).replace("\n", "")
            for row in block.find("_entity_poly.", ["entity_id", "pdbx_seq_one_letter_code_can"])
        }
        assert filled == expected

    def test_release_date_of_pdb_entry_is_kept(self, database):
        block = gemmi.cif.read(str(database.pdb_database_path / "1bjp.cif")).sole_block()

        assert block.find_values("_pdbx_audit_revision_history.revision_date")[0] == "1998-12-02"

    def test_unusable_files_are_skipped(self, tmp_path, warnings_log):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "garbage.cif").write_text("this is not an mmCIF file\n")
        (custom_dir / "notgzip.cif.gz").write_bytes(b"not gzip data")
        _copy(LIGAND_CIF, custom_dir / "ligand.cif")
        _copy(PDB_ENTRY_CIF, custom_dir / "1BJP.cif")

        database = prepare_custom_template_database(custom_dir, tmp_path / "work")

        assert list(database.sources) == ["1bjp"]
        assert sorted(path.name for path in database.pdb_database_path.glob("*.cif")) == ["1bjp.cif"]
        for name in ("garbage.cif", "notgzip.cif.gz", "ligand.cif"):
            assert any(name in message for message in warnings_log)

    def test_unusual_metadata_gives_a_searchable_database(self, tmp_path, warnings_log):
        text = PDB_ENTRY_CIF.read_text()
        multiline_title = text.replace(
            "_struct_keywords.text            'TAUTOMERASE, ISOMERASE, MICROBIAL BIODEGRADATION'",
            "_struct_keywords.text\n;TAUTOMERASE,\nMICROBIAL BIODEGRADATION\n;",
        )
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        (custom_dir / "quoted.cif").write_text(
            multiline_title.replace("1 'Structure model' 1 0 1998-12-02", "1 'Structure model' 1 0 '1998-12-02'")
        )
        (custom_dir / "notadate.cif").write_text(
            text.replace("1 'Structure model' 1 0 1998-12-02", "1 'Structure model' 1 0 02/12/1998")
        )
        (custom_dir / "badchain.cif").write_text(
            text.replace(
                "_entity_poly.pdbx_strand_id                 A,B,C,D,E", "_entity_poly.pdbx_strand_id A-1,B,C,D,E"
            )
        )

        database = prepare_custom_template_database(custom_dir, tmp_path / "work")

        assert sorted(database.sources) == ["notadate", "quoted"]
        lines = database.seqres_database_path.read_text().splitlines()
        assert len(lines) == 2 * 10  # One header line and one sequence line per chain.
        for header in lines[::2]:
            name, description = header.removeprefix(">").split(" ", 1)
            _parse_hit_description(f"{name}/1-10 [subseq from] {description}")
        store = StructureStore(database.pdb_database_path)
        release_dates = {
            template_id: _parse_hit_metadata_with_gemmi(store, template_id, "A")[0] for template_id in database.sources
        }
        assert release_dates == {"quoted": "1998-12-02", "notadate": PLACEHOLDER_RELEASE_DATE}
        assert any("02/12/1998" in message for message in warnings_log)
        assert any("badchain.cif" in message and "A-1" in message for message in warnings_log)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            prepare_custom_template_database(tmp_path / "missing", tmp_path / "work")

    def test_directory_without_structure_files_raises(self, tmp_path):
        _touch(tmp_path, "notes.txt")

        with pytest.raises(CustomTemplateError, match="No PDB or mmCIF files"):
            prepare_custom_template_database(tmp_path, tmp_path / "work")

    def test_directory_without_usable_structure_raises(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(LIGAND_CIF, custom_dir / "ligand.cif")

        with pytest.raises(CustomTemplateError, match="None of the structure files"):
            prepare_custom_template_database(custom_dir, tmp_path / "work")


@needs_hmmer
class TestTemplateSearchWithCustomTemplates:
    def test_identical_structure_is_used_only_without_subsequence_filter(self, tmp_path):
        custom_dir = tmp_path / "custom"
        custom_dir.mkdir()
        _copy(PREDICTED_PDB, custom_dir / "P9WIC5_v6.pdb.gz", compress=True)
        database = prepare_custom_template_database(custom_dir, tmp_path / "work")
        query = _query_sequence_of(database.pdb_database_path / "p9wic5v6.cif")

        def search(max_subsequence_ratio):
            return search_templates(
                msa_a3m_string=f">query\n{query}\n",
                pdb_database_path=database.pdb_database_path,
                seqres_database_path=database.seqres_database_path,
                max_subsequence_ratio=max_subsequence_ratio,
            )

        templates = search(None)
        filtered = search(0.95)

        assert len(templates) == 1
        assert templates[0]["mmcif"].startswith("data_p9wic5v6")
        assert templates[0]["queryIndices"] == list(range(len(query)))
        assert filtered == []


class TestResolveMaxSubsequenceRatio:
    @pytest.mark.parametrize(
        "given, custom_template, expected",
        [
            (None, False, 0.95),
            (None, True, None),  # 1.0 disables the subsequence-ratio filter.
            (0.95, True, 0.95),
            (0.8, True, 0.8),
            (1.0, False, None),
        ],
    )
    def test_resolves_value(self, given, custom_template, expected):
        assert resolve_max_subsequence_ratio(given, custom_template=custom_template) == expected

    def test_warns_only_when_custom_template_changes_the_default(self, warnings_log):
        resolve_max_subsequence_ratio(0.95, custom_template=True)
        resolve_max_subsequence_ratio(None, custom_template=False)
        assert warnings_log == []

        resolve_max_subsequence_ratio(None, custom_template=True)
        assert len(warnings_log) == 1
        assert "--custom_template" in warnings_log[0]
        assert "1.0" in warnings_log[0] and "0.95" in warnings_log[0]
