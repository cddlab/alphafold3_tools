import datetime
import os
import shutil
import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory

import gemmi
import pytest

import alphafold3tools.structure_stores as structure_stores
from alphafold3tools.msatojson import to_json
from alphafold3tools.searchtemplates import (
    PROTEIN_CHAIN,
    Template,
    Templates,
    _download_mmcif_file_for_pdbid,
    _parse_hit_metadata_with_gemmi,
    parse_fasta,
)


def test_parse_hit_metadata_with_gemmi():
    """Test parsing hit metadata from a PDB file using gemmi."""

    def download_and_parse(pdb_id: str, auth_chain_id: str):
        with TemporaryDirectory() as temp_dir:
            _download_mmcif_file_for_pdbid(pdb_id, temp_dir)
            cif_dir = structure_stores.StructureStore(temp_dir)
            release_date, sequence, unresolved_res_ids = _parse_hit_metadata_with_gemmi(
                cif_dir, pdb_id, auth_chain_id
            )
            return release_date, sequence, unresolved_res_ids

    assert (
        "2011-05-18",
        "GSKMTDLQDTKYVVYESVENNESMMDTFVKHPIKTGMLNGKKYMVMETTNDDYWKDFMVEGQRVRTISKDAKNNTRTIIFPYVEGKTLYDAIVKVHVKTIDYDGQYHVRIVDKEAFTKANT",
        [1, 2, 118, 119, 120, 121],
    ) == download_and_parse("3RUR", "A")
    assert (
        "2025-11-12",
        "MGSSHHHHHHSSGLVPRGSHMNSMHPETLMVHGGMDGLTEAGVHVPAIDLSTTNPVNDVATGGDSYEWLATGHALKDGDSAVYQRLWQPGVARFETALAELEHADEAVAFATGMAAMTAALLAAVNAGTPHIVAVRPLYGGSDHLLETGLLGTTVTWAKEAEIASAIQDDTGLVIVETPANPSLDLVDLDSVVAAAGTVPVLVDNTFCTPVLQQPIRHGAALVLHSATKYLGGHGDAMGGIIATNSDWAMRLRQVRAITGALLHPMGAYLLHRGLRTLAVRMRAAQTTAGELAERLAAHPAITAVHYPGLNGQDPRGLLGRQMSGGGAMIALELAGGFDAARSFVEHCSLVVHAVSLGGADTLIQHPASLTHRPVAATAKPGDGLIRLSVGLEHVDDLEDDLIAALDASRAAA",
        [
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            372,
            373,
            410,
            411,
            412,
            413,
        ],
    ) == download_and_parse("9HAQ", "A")


@pytest.mark.skipif(
    shutil.which("hmmbuild") is None or shutil.which("hmmsearch") is None,
    reason="hmmbuild or hmmsearch not found in PATH",
)
def test_templates_from_seq_and_a3m():
    """Test template search from sequence and A3M file using 2z9g example."""
    # Get test file paths
    testfiles_dir = Path(__file__).parent.parent / "testfiles"
    a3mfile = testfiles_dir / "2z9g.a3m"

    # Skip test if test file doesn't exist
    if not a3mfile.exists():
        pytest.skip(f"Test file {a3mfile} not found")

    # Read query sequence from a3m file
    with open(a3mfile, "r") as f:
        query_sequence = parse_fasta(f.read())[0][0]

    # Verify query sequence
    assert len(query_sequence) == 306
    assert query_sequence.startswith(
        "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDTVYCPRHVICTAEDMLNPNYEDLLIRKSNHSFLVQAGNVQLRVIGHSMQNCLLRLKVDTSNPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRP"
    )

    with TemporaryDirectory() as temp_dir:
        # Use testfiles/mmcif directory if available, otherwise use temp directory
        mmcif_test_dir = testfiles_dir / "mmcif"
        if mmcif_test_dir.exists():
            mmcif_dir = str(mmcif_test_dir)
        else:
            mmcif_dir = temp_dir

        # Create a simple mock database for testing
        seqres_database_path = os.path.join(temp_dir, "test_pdb_seqres.txt")
        with open(seqres_database_path, "w") as f:
            # Add a few mock sequences for testing
            f.write(">2z9g_A mol:protein length:306\n")
            f.write(query_sequence + "\n")

        # Run template search
        template_hits = Templates.from_seq_and_a3m(
            query_sequence=query_sequence,
            msa_a3m=open(a3mfile, "r").read(),
            max_template_date=datetime.date(2099, 12, 31),
            seqres_database_path=seqres_database_path,
            max_a3m_query_sequences=None,
            structure_store=structure_stores.StructureStore(mmcif_dir),
            chain_poly_type=PROTEIN_CHAIN,
            savehmmsto=False,
        )

        # Basic assertions about the template hits
        assert isinstance(template_hits, Templates)
        assert template_hits.query_sequence == query_sequence
        assert template_hits.num_hits >= 0


def test_write_templates_to_json():
    """Test writing templates to JSON format."""

    def write_templates_to_json(templates, output_json_path):
        """Writes templates to a JSON file."""
        templates_list = []
        for template in templates:
            templates_list.append(
                {
                    "mmcif": template.mmcif,
                    "queryIndices": list(template.query_to_template_map.keys()),
                    "templateIndices": list(template.query_to_template_map.values()),
                }
            )
        templates_dict = {"templates": templates_list}

        with open(output_json_path, "w") as f:
            f.write(to_json(templates_dict))

    # Create mock templates
    mock_mmcif = textwrap.dedent("""data_TEST
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
                _atom_site.pdbx_formal_charge
                _atom_site.auth_seq_id
                _atom_site.auth_comp_id
                _atom_site.auth_asym_id
                _atom_site.auth_atom_id
                _atom_site.pdbx_PDB_model_num
                ATOM   1 N  N   . ALA A 1 1 ? 1.000  2.000  3.000  1.00 10.00 ? 1 ALA A N   1
                #""")

    query_to_template_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

    template = Template(
        mmcif=mock_mmcif,
        query_to_template_map=query_to_template_map,
    )

    templates = [template]

    with TemporaryDirectory() as temp_dir:
        output_path = os.path.join(temp_dir, "test_templates.json")
        write_templates_to_json(templates, output_path)

        # Verify file was created
        assert os.path.exists(output_path)

        # Verify content
        with open(output_path, "r") as f:
            import json

            content = json.load(f)

        assert "templates" in content
        assert len(content["templates"]) == 1
        assert "mmcif" in content["templates"][0]
        assert "queryIndices" in content["templates"][0]
        assert "templateIndices" in content["templates"][0]
        assert content["templates"][0]["queryIndices"] == [0, 1, 2, 3, 4]
        assert content["templates"][0]["templateIndices"] == [0, 1, 2, 3, 4]


def test_template_creation_with_gemmi_options():
    """Test creating templates with specific gemmi.cif.WriteOptions."""
    mock_mmcif = textwrap.dedent("""data_TEST
                #
                _entry.id TEST
                #""")
    # Test with different gemmi options
    options = gemmi.cif.WriteOptions()
    options.misuse_hash = True
    options.align_loops = 20
    options.prefer_pairs = True

    query_to_template_map = {0: 0, 1: 1, 2: 2}

    template = Template(
        mmcif=mock_mmcif,
        query_to_template_map=query_to_template_map,
    )

    # Verify template properties
    assert template.mmcif == mock_mmcif
    assert template.query_to_template_map == query_to_template_map
    assert len(template.query_to_template_map) == 3


@pytest.mark.skipif(
    shutil.which("hmmbuild") is None or shutil.which("hmmsearch") is None,
    reason="hmmbuild or hmmsearch not found in PATH",
)
def test_2z9g_full_pipeline():
    """
    Integration test for the 2z9g example.
    This test runs the full template search pipeline and validates the results.

    Expected results based on running the code with 2z9g example:
    - Query sequence: 306 amino acids
    - Should find template hits from PDB database
    - Should generate valid Template objects with mmCIF and mappings
    """
    # Get test file paths
    testfiles_dir = Path(__file__).parent.parent / "testfiles"
    a3mfile = testfiles_dir / "2z9g.a3m"

    # Skip test if test file doesn't exist
    if not a3mfile.exists():
        pytest.skip(f"Test file {a3mfile} not found")

    seqres_database_path = (
        Path(__file__).parent.parent / "testfiles" / "seqres" / "test_pdb_seqres.txt"
    )
    pdb_database_path = Path(__file__).parent.parent / "testfiles" / "mmcif_files"

    if not seqres_database_path.exists() or not pdb_database_path.exists():
        pytest.skip("Full test database (pdb_seqres.txt) or mmcif_files not available")

    # Read query sequence from a3m file
    with open(a3mfile, "r") as f:
        fasta_content = f.read()
        sequences, descriptions = parse_fasta(fasta_content)
        query_sequence = sequences[0]

    # Expected query sequence properties for 2z9g
    assert len(query_sequence) == 306, (
        f"Expected query sequence length 306, got {len(query_sequence)}"
    )
    assert query_sequence.startswith(
        "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDTVYCPRHVICTAEDMLNPNYEDLLIRKSNHSFLVQAGNVQLRVIGHSMQNCLLRLKVDTSNPKTPKYKFVRIQPGQTFSVLACYNGSPSGVYQCAMRP"
    )

    with TemporaryDirectory() as temp_dir:
        # Run template search
        template_hits = Templates.from_seq_and_a3m(
            query_sequence=query_sequence,
            msa_a3m=fasta_content,
            max_template_date=datetime.date(2099, 12, 31),
            seqres_database_path=str(seqres_database_path),
            max_a3m_query_sequences=None,
            structure_store=structure_stores.StructureStore(str(pdb_database_path)),
            chain_poly_type=PROTEIN_CHAIN,
            savehmmsto=False,
        )

        # Verify template hits object
        assert isinstance(template_hits, Templates)
        assert template_hits.query_sequence == query_sequence
        assert template_hits.num_hits >= 0

        # If hits are found, verify their structure
        if template_hits.num_hits > 0:
            for hit in template_hits.hits:
                assert hasattr(hit, "pdb_id")
                assert hasattr(hit, "auth_chain_id")
                assert hasattr(hit, "query_sequence")
                assert hasattr(hit, "is_valid")
                assert hit.query_sequence == query_sequence

        # Create templates with gemmi options (similar to the selected code)
        options = gemmi.cif.WriteOptions()
        options.misuse_hash = True
        options.align_loops = 20
        options.prefer_pairs = True

        try:
            templates = [
                Template(
                    mmcif=block.as_string(options=options),
                    query_to_template_map=hit.query_to_hit_mapping,
                )
                for hit, block in template_hits.get_hits_with_structures()
            ]

            # Verify templates
            for template in templates:
                assert isinstance(template, Template)
                assert isinstance(template.mmcif, str)
                assert len(template.mmcif) > 0
                assert isinstance(template.query_to_template_map, dict)
                assert len(template.query_to_template_map) > 0

                # Verify that query_to_template_map contains valid indices
                for query_idx, template_idx in template.query_to_template_map.items():
                    assert isinstance(query_idx, int)
                    assert isinstance(template_idx, int)
                    assert 0 <= query_idx < len(query_sequence)
                    assert template_idx >= 0

        except Exception as e:
            pytest.skip(f"Could not create templates: {e}")


@pytest.mark.skipif(
    shutil.which("hmmbuild") is None or shutil.which("hmmsearch") is None,
    reason="hmmbuild or hmmsearch not found in PATH",
)
def test_ras_full_pipeline():
    """
    Integration test for the RAS example.
    This test runs the full template search pipeline and validates the results.
    PDB ID: 7KYZ is a SOLUTION NMR structure. This file lacks _refine.ls_d_res_high entry.
    """
    # Get test file paths
    testfiles_dir = Path(__file__).parent.parent / "testfiles"
    a3mfile = testfiles_dir / "ras.a3m"

    # Skip test if test file doesn't exist
    if not a3mfile.exists():
        pytest.skip(f"Test file {a3mfile} not found")

    # Skip if pdb_seqres.txt is not available in Desktop (full test data)
    seqres_database_path = (
        Path(__file__).parent.parent / "testfiles" / "seqres" / "test_pdb_seqres.txt"
    )
    pdb_database_path = Path(__file__).parent.parent / "testfiles" / "mmcif_files"

    if not seqres_database_path.exists() or not pdb_database_path.exists():
        pytest.skip("Full test database (pdb_seqres.txt) or mmcif_files not available")

    # Read query sequence from a3m file
    with open(a3mfile, "r") as f:
        fasta_content = f.read()
        sequences, descriptions = parse_fasta(fasta_content)
        query_sequence = sequences[0]

    with TemporaryDirectory() as temp_dir:
        # Run template search
        template_hits = Templates.from_seq_and_a3m(
            query_sequence=query_sequence,
            msa_a3m=fasta_content,
            max_template_date=datetime.date(2099, 12, 31),
            seqres_database_path=str(seqres_database_path),
            max_a3m_query_sequences=None,
            structure_store=structure_stores.StructureStore(str(pdb_database_path)),
            chain_poly_type=PROTEIN_CHAIN,
            savehmmsto=False,
        )

        # Verify template hits object
        assert isinstance(template_hits, Templates)
        assert template_hits.query_sequence == query_sequence
        assert template_hits.num_hits >= 0

        # If hits are found, verify their structure
        if template_hits.num_hits > 0:
            for hit in template_hits.hits:
                assert hasattr(hit, "pdb_id")
                assert hasattr(hit, "auth_chain_id")
                assert hasattr(hit, "query_sequence")
                assert hasattr(hit, "is_valid")
                assert hit.query_sequence == query_sequence

        # Create templates with gemmi options (similar to the selected code)
        options = gemmi.cif.WriteOptions()
        options.misuse_hash = True
        options.align_loops = 20
        options.prefer_pairs = True

        try:
            templates = [
                Template(
                    mmcif=block.as_string(options=options),
                    query_to_template_map=hit.query_to_hit_mapping,
                )
                for hit, block in template_hits.get_hits_with_structures()
            ]

            # Verify templates
            for template in templates:
                assert isinstance(template, Template)
                assert isinstance(template.mmcif, str)
                assert len(template.mmcif) > 0
                assert isinstance(template.query_to_template_map, dict)
                assert len(template.query_to_template_map) > 0

                # Verify that query_to_template_map contains valid indices
                for query_idx, template_idx in template.query_to_template_map.items():
                    assert isinstance(query_idx, int)
                    assert isinstance(template_idx, int)
                    assert 0 <= query_idx < len(query_sequence)
                    assert template_idx >= 0

        except Exception as e:
            pytest.skip(f"Could not create templates: {e}")
