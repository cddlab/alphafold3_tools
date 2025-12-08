from tempfile import TemporaryDirectory

import alphafold3tools.structure_stores as structure_stores
from alphafold3tools.searchtemplates import (
    _parse_hit_metadata_with_gemmi,
    download_ciffile,
)


def test_parse_hit_metadata_with_gemmi():
    """Test parsing hit metadata from a PDB file using gemmi."""

    def download_and_parse(pdb_id: str, auth_chain_id: str):
        with TemporaryDirectory() as temp_dir:
            download_ciffile(pdb_id, temp_dir)
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
