import textwrap
from dataclasses import dataclass

import pytest

from alphafold3tools.jsontomsa import (
    get_info_from_json,
    write_header,
    write_pairedmsasection,
    write_unpairedmsasection,
)


@dataclass
class Seq:
    name: str
    sequence: str


def test_get_info_from_json():
    sample_jsonfile = "testfiles/testheterocomplex.json"
    (seq_lens, stoichiometries, protein_seqs, unpairedmsas, pairedmsas) = (
        get_info_from_json(sample_jsonfile)
    )

    # Expected outputs based on sample data
    expected_seq_lens = [139, 126]
    expected_stoichiometries = [1, 1]
    assert seq_lens == expected_seq_lens
    assert stoichiometries == expected_stoichiometries
    assert protein_seqs[0].startswith("MTTFTVPFDPEKPDLTD")
    assert protein_seqs[1].startswith("MSQLTNEVEKHTDLYVE")
    assert unpairedmsas[0].startswith(">101\nMTTFTVPFDP")
    assert unpairedmsas[1].startswith(">102\nMSQLTNEVEKHTDLYVEAF")
    assert pairedmsas[0].startswith(">101\t102\nMTTFTVPFDPEKPDLT")
    assert pairedmsas[1].startswith(">101\t102\nMSQLTNEVEKHTDLYVEAFNR")


def test_writeheader():
    # Expected output based on sample data
    expected_output = "#139,126\t1,1\n"
    assert write_header([139, 126], [1, 1]) == expected_output


def test_write_pairedmsasection_valid():
    # Define two valid MSA strings with matching headers.
    msa1 = ">seq1\nAKQPT\n>seq2\nAK-PT\n>seq3\nTGCakKS"
    msa2 = ">seq4\nGCKS\n>seq5\nGCRS"
    query_seqs = ["AKQPT", "GCKS"]
    pairedmsas = [msa1, msa2]

    # Expected concatenated result:
    expected_output = ">101\t102\nAKQPTGCKS\n>seq1\tseq4\nAKQPTGCKS\n>seq2\tseq5\nAK-PTGCRS\n>seq3\tdummy\nTGCakKS----\n"

    result = write_pairedmsasection(query_seqs, pairedmsas)
    assert result == expected_output


def test_write_unpairedmsasection():
    # Define sample data
    seq_lens = [4, 5, 3]
    unpairedmsas = [
        ">101\nMTTF\n>test2\nAAAA",
        ">102\nAAAAA\n>test2\nTTaTTT",
        ">103\nTTC\n>test2\nGGG",
    ]

    # Expected concatenated result:
    # For 101: header ">101", sequence "MTTF--------"
    # For test2: header ">test2", sequence "AAAA--------"
    # For 102: header ">102", sequence "----AAAAA---"
    # For test2: header ">test2", sequence "----TTaTTT---"
    # For 103: header ">103", sequence "---------TTC"
    # For test2: header ">test2", sequence "---------GGG"
    expected_output = textwrap.dedent(
        """\
        >101
        MTTF--------
        >test2
        AAAA--------
        >102
        ----AAAAA---
        >test2
        ----TTaTTT---
        >103
        ---------TTC
        >test2
        ---------GGG
    """
    )
    result = write_unpairedmsasection(seq_lens, unpairedmsas)
    assert result == expected_output
