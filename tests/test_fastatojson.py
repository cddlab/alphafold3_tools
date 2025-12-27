from alphafold3tools.fastatojson import (
    _write_protein_seq_section,
    convert_fasta_to_json,
)


def test_write_protein_seq_section_single():
    seq = "MKTAYIAKQRQISFVKSHFSR"
    expected = [{"protein": {"id": ["A"], "sequence": "MKTAYIAKQRQISFVKSHFSR"}}]
    result = _write_protein_seq_section(seq)
    assert result == expected


def test_write_protein_seq_section_multiple():
    seq = "MKTAYIAKQRQISFVKSHFSR:MKTA"
    expected = [
        {"protein": {"id": ["A"], "sequence": "MKTAYIAKQRQISFVKSHFSR"}},
        {"protein": {"id": ["B"], "sequence": "MKTA"}},
    ]
    result = _write_protein_seq_section(seq)
    assert result == expected
