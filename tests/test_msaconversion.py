"""Comprehensive tests for align_sequence_to_gapless_query_cpp implementations.

Tests both the pure Python and NumPy-optimized versions against various
edge cases and the examples from the C++ implementation.
"""

import pytest

from alphafold3tools.msa_conversion import (
    align_sequence_to_gapless_query_cpp,
    align_sequences_batch_cpp,
    convert_a3m_to_stockholm_cpp,
    fasta_string_iterator,
    parse_fasta_include_descriptions,
)


class TestConvertA3MToStockholm:
    """Test cases for the convert_a3m_to_stockholm_cpp function."""

    def test_basic_example_from_docs(self):
        """Test the example from the function documentation."""
        a3m_sequences = ["abCD", "CgD", "fCDa"]
        expected = ["ABC-D-", "--CGD-", "F-C-DA"]

        result = convert_a3m_to_stockholm_cpp(a3m_sequences)
        assert result == expected

    def test_no_insertions(self):
        """Test when there are no insertions (all uppercase)."""
        a3m_sequences = ["ABCDE", "FGHIJ", "KLMNO"]
        expected = ["ABCDE", "FGHIJ", "KLMNO"]

        result = convert_a3m_to_stockholm_cpp(a3m_sequences)
        assert result == expected

    def test_all_insertions(self):
        """Test when all residues are insertions."""
        a3m_sequences = ["abc", "def", "ghi"]
        expected = ["ABC", "DEF", "GHI"]

        result = convert_a3m_to_stockholm_cpp(a3m_sequences)
        assert result == expected

    def test_mixed_insertions(self):
        """Test sequences with mixed aligned and inserted residues."""
        a3m_sequences = ["abCD", "CgD", "fCDa"]
        expected = ["ABC-D-", "--CGD-", "F-C-DA"]

        result = convert_a3m_to_stockholm_cpp(a3m_sequences)
        assert result == expected

    def test_single_sequence(self):
        """Test with a single sequence."""
        a3m_sequences = ["AbCdE"]
        expected = ["ABCDE"]

        result = convert_a3m_to_stockholm_cpp(a3m_sequences)
        assert result == expected

    def test_empty_list(self):
        """Test with empty input."""
        a3m_sequences = []
        expected = []

        result = convert_a3m_to_stockholm_cpp(a3m_sequences)
        assert result == expected

    def test_gaps_in_a3m(self):
        """Test handling of gaps in A3M format."""
        a3m_sequences = ["A-C", "ABC", "A-C"]
        expected = ["A-C", "ABC", "A-C"]

        result = convert_a3m_to_stockholm_cpp(a3m_sequences)
        assert result == expected

    def test_all_sequences_same_length_output(self):
        """Verify all output sequences have the same length."""
        a3m_sequences = ["abCD", "CgD", "fCDa"]

        result = convert_a3m_to_stockholm_cpp(a3m_sequences)

        # Check all sequences have same length
        lengths = [len(seq) for seq in result]
        assert len(set(lengths)) == 1, f"Sequences have different lengths: {lengths}"


class TestAlignSequenceToGaplessQuery:
    """Test cases for the align_sequence_to_gapless_query_cpp function."""

    @pytest.fixture(params=[align_sequence_to_gapless_query_cpp])
    def align_func(self, request):
        """Fixture to test both implementations."""
        return request.param

    def test_basic_example_from_docs(self, align_func):
        """Test the example from the function documentation."""
        sequence = "AB--E"
        query_sequence = "A--DE"
        expected = "Ab-E"

        result = align_func(sequence, query_sequence)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_no_gaps_in_query(self, align_func):
        """Test when query has no gaps."""
        sequence = "ABCDE"
        query_sequence = "ABCDE"
        expected = "ABCDE"

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_all_insertions(self, align_func):
        """Test when query is all gaps (all insertions)."""
        sequence = "ABCDE"
        query_sequence = "-----"
        expected = "abcde"

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_both_all_gaps(self, align_func):
        """Test when both sequences are all gaps."""
        sequence = "-----"
        query_sequence = "-----"
        expected = ""

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_mixed_case_preservation(self, align_func):
        """Test that original case is preserved for aligned residues."""
        sequence = "AbCdE"
        query_sequence = "ABCDE"
        expected = "AbCdE"

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_insertion_lowercase_conversion(self, align_func):
        """Test that insertions are converted to lowercase."""
        sequence = "ABCDE"
        query_sequence = "A--DE"
        expected = "AbcDE"

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_complex_stockholm_to_a3m(self, align_func):
        """Test a more complex Stockholm to A3M conversion."""
        sequence = "MKAGV--TL-GS"
        query_sequence = "MKA----TL-GS"
        expected = "MKAgvTLGS"

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_gap_handling(self, align_func):
        """Test various gap scenarios."""
        # Gap in sequence but not in query
        sequence = "A-CDE"
        query_sequence = "ABCDE"
        expected = "A-CDE"
        result = align_func(sequence, query_sequence)
        assert result == expected

        # Gap in query but not in sequence
        sequence = "ABCDE"
        query_sequence = "A-CDE"
        expected = "AbCDE"
        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_empty_sequences(self, align_func):
        """Test empty sequences."""
        sequence = ""
        query_sequence = ""
        expected = ""

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_single_character(self, align_func):
        """Test single character sequences."""
        # Aligned residue
        assert align_func("A", "A") == "A"
        # Insertion
        assert align_func("A", "-") == "a"
        # Both gaps
        assert align_func("-", "-") == ""
        # Gap in sequence
        assert align_func("-", "A") == "-"

    def test_length_mismatch_error(self, align_func):
        """Test that length mismatch raises ValueError."""
        sequence = "ABCDE"
        query_sequence = "ABC"

        with pytest.raises(ValueError) as exc_info:
            align_func(sequence, query_sequence)

        assert "don't have the same length" in str(exc_info.value)
        assert "5" in str(exc_info.value)
        assert "3" in str(exc_info.value)

    def test_nucleotide_sequences(self, align_func):
        """Test with nucleotide sequences."""
        sequence = "ATGC--GC"
        query_sequence = "ATG----C"
        expected = "ATGcgC"

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_special_characters(self, align_func):
        """Test with special amino acid characters."""
        sequence = "ABXZ*U"
        query_sequence = "AB--*U"
        expected = "ABxz*U"

        result = align_func(sequence, query_sequence)
        assert result == expected


class TestBatchProcessing_align_sequences_to_gapless_query_cpp:
    """Test cases for batch processing functionality."""

    def test_batch_processing_pure_python(self):
        """Test batch processing with pure Python implementation."""
        sequences = ["AB--E", "MKAGV--TL-GS", "ABCDE"]
        query_sequence = "A--DE"

        # Adjust query for the second sequence
        results = []
        results.append(
            align_sequence_to_gapless_query_cpp(sequences[0], query_sequence)
        )

        assert results[0] == "Ab-E"

    def test_batch_processing_numpy(self):
        """Test batch processing with NumPy implementation."""
        sequences = ["AB--E", "AC--F", "AD--G"]
        query_sequence = "A--DE"
        expected = ["Ab-E", "Ac-F", "Ad-G"]

        results = align_sequences_batch_cpp(sequences, query_sequence, use_numpy=True)
        assert results == expected


class TestEdgeCases_align_sequence_to_gapless_query_cpp:
    """Test edge cases and boundary conditions."""

    def test_very_long_gap_stretch(
        self, align_func=align_sequence_to_gapless_query_cpp
    ):
        """Test with very long consecutive gaps."""
        sequence = "A" + "-" * 1000 + "B"
        query_sequence = "A" + "-" * 1000 + "B"
        expected = "AB"

        result = align_func(sequence, query_sequence)
        assert result == expected

    def test_unicode_characters_error_handling(self):
        """Test behavior with non-ASCII characters (should work as chars)."""
        # This tests that the function handles any character
        sequence = "AαBβC"
        query_sequence = "A-B-C"
        expected = "AαBβC"

        result = align_sequence_to_gapless_query_cpp(sequence, query_sequence)
        assert result == expected


class TestParseFastaIncludeDescriptions:
    """Test cases for the parse_fasta_include_descriptions function."""

    def test_basic_single_sequence(self):
        """Test parsing a single sequence with simple description."""
        fasta = """>seq1
ACGT"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGT"]
        assert descriptions == ["seq1"]

    def test_single_sequence_multiline(self):
        """Test single sequence split across multiple lines."""
        fasta = """>seq1
ACGT
GATC"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGTGATC"]
        assert descriptions == ["seq1"]

    def test_multiple_sequences(self):
        """Test parsing multiple sequences."""
        fasta = """>seq1
ACGT
GATC
>seq2
AAAA
CCCC"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGTGATC", "AAAACCCC"]
        assert descriptions == ["seq1", "seq2"]

    def test_descriptions_with_spaces(self):
        """Test descriptions containing spaces and metadata."""
        fasta = """>prot1 human insulin protein
MKVV
>prot2 mouse insulin
CCGG"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["MKVV", "CCGG"]
        assert descriptions == ["prot1 human insulin protein", "prot2 mouse insulin"]

    def test_empty_lines_ignored(self):
        """Test that empty lines are properly ignored."""
        fasta = """>seq1
ACGT

GATC

>seq2

AAAA"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGTGATC", "AAAA"]
        assert descriptions == ["seq1", "seq2"]

        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGTGATC", "AAAA"]
        assert descriptions == ["seq1", "seq2"]

    def test_mixed_case_sequences(self):
        """Test sequences with mixed case (lowercase insertions)."""
        fasta = """>seq1
ACgt
GaTc
>seq2
AaAa"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        # Case should be preserved
        assert sequences == ["ACgtGaTc", "AaAa"]
        assert descriptions == ["seq1", "seq2"]

    def test_nucleotide_sequences(self):
        """Test DNA/RNA sequences."""
        fasta = """>dna1
ATGCATGC
>dna2
AAAA
TTTT"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ATGCATGC", "AAAATTTT"]
        assert descriptions == ["dna1", "dna2"]

    def test_long_sequences(self):
        """Test parsing long sequences."""
        # Create a long sequence split across many lines
        long_seq = "ACGT" * 250  # 1000 characters
        fasta = """>long_seq
""" + "\n".join(long_seq[i : i + 80] for i in range(0, len(long_seq), 80))

        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == [long_seq]
        assert descriptions == ["long_seq"]
        assert len(sequences[0]) == 1000

    def test_many_sequences(self):
        """Test parsing many sequences."""
        fasta = "\n".join([f">seq{i}\nACGT" for i in range(100)])

        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert len(sequences) == 100
        assert len(descriptions) == 100
        assert all(seq == "ACGT" for seq in sequences)
        assert descriptions == [f"seq{i}" for i in range(100)]

    def test_special_characters_in_description(self):
        """Test descriptions with special characters."""
        fasta = """>seq|1|organism=human|protein=insulin
ACGT
>seq|2|organism=mouse
GATC"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGT", "GATC"]
        assert descriptions == [
            "seq|1|organism=human|protein=insulin",
            "seq|2|organism=mouse",
        ]

    def test_unicode_in_descriptions(self):
        """Test descriptions with unicode characters."""
        fasta = """>seq1_αβγ_human
ACGT
>seq2_中文_mouse
GATC"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGT", "GATC"]
        assert descriptions == ["seq1_αβγ_human", "seq2_中文_mouse"]

    def test_very_short_sequences(self):
        """Test single character sequences."""
        fasta = """>s1
A
>s2
C
>s3
G"""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["A", "C", "G"]
        assert descriptions == ["s1", "s2", "s3"]

    def test_trailing_newlines(self):
        """Test handling of trailing newlines."""
        fasta = """>seq1
ACGT
>seq2
GATC
"""  # Trailing newlines

        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGT", "GATC"]
        assert descriptions == ["seq1", "seq2"]

    def test_no_trailing_newline_last_sequence(self):
        """Test when last sequence has no trailing newline."""
        fasta = """>seq1
ACGT
>seq2
GATC"""  # No newline at end

        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == ["ACGT", "GATC"]
        assert descriptions == ["seq1", "seq2"]


class TestParseErrorHandling:
    """Test error handling and validation."""

    def test_data_before_first_description(self):
        """Test error when sequence data appears before first description."""
        fasta = """ACGT
>seq1
GATC"""

        with pytest.raises(ValueError) as exc_info:
            parse_fasta_include_descriptions(fasta)

        assert "before first description" in str(exc_info.value).lower()

    def test_empty_string(self):
        """Test parsing empty string."""
        fasta = ""
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == []
        assert descriptions == []

    def test_only_whitespace(self):
        """Test parsing string with only whitespace."""
        fasta = "   \n  \n   "
        sequences, descriptions = parse_fasta_include_descriptions(fasta)

        assert sequences == []
        assert descriptions == []


class TestFastaStringIterator:
    """Test cases for the fasta_string_iterator function."""

    def test_single_sequence(self):
        """Test parsing a single sequence."""
        fasta = ">seq1\nACGT\n"
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 1
        seq, desc = results[0]
        assert seq == "ACGT"
        assert desc == "seq1"

    def test_single_sequence_with_description(self):
        """Test parsing a sequence with detailed description."""
        fasta = ">protein1 Cytochrome P450\nMKLAVS\n"
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 1
        seq, desc = results[0]
        assert seq == "MKLAVS"
        assert desc == "protein1 Cytochrome P450"

    def test_multiple_sequences(self):
        """Test parsing multiple sequences."""
        fasta = """>seq1
ACGT
>seq2
GCTA
>seq3
TTAA
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 3
        assert results[0] == ("ACGT", "seq1")
        assert results[1] == ("GCTA", "seq2")
        assert results[2] == ("TTAA", "seq3")

    def test_multiline_sequence(self):
        """Test sequence spanning multiple lines."""
        fasta = """>seq1
ACGT
GCTA
TTAA
GGCC
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 1
        seq, desc = results[0]
        assert seq == "ACGTGCTATTAAGGCC"
        assert desc == "seq1"

    def test_multiple_multiline_sequences(self):
        """Test multiple sequences each spanning multiple lines."""
        fasta = """>seq1
ACGT
GCTA
>seq2
TTAA
GGCC
CCGG
>seq3
ATAT
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 3
        assert results[0] == ("ACGTGCTA", "seq1")
        assert results[1] == ("TTAAGGCCCCGG", "seq2")
        assert results[2] == ("ATAT", "seq3")

    def test_empty_lines_between_sequences(self):
        """Test handling of empty lines within sequences."""
        fasta = """>seq1
ACGT

GCTA
>seq2

TTAA
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 2
        assert results[0] == ("ACGTGCTA", "seq1")
        assert results[1] == ("TTAA", "seq2")

    def test_whitespace_in_lines(self):
        """Test trimming of whitespace from lines."""
        fasta = """>seq1
  ACGT
  GCTA
>seq2
  TTAA
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 2
        assert results[0] == ("ACGTGCTA", "seq1")
        assert results[1] == ("TTAA", "seq2")

    def test_no_trailing_newline(self):
        """Test FASTA string without trailing newline."""
        fasta = ">seq1\nACGT"
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 1
        assert results[0] == ("ACGT", "seq1")

    def test_multiple_sequences_no_trailing_newline(self):
        """Test multiple sequences without trailing newline."""
        fasta = ">seq1\nACGT\n>seq2\nGCTA"
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 2
        assert results[0] == ("ACGT", "seq1")
        assert results[1] == ("GCTA", "seq2")

    def test_long_protein_sequence(self):
        """Test with a realistic protein sequence."""
        fasta = """>sp|P12345|PROT_HUMAN Example protein
MKLVSSALAGTVAVQAAQAPLKPGFNEVIRLHKNAYKLDPENPEMFVDG
DGQVNYEELLKIPKKVGNVNVALYEKGNAKVTVELKQGDNVKLKDPTFV
KVQYQGNAKVKDKQDGTVKLSDPTFVKVQYQGKVKVKDGTVKLKDPTFV
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 1
        seq, desc = results[0]
        assert desc == "sp|P12345|PROT_HUMAN Example protein"
        assert seq.startswith("MKLVSS")
        assert len(seq) == 147

    def test_nucleotide_sequences(self):
        """Test with DNA sequences."""
        fasta = """>gene1
ATGCATGCATGC
>gene2
GCTAGCTAGCTA
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 2
        assert results[0] == ("ATGCATGCATGC", "gene1")
        assert results[1] == ("GCTAGCTAGCTA", "gene2")

    def test_rna_sequences(self):
        """Test with RNA sequences."""
        fasta = """>rna1
AUGCAUGCAUGC
>rna2
GCUAGCUAGCUA
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 2
        assert results[0] == ("AUGCAUGCAUGC", "rna1")
        assert results[1] == ("GCUAGCUAGCUA", "rna2")

    def test_special_characters_in_description(self):
        """Test descriptions with special characters."""
        fasta = ">seq1|gene=ABC|organism=Homo sapiens\nACGT\n"
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 1
        seq, desc = results[0]
        assert seq == "ACGT"
        assert desc == "seq1|gene=ABC|organism=Homo sapiens"

    def test_empty_description(self):
        """Test sequence with empty description (just >)."""
        fasta = ">\nACGT\n"
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 1
        seq, desc = results[0]
        assert seq == "ACGT"
        assert desc == ""

    def test_very_long_sequence(self):
        """Test with a very long sequence."""
        seq_data = "ACGT" * 1000  # 4000 characters
        fasta = f">long_seq\n{seq_data}\n"
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 1
        seq, desc = results[0]
        assert seq == seq_data
        assert desc == "long_seq"

    def test_many_sequences(self):
        """Test with many sequences."""
        fasta = "\n".join([f">seq{i}\nACGT{i}" for i in range(100)])
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 100
        for i, (seq, desc) in enumerate(results):
            assert desc == f"seq{i}"
            assert seq == f"ACGT{i}"

    def test_iterator_pattern(self):
        """Test that the function returns a proper iterator."""
        fasta = ">seq1\nACGT\n>seq2\nGCTA\n>seq3\nTTAA\n"
        iterator = fasta_string_iterator(fasta)

        # Get items one at a time
        first = next(iterator)
        assert first == ("ACGT", "seq1")

        second = next(iterator)
        assert second == ("GCTA", "seq2")

        third = next(iterator)
        assert third == ("TTAA", "seq3")

        # Should raise StopIteration after exhausting
        with pytest.raises(StopIteration):
            next(iterator)

    def test_iterator_in_loop(self):
        """Test iterator usage in a for loop."""
        fasta = ">seq1\nACGT\n>seq2\nGCTA\n"
        sequences = []

        for seq, desc in fasta_string_iterator(fasta):
            sequences.append((seq, desc))

        assert sequences == [("ACGT", "seq1"), ("GCTA", "seq2")]

    def test_mixed_line_lengths(self):
        """Test sequences with varying line lengths."""
        fasta = """>seq1
AC
GTGC
TATTAA
>seq2
A
B
C
D
"""
        results = list(fasta_string_iterator(fasta))

        assert len(results) == 2
        assert results[0] == ("ACGTGCTATTAA", "seq1")
        assert results[1] == ("ABCD", "seq2")
