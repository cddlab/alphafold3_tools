from collections.abc import Iterable
from typing import Iterator, Tuple

import numpy as np


def fasta_string_iterator(fasta_string: str) -> Iterator[Tuple[str, str]]:
    """Parse a FASTA format string and yield (sequence, description) pairs.

    This function implements an iterator pattern similar to the C++
    FastaStringIterator class. It processes the FASTA string line by line,
    yielding each complete sequence with its description as soon as the next
    sequence header is encountered or the end of the string is reached.

    Args:
        fasta_string: A string in FASTA format. Each sequence should start with
            a description line beginning with '>', followed by one or more lines
            of sequence data.

    Yields:
        Tuples of (sequence, description) where:
        - sequence: The concatenated sequence data (str)
        - description: The description line without the '>' prefix (str)

    Raises:
        ValueError: If the FASTA string is invalid (doesn't start with '>').

    Examples:
        >>> fasta = ">seq1 description\\nACGT\\nGCTA\\n>seq2\\nTTAA\\n"
        >>> for seq, desc in fasta_string_iterator(fasta):
        ...     print(f"{desc}: {seq}")
        seq1 description: ACGTGCTA
        seq2: TTAA

        >>> # Single sequence
        >>> fasta = ">protein1\\nMKLAVS\\nALAGTV\\n"
        >>> list(fasta_string_iterator(fasta))
        [('MKLAVSALAGTV', 'protein1')]
    """
    description = None
    sequence = ""

    for line_raw in fasta_string.split("\n"):
        line = line_raw.strip()

        if line.startswith(">"):
            # Remove the '>' prefix
            new_description = line[1:]

            if description is None:
                # First description encountered
                description = new_description
            else:
                # Yield the previous sequence before starting a new one
                yield (sequence, description)
                description = new_description
                sequence = ""
        elif description is not None and line:
            # Accumulate sequence data (skip empty lines)
            sequence += line

    # Yield the last sequence
    if description is not None:
        yield (sequence, description)
    else:
        raise ValueError("Invalid FASTA string: no sequence header found")


def parse_fasta_include_descriptions(fasta_string: str) -> tuple[list[str], list[str]]:
    """Parse FASTA string and return sequences with their descriptions.

    Reads a FASTA format string and extracts both the amino acid/nucleotide sequences
    and the descriptions associated with each sequence.

    Algorithm (matching C++ implementation):
    1. Split input by newlines
    2. For each line:
        - Strip whitespace
        - If line starts with '>': this is a description
            - Store the description (everything after '>')
            - Initialize new sequence
        - Else if line is not empty and we have a description:
            - Append line to current sequence

    Args:
        fasta_string: FASTA formatted string with sequences and descriptions

    Returns:
        Tuple of (sequences_list, descriptions_list) where:
        - sequences_list: List of concatenated sequence strings (one per sequence)
        - descriptions_list: List of descriptions (one per sequence)

    Raises:
        ValueError: If FASTA string is malformed (e.g., sequence without description)

    Examples:
        >>> fasta = ">seq1\\nACGT\\nGATC\\n>seq2\\nAAAA"
        >>> seqs, descs = parse_fasta_include_descriptions(fasta)
        >>> seqs
        ['ACGTGATC', 'AAAA']
        >>> descs
        ['seq1', 'seq2']

        >>> fasta = ">prot1 human insulin\\nMK\\nVV\\n>prot2 mouse insulin\\nAA"
        >>> seqs, descs = parse_fasta_include_descriptions(fasta)
        >>> seqs
        ['MKVV', 'AA']
        >>> descs
        ['prot1 human insulin', 'prot2 mouse insulin']
    """
    sequences = []
    descriptions = []
    current_sequence = None

    # Split by newlines and process each line
    lines = fasta_string.split("\n")

    for line_raw in lines:
        # Strip whitespace (both leading and trailing)
        line = line_raw.strip()

        # If line starts with "#", it's a comment - skip it
        if line.startswith("#"):
            continue
        # Check if this is a description line (starts with '>')
        elif line.startswith(">"):
            # Remove the '>' prefix to get the description
            description = line[1:]
            descriptions.append(description)

            # Start a new sequence
            current_sequence = ""
            sequences.append(current_sequence)
        elif line and current_sequence is not None:
            # Non-empty line and we have an active sequence
            # Append to the current sequence
            # Use index to update (works because sequence is in list)
            sequences[-1] += line
        elif line and current_sequence is None:
            # Non-empty line but no description yet - invalid FASTA
            raise ValueError(
                f"Invalid FASTA format: found sequence data before first description. "
                f"Line: '{line}'"
            )

    return sequences, descriptions


def convert_a3m_to_stockholm_cpp(a3m_sequences: Iterable[str]) -> list[str]:
    """Converts a list of sequences in A3M format to Stockholm format.

    In A3M format, insertions (residues not aligned to the query) are
    represented as lowercase letters. In Stockholm format, all sequences
    must have the same length, with insertions padded using gaps ('-').

    Example:
        Input A3M:
            abCD
            CgD
            fCDa

        Output Stockholm:
            ABC-D-
            --CGD-
            F-C-DA

    Args:
        a3m_sequences: A list of strings in A3M format. Insertions are
            represented as lowercase letters, aligned residues as uppercase.

    Returns:
        A list of strings converted to Stockholm format, where all sequences
        have the same length and insertions are uppercase with gaps for padding.

    Raises:
        ValueError: If A3M rows have inconsistent lengths after processing
            (indicates invalid A3M input).
    """
    a3m_sequences = list(a3m_sequences)
    if not a3m_sequences:
        return []

    num_sequences = len(a3m_sequences)

    # Convert to list of lists for easier manipulation
    sequences = [list(seq) for seq in a3m_sequences]

    # Pre-allocate output arrays
    stockholm_sequences = [[] for _ in range(num_sequences)]

    # Track current position in each sequence
    positions = [0] * num_sequences

    # Process column by column
    while any(positions[i] < len(sequences[i]) for i in range(num_sequences)):
        # Check if any sequence has an insertion (lowercase) at current position
        has_insertion = False
        for i in range(num_sequences):
            if positions[i] < len(sequences[i]):
                char = sequences[i][positions[i]]
                if char.islower():
                    has_insertion = True
                    break

        if has_insertion:
            # Process insertion column
            for i in range(num_sequences):
                if positions[i] < len(sequences[i]):
                    char = sequences[i][positions[i]]
                    if char.islower():
                        # Convert insertion to uppercase
                        stockholm_sequences[i].append(char.upper())
                        positions[i] += 1
                    else:
                        # Pad with gap
                        stockholm_sequences[i].append("-")
                else:
                    # Sequence already exhausted, pad with gap
                    stockholm_sequences[i].append("-")
        else:
            # Process aligned column (no insertions)
            for i in range(num_sequences):
                if positions[i] < len(sequences[i]):
                    char = sequences[i][positions[i]]
                    stockholm_sequences[i].append(char)
                    positions[i] += 1
                else:
                    # This should not happen with valid A3M input
                    raise ValueError(
                        f"A3M rows have inconsistent lengths; row {i} has no "
                        f"columns left but not all rows are exhausted"
                    )

    # Convert lists back to strings
    return ["".join(seq) for seq in stockholm_sequences]


def convert_a3m_to_stockholm_batch_cpp(
    a3m_sequences_list: list[list[str]],
) -> list[list[str]]:
    """Batch process multiple A3M alignments.

    Args:
        a3m_sequences_list: List of A3M alignments, where each alignment
            is a list of sequences.

    Returns:
        List of Stockholm alignments.
    """
    return [convert_a3m_to_stockholm_cpp(a3m_seqs) for a3m_seqs in a3m_sequences_list]


def validate_a3m_format_cpp(a3m_sequences: list[str]) -> bool:
    """Validate that sequences are in proper A3M format.

    Args:
        a3m_sequences: List of A3M sequences to validate.

    Returns:
        True if format is valid.

    Raises:
        ValueError: If format is invalid with description of the issue.
    """
    if not a3m_sequences:
        return True

    # Check that all sequences contain only valid characters
    valid_chars = set("ABCDEFGHIKLMNPQRSTVWXYZabcdefghiklmnpqrstvwxyz-*")
    for i, seq in enumerate(a3m_sequences):
        invalid = set(seq) - valid_chars
        if invalid:
            raise ValueError(f"Sequence {i} contains invalid characters: {invalid}")

    return True


def align_sequence_to_gapless_query_cpp(sequence: str, query_sequence: str) -> str:
    """NumPy-optimized version of align_sequence_to_gapless_query.

    Uses NumPy arrays for faster processing of large sequences.

    Args:
        sequence: A string containing the sequence to be aligned.
        query_sequence: A string containing the reference sequence to align to.

    Returns:
        The aligned sequence string.

    Raises:
        ValueError: If sequence and query_sequence have different lengths.
    """
    if len(sequence) != len(query_sequence):
        raise ValueError(
            f"The sequence ({len(sequence)}) and the query sequence "
            f"({len(query_sequence)}) don't have the same length."
        )

    # Convert strings to numpy arrays for vectorized operations
    seq_arr = np.array(list(sequence), dtype="U1")
    query_arr = np.array(list(query_sequence), dtype="U1")

    # Create masks for different conditions
    query_not_gap = query_arr != "-"
    seq_is_gap = seq_arr == "-"
    insertion = (query_arr == "-") & ~seq_is_gap

    # Build output array
    # Start with positions where query has no gap
    output_mask = query_not_gap | insertion
    output_arr = seq_arr[output_mask].copy()

    # Convert insertions to lowercase
    insertion_in_output = insertion[output_mask]
    if insertion_in_output.any():
        # Use numpy's char operations for lowercase conversion
        output_arr[insertion_in_output] = np.char.lower(output_arr[insertion_in_output])

    return "".join(output_arr)


def align_sequences_batch_cpp(
    sequences: list[str], query_sequence: str, use_numpy: bool = True
) -> list[str]:
    """Batch process multiple sequences for alignment.

    Args:
        sequences: List of sequences to align.
        query_sequence: Reference query sequence.
        use_numpy: Whether to use NumPy-optimized version.

    Returns:
        List of aligned sequences.
    """
    return [
        align_sequence_to_gapless_query_cpp(seq, query_sequence) for seq in sequences
    ]
