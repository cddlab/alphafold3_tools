#!/usr/bin/env python3
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass

from loguru import logger

from alphafold3tools.log import log_setup


@dataclass
class Seq:
    name: str
    sequence: str


def get_info_from_json(
    jsonfile: str,
) -> tuple[list[int], list[int], list[str], list[str], list[str]]:
    """
    Extracts sequence information from a JSON file.

    This function opens the specified AlphaFold3 JSON file and parses its content
    to extract the following information for each sequence:
      - Sequence lengths (calculated from the protein sequence)
      - Stoichiometries (determined by the number of IDs in the protein data)
      - Query protein sequences
      - Unpaired MSA data
      - Paired MSA data

    Args:
        jsonfile (str): The file path to the JSON file containing sequence data.

    Returns:
        tuple[list[int], list[int], list[str], list[str], list[str]]:
            This tuple containing:
              - A list of sequence lengths (int).
              - A list of stoichiometries (int).
              - A list of query protein sequences (str).
              - A list of unpaired MSA information (str).
              - A list of paired MSA information (str).
    """
    logger.debug(f"Reading JSON file: {jsonfile}")
    with open(jsonfile, "r") as f:
        data = json.load(f)
        query_seqs = []
        stoichiometries = []
        unpairedmsas = []
        pairedmsas = []
        for i in range(len(data["sequences"])):
            if "protein" not in data["sequences"][i]:
                logger.debug(f"Skipping entity {i}: {data['sequences'][i]}")
                continue
            else:
                logger.debug(f"Found protein data for entity {i}.")
                stoichiometries.append(len(data["sequences"][i]["protein"]["id"]))
                query_seqs.append(data["sequences"][i]["protein"]["sequence"])
                unpairedmsas.append(data["sequences"][i]["protein"]["unpairedMsa"])
                pairedmsas.append(data["sequences"][i]["protein"]["pairedMsa"])
        seq_lens = [len(seq) for seq in query_seqs]
    return seq_lens, stoichiometries, query_seqs, unpairedmsas, pairedmsas


def write_header(seq_lens: list[int], stoichiometries: list[int]) -> str:
    """
    Writes the header information to the output MSA file.

    This function writes the header information to the output MSA file
    based on the sequence lengths and stoichiometries provided.

    If there is only one stoichiometry, the header will be empty.
    Otherwise, the header will contain the sequence lengths and stoichiometriesss
    Args:
        seq_lens (list[int]): A list of sequence lengths.
        stoichiometries (list[int]): A list of stoichiometries.
    Returns:
        str: The header information
    Examples:
        >>> write_header([139], [1])
        ""
        >>> write_header([62], [6])
        #62\t6\n
        >>> write_header([139, 126], [1, 1])
        #139,126\t1,1\n
    """
    if len(stoichiometries) == 1 and stoichiometries[0] == 1:
        logger.debug("Skipping header because there is only a monomer.")
        header = ""
    else:
        header = (
            f"#{','.join(map(str, seq_lens))}\t{','.join(map(str, stoichiometries))}\n"
        )
    return header


def write_pairedmsasection(query_seqs: list[str], pairedmsas: list[str]) -> str:
    """
    Processes and concatenates paired MSA data.

    This function takes a list of MSA strings where each string contains MSA data.
    For each MSA string, it splits the content into lines and creates a list of Seq
    objects by treating lines starting with ">" as headers and the following lines
    as part of the sequence. It then performs a sanity check to ensure that all MSAs
    have the same number of sequence entries. Finally, it concatenates the corresponding
    sequences (i.e., sequences with the same header across different MSAs) into a single
    Seq for each header, and returns a formatted string containing
    the concatenated results.

    Args:
        query_seqs (list[str]): A list of query sequences.
        pairedmsas (list[str]): A list of strings, each representing MSA data.
            Each MSA string should contain header lines starting with ">" followed
            by the sequence lines.

    Returns:
        str: A formatted string where each concatenated sequence is represented by
             its header followed by the concatenated sequence, separated by newlines.

    E.g.:
        query_seqs = [
            "AKQPT",
            "GCKS"
        ]
        pairedmsas = [
            ">seq1\nAKQPT\n>seq2\nAK-PT\n>seq3\nTGCakKS",
            ">seq4\nGCKS\n>seq5\nGCRS"
        ]
        The output will be:
        ">101\t102\nAKQPTGCKS\n>seq1\tseq4\nAKQPTGCKS\n>seq2\tseq5\nAK-PTGCRS\n>seq3\nTGCakKS----\n"
    """
    logger.debug(f"Processing paired MSA section with {len(pairedmsas)} MSAs.")
    # header_str is like ">101\t102"
    header_str = ">" + "\t".join(map(str, [101 + i for i in range(len(query_seqs))]))
    query_seqs_str = "".join(query_seqs)
    query_seqs_len = [len(seq) for seq in query_seqs]
    msachunks = [[] for _ in range(len(pairedmsas))]
    for i in range(len(pairedmsas)):
        for line in pairedmsas[i].split("\n"):
            if line == "":
                continue
            elif line.startswith(">"):
                msachunks[i].append(Seq(name=line, sequence=""))
            else:
                msachunks[i][-1].sequence = line
        logger.debug(f"Number of MSA sequences in {i}-th query: {len(msachunks[i])}")

    # all MSA chunks must have the same number of sequences
    # if not, add padding to the shorter ones
    if not all(len(msachunks[0]) == len(msachunk) for msachunk in msachunks):
        max_msa_chunks_len = max(len(msachunk) for msachunk in msachunks)
        for i in range(len(msachunks)):
            while len(msachunks[i]) < max_msa_chunks_len:
                msachunks[i].append(Seq(name="dummy", sequence="-" * query_seqs_len[i]))

    # Concatenate sequences for each entry
    concat_msas = []
    for j in range(len(msachunks[0])):
        names = [msachunks[i][j].name for i in range(len(msachunks))]
        sequences = [msachunks[i][j].sequence for i in range(len(msachunks))]
        name_str = ">" + "\t".join([n.lstrip(">") for n in names])
        sequence_str = "".join(sequences)
        concat_msas.append(Seq(name=name_str, sequence=sequence_str))
    # Build output string
    output = f"{header_str}\n{query_seqs_str}\n"
    for seq in concat_msas:
        output += f"{seq.name}\n{seq.sequence}\n"
    return output


def write_unpairedmsasection(seq_lens: list[int], unpairedmsas: list[str]) -> str:
    """write unpaired MSA section
    Args:
        seq_lens (list[int])
        unpairedmsas (list[str])
    Returns:
        concat_seq_str (str)
    """
    seqs = [[] for _ in range(len(unpairedmsas))]
    for i in range(len(unpairedmsas)):
        for line in unpairedmsas[i].split("\n"):
            if line == "":
                continue
            elif line.startswith(">"):
                seqs[i].append(Seq(name=line, sequence=""))
            else:
                seqs[i][-1].sequence = "".join(
                    [
                        "-" * sum(seq_lens[:i]),
                        line,
                        "-" * sum(seq_lens[i + 1 :]),
                    ]
                )

    concat_seq_str = "".join(
        f"{item.name}\n{item.sequence}\n" for seq in seqs for item in seq
    )
    return concat_seq_str


def write_a3m_file(
    outfile: str,
    seq_lens: list[int],
    query_seqs: list[str],
    stoichiometries: list[int],
    unpairedmsas: list[str],
    pairedmsas: list[str],
    output_paired: bool = True,
):
    """
    Writes the MSA data to an output file."
    """
    with open(outfile, "w") as f:
        f.write(write_header(seq_lens, stoichiometries))
        logger.debug(f"State: output_paired={output_paired}")
        if output_paired:
            f.write(write_pairedmsasection(query_seqs, pairedmsas))
        f.write(write_unpairedmsasection(seq_lens, unpairedmsas))


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Convert AlphaFold3 JSON to a3m-formatted MSA format.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input JSON file containing MSA data.",
        type=str,
        required=True,
    )
    parser.add_argument("-o", "--out", help="Output a3m file.", type=str, required=True)
    parser.add_argument(
        "-p",
        "--nopaired",
        help="Do not output paired MSA section.",
        action="store_false",
        dest="output_paired",
        default=True,
    )
    parser.add_argument(
        "-d",
        "--debug",
        help="Print lots of debugging statements",
        dest="loglevel",
        action="store_const",
        const="DEBUG",
        default="SUCCESS",
    )
    args = parser.parse_args()
    log_setup(args.loglevel)
    seq_lens, stoichiometries, query_seqs, unpairedmsas, pairedmsas = (
        get_info_from_json(args.input)
    )
    write_a3m_file(
        args.out,
        seq_lens,
        query_seqs,
        stoichiometries,
        unpairedmsas,
        pairedmsas,
        args.output_paired,
    )
    logger.debug(f"Successfully wrote MSA data to {args.out}")


if __name__ == "__main__":
    main()
