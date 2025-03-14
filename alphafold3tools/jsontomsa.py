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
      - (Query) Protein sequences
      - Unpaired MSA data
      - Paired MSA data

    Args:
        jsonfile (str): The file path to the JSON file containing sequence data.

    Returns:
        tuple[list[int], list[int], list[str], list[str], list[str]]:
            This tuple containing:
              - A list of sequence lengths (int).
              - A list of stoichiometries (int).
              - A list of protein sequences (str).
              - A list of unpaired MSA information (str).
              - A list of paired MSA information (str).
    """
    logger.debug(f"Reading JSON file: {jsonfile}")
    with open(jsonfile, "r") as f:
        data = json.load(f)
        protein_seqs = []
        stoichiometries = []
        unpairedmsas = []
        pairedmsas = []
        for i in range(len(data["sequences"])):
            if "protein" not in data["sequences"][i]:
                logger.debug(f"Skipping entity {i}: {data['sequences'][i]}")
                continue
            else:
                logger.debug(f"Found protein data for entity {i}.")
                protein_seqs.append(data["sequences"][i]["protein"]["sequence"])
                stoichiometries.append(len(data["sequences"][i]["protein"]["id"]))
                unpairedmsas.append(data["sequences"][i]["protein"]["unpairedMsa"])
                pairedmsas.append(data["sequences"][i]["protein"]["pairedMsa"])
        seq_lens = [len(seq) for seq in protein_seqs]
    return seq_lens, stoichiometries, protein_seqs, unpairedmsas, pairedmsas


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


def write_pairedmsasection(pairedmsas: list[str]) -> str:
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
        pairedmsas (list[str]): A list of strings, each representing MSA data.
            Each MSA string should contain header lines starting with ">" followed
            by the sequence lines.

    Returns:
        str: A formatted string where each concatenated sequence is represented by
             its header followed by the concatenated sequence, separated by newlines.

    TODO: sanity check for uppercase characters with seq_lens?
    """
    seqs = [[] for _ in range(len(pairedmsas))]

    for i in range(len(pairedmsas)):
        for line in pairedmsas[i].split("\n"):
            if line == "":
                continue
            elif line.startswith(">"):
                seqs[i].append(Seq(name=line, sequence=""))
            else:
                seqs[i][-1].sequence = line
    # sanity check
    if not all(len(seq) == len(seqs[0]) for seq in seqs):
        raise AssertionError("The number of sequences in each MSA does not match.")
    concat_seqs = []
    for j in range(len(seqs[0])):
        name = seqs[0][j].name
        concat_seq = "".join([msa[j].sequence for msa in seqs])
        concat_seqs.append(Seq(name, concat_seq))

    concat_seq_str = "".join(
        [
            f"{concat_seqs[_].name}\n{concat_seqs[_].sequence}\n"
            for _ in range(len(concat_seqs))
        ]
    )
    return concat_seq_str


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
    stoichiometries: list[int],
    unpairedmsas: list[str],
    pairedmsas: list[str],
):
    """
    Writes the MSA data to an output file."
    """
    with open(outfile, "w") as f:
        f.write(write_header(seq_lens, stoichiometries))
        f.write(write_pairedmsasection(pairedmsas))
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
    seq_lens, stoichiometries, protein_seqs, unpairedmsas, pairedmsas = (
        get_info_from_json(args.input)
    )
    write_a3m_file(args.out, seq_lens, stoichiometries, unpairedmsas, pairedmsas)
    logger.debug(f"Successfully wrote MSA data to {args.out}")


if __name__ == "__main__":
    main()
