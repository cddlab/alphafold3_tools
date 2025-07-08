#!/usr/bin/env python3
import json
from argparse import ArgumentParser, RawTextHelpFormatter

from Bio import SeqIO
from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.utils import int_id_to_str_id, sanitised_name


def _write_protein_seq_section(seq: str) -> list:
    """
    Write the protein sequence section in JSON format.

    This function formats protein sequence data into the JSON structure required by AlphaFold3.
    If the input sequence contains colon separators, it will be split into multiple sequences
    with incremental IDs.

    Args:
        seq (str): The protein sequence string. May contain multiple sequences separated by colons.

    Returns:
        list: A list of dictionaries, each containing a protein ID and its corresponding sequence.
    e.g.
        seq = "MKTAYIAKQRQISFVKSHFSR:MKTA"
        expected = (
            {"protein": {"id": ["A"], "sequence": "MKTAYIAKQRQISFVKSHFSR"}},
            {"protein": {"id": ["B"], "sequence": "MKTA"}},
        )
    """
    if ":" in seq:
        seqs = seq.split(":")
        return [
            {"protein": {"id": [int_id_to_str_id(i + 1)], "sequence": s}}
            for i, s in enumerate(seqs)
        ]
    else:
        return [{"protein": {"id": [int_id_to_str_id(1)], "sequence": seq}}]


def convert_fasta_to_json(inputfile: str, seeds: tuple[int] = (1,)) -> None:
    """
    Convert a FASTA file contatining multiple protein sequences to JSON format
    compatible with AlphaFold3.

    Args:
        inputfile (str): Path to the input FASTA file.
    """
    if seeds is None:
        seeds = [1]
    with open(inputfile, "r") as input_handle:
        for record in SeqIO.parse(input_handle, "fasta"):
            content = {
                "name": sanitised_name(record.id),
                "dialect": "alphafold3",
                "version": 1,
            }
            content["sequences"] = _write_protein_seq_section(str(record.seq))
            content["modelSeeds"] = list(seeds)
            with open(record.id + ".json", "w") as output_handle:
                json.dump(content, output_handle, indent=2)
                logger.info(f"Converted {record.id} to JSON format.")


def main():
    parser = ArgumentParser(
        description="Convert FASTA file to JSON format compatible with AlphaFold3.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="input FASTA file to convert to JSON file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seeds",
        help="model seeds to use for the conversion (default: 1)",
        type=int,
        nargs="+",
        default=(1,),
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
    inputfile = args.input
    seeds = args.seeds
    log_setup(level=args.loglevel)

    if not inputfile.endswith(".fasta") and not inputfile.endswith(".fa"):
        raise ValueError(
            "Input file must be a FASTA file with .fasta or .fa extension."
        )
    convert_fasta_to_json(inputfile, seeds=seeds)


if __name__ == "__main__":
    main()
