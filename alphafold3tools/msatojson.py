import concurrent.futures
import datetime
import json
import os
import re
import shutil
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.searchtemplates import search_templates
from alphafold3tools.utils import add_version_option, int_id_to_str_id


@dataclass
class Seq:
    name: str
    sequence: str


def get_residuelens_stoichiometries(lines: list[str]) -> tuple[list[int], list[int]]:
    """Get residue lengths and stoichiometries from msa file.
    Args:
        lines: list[str]
            Lines of input msa file
    Returns:
        residue_lens: list[int]
            Residue lengths of each polypeptide chain
        stoichiometries: list[int]
            Stoichiomerties of each polypeptide chain
    """
    if lines[0].startswith("#"):
        residue_lens_, stoichiometries_ = lines[0].split("\t")
        residue_lens = list(map(int, residue_lens_.lstrip("#").split(",")))
        stoichiometries = list(map(int, stoichiometries_.split(",")))
    else:
        # If the first line does not start with '#',
        # get the residue length from the first sequence.
        # Always assume a monomer prediction.
        if not lines[0].startswith(">"):
            raise ValueError(
                "The first line of the input MSA file must start with '#' or '>'."
            )
        residue_lens = [len(lines[1].strip())]
        stoichiometries = [1]
    return residue_lens, stoichiometries


def split_a3msequences(residue_lens: list[int], line: str) -> list[str]:
    """Split a3m sequences into a list of a3m sequences.
    Note: The a3m-format MSA file represents inserted residues with lowercase.
    The first line (starting with '#') of the MSA file contains residue lengths
    and stoichiometries of each polypeptide chain.
    From the second line, the first sequence is the query.
    After this, the paired MSA blocks are followed by the unpaired MSA.
    Args:
        residue_lens: list[int]
            Residue lengths of each polypeptide chain
        line: str
            A3M sequences
    Returns:
        a3msequences: list[str]
            A3M sequences, len(a3msequences) should be the same as len(residue_lens).
    """
    a3msequences = [""] * len(residue_lens)
    i = 0
    count = 0
    current_residue = []

    for char in line:
        current_residue.append(char)
        if char == "-" or char.isupper():
            count += 1
        if count == residue_lens[i]:
            a3msequences[i] = "".join(current_residue)
            current_residue = []
            count = 0
            i += 1
            if i == len(residue_lens):
                break

    if current_residue and i < len(residue_lens):
        a3msequences[i] = "".join(current_residue)

    return a3msequences


def get_paired_and_unpaired_msa(
    lines: list[str], residue_lens: list[int], cardinality: int
) -> tuple[list[list[Seq]], list[list[Seq]]]:
    """Get paired and unpaired MSAs from input MSA file.
    Args:
        lines: list[str]
            Lines of input MSA file
        residue_lens: list[int]
            Residue lengths of each polypeptide chain
        cardinality: int
            Number of polypeptide chains
        query_seqnames: list[int]
            Query sequence names
    Returns:
        pairedmsas: list[list[Seq]]
            Paired MSAs, len(pairedmsa) should be the cardinality.
            If cardinality is 1, pairedmsas returns [[Seq("", "")]].
        unpairedmsas: list[list[Seq]]
            Unpaired MSAs, len(unpairedmsa) should be the cardinality.
    """
    pairedmsas: list[list[Seq]] = [[] for _ in range(cardinality)]
    unpairedmsas: list[list[Seq]] = [[] for _ in range(cardinality)]
    pairedflag = False
    unpairedflag = False
    seen = False
    seqnames_seen = []
    query_seqnames = [int(101 + i) for i in range(cardinality)]
    chain = -1
    start = 1 if lines[0].startswith("#") else 0
    for line in lines[start:]:
        if line.startswith(">"):
            if line not in seqnames_seen:
                seqnames_seen.append(line)
            else:
                seen = True
                continue
            if cardinality > 1 and line.startswith(
                ">" + "\t".join(map(str, query_seqnames)) + "\n"
            ):
                pairedflag = True
                unpairedflag = False
            elif any(line.startswith(f">{seq}\n") for seq in query_seqnames):
                pairedflag = False
                unpairedflag = True
                chain += 1
            seqname = line
        else:
            if seen:
                seen = False
                continue
            if pairedflag:
                a3mseqs = split_a3msequences(residue_lens, line)
                for i in range(cardinality):
                    pairedmsas[i].append(Seq(seqname, a3mseqs[i]))

            elif unpairedflag:
                a3mseqs = split_a3msequences(residue_lens, line)
                for i in range(cardinality):
                    # Remove all-gapped sequences
                    if a3mseqs[i] == "-" * residue_lens[i]:
                        continue
                    unpairedmsas[i].append(Seq(seqname, a3mseqs[i]))
            else:
                raise ValueError("Flag must be either paired or unpaired.")
    return pairedmsas, unpairedmsas


def convert_msas_to_str(msas: list[Seq]) -> str:
    """convert MSAs to str format for AlphaFold3 input JSON file."""
    if msas == []:
        return ""
    else:
        return "\n".join(f"{seq.name}{seq.sequence}" for seq in msas) + "\n"


def generate_input_json_content(
    name: str,
    cardinality: int,
    stoichiometries: list[int],
    pairedmsas: list[list[Seq]],
    unpairedmsas: list[list[Seq]],
    includetemplates: bool = False,
    savehmmsto: bool = False,
    pdb_database_path: str | os.PathLike[str] | None = None,
    seqres_database_path: str | os.PathLike[str] | None = None,
    max_template_date: datetime.date = datetime.date(2099, 12, 31),
    max_subsequence_ratio: float | None = 0.95,
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
) -> dict[str, Any]:
    """generate AlphaFold3 input JSON file.

    Args:
        name (str): Name of the protein complex.
                    Used for the name field in the JSON file.
        cardinality (int): The number of distinct polypeptide chains.
        stoichiometries (list[int]): Stoichiometries of each polypeptide chain.
        pairedmsas (list[list[Seq]]): Paired MSAs.
        unpairedmsas (list[list[Seq]]): Unpaired MSAs.
        includetemplates (bool): Whether to include template search results.
        savehmmsto (bool): Whether to save intermediate HMM sto files.
        pdb_database_path (str): Path to the PDB mmCIF database for template search.
        max_template_date (datetime.date): Maximum template date for template search.
        max_subsequence_ratio (float | None): Maximum ratio of the length of a template subsequence to the length of the query sequence.
            If a template is an exact subsequence of the query sequence and its length ratio is above this threshold, it is excluded.
            This is to avoid ground truth leakage from templates which are almost the same as the query.
        hmmbuild_binary_path (str): Path to the hmmbuild binary.
        hmmsearch_binary_path (str): Path to the hmmsearch binary.
    Returns:
        str: JSON string for AlphaFold3 input file.
    """
    sequences = []
    chain_id_count = 0
    null = None
    for i in range(cardinality):
        # unpairedmsa[i][0] is more appropriate than pairedmsa[i][0].
        query_seq = unpairedmsas[i][0].sequence
        chain_ids = [
            int_id_to_str_id(chain_id_count + j + 1) for j in range(stoichiometries[i])
        ]
        chain_id_count += stoichiometries[i]
        if includetemplates:
            logger.info(
                f"Searching templates for chain {i + 1} with sequence length {len(query_seq)}..."
            )
            templates_list = search_templates(
                msa_a3m_string=convert_msas_to_str(unpairedmsas[i]),
                pdb_database_path=pdb_database_path,
                seqres_database_path=seqres_database_path,
                savehmmsto=savehmmsto,
                max_template_date=max_template_date,
                max_subsequence_ratio=max_subsequence_ratio,
                hmmbuild_binary_path=hmmbuild_binary_path,
                hmmsearch_binary_path=hmmsearch_binary_path,
            )
        else:
            templates_list = []
        sequences.append(
            {
                "protein": {
                    "id": chain_ids,
                    "sequence": query_seq,
                    "modifications": [],
                    "unpairedMsa": convert_msas_to_str(unpairedmsas[i]),
                    "pairedMsa": convert_msas_to_str(pairedmsas[i]),
                    "templates": templates_list,
                }
            }
        )
    content = {
        "dialect": "alphafold3",
        "version": 1,
        "name": f"{name}",
        "sequences": sequences,
        "modelSeeds": [1],
        "bondedAtomPairs": null,
        "userCCD": null,
    }
    return content


def write_input_json_file(
    inputmsafile: str | Path,
    name: str,
    outputjsonfile: str | Path,
    includetemplates: bool = True,
    savehmmsto: bool = False,
    pdb_database_path: str | os.PathLike[str] | None = None,
    seqres_database_path: str | os.PathLike[str] | None = None,
    max_template_date: datetime.date = datetime.date(2099, 12, 31),
    max_subsequence_ratio: float | None = 0.95,
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
) -> None:
    """Write AlphaFold3 input JSON file from a3m-format MSA file.

    Args:
        inputmsafile (str): Input MSA file path.
        name (str): Name of the protein complex.
                    Used for the name field in the JSON file.
        outputjsonfile (str): Output JSON file path.
        includetemplates (bool): Whether to include template search results.
        savehmmsto (bool): Whether to save intermediate HMM sto files.
        pdb_database_path (str): Path to the PDB mmCIF database for template search.
        max_template_date (datetime.date): Maximum template date for template search.
        max_subsequence_ratio (float | None): Maximum ratio of the length of a template subsequence to the length of the query sequence.
        hmmbuild_binary_path (str): Path to the hmmbuild binary.
        hmmsearch_binary_path (str): Path to the hmmsearch binary.
    """
    with open(inputmsafile, "r") as f:
        lines = f.readlines()
    residue_lens, stoichiometries = get_residuelens_stoichiometries(lines)
    if len(residue_lens) != len(stoichiometries):
        raise ValueError("Length of residue_lens and stoichiometries must be the same.")
    cardinality = len(residue_lens)
    logger.info(
        f"The input MSA file contains {cardinality} distinct polypeptide chains."
    )
    logger.info(f"Residue lengths: {residue_lens}")
    logger.info(f"Stoichiometries: {stoichiometries}")
    pairedmsas, unpairedmsas = get_paired_and_unpaired_msa(
        lines, residue_lens, cardinality
    )
    content = generate_input_json_content(
        name=f"{name}",
        cardinality=cardinality,
        stoichiometries=stoichiometries,
        pairedmsas=pairedmsas,
        unpairedmsas=unpairedmsas,
        includetemplates=includetemplates,
        savehmmsto=savehmmsto,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        max_template_date=max_template_date,
        max_subsequence_ratio=max_subsequence_ratio,
        hmmbuild_binary_path=hmmbuild_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
    )
    with open(outputjsonfile, "w") as f:
        f.write(to_json(content))


def _process_a3m_file(
    a3m_file: Path,
    output_dir: Path,
    includetemplates: bool,
    savehmmsto: bool,
    pdb_database_path: str | os.PathLike[str] | None,
    seqres_database_path: str | os.PathLike[str] | None,
    max_template_date: datetime.date,
    max_subsequence_ratio: float | None,
    hmmbuild_binary_path: str | None,
    hmmsearch_binary_path: str | None,
) -> None:
    """Process a single A3M file and write the output JSON file."""
    name = Path(a3m_file).stem
    output_file = os.path.join(output_dir, f"{name}.json")
    write_input_json_file(
        inputmsafile=a3m_file,
        name=name,
        outputjsonfile=output_file,
        includetemplates=includetemplates,
        savehmmsto=savehmmsto,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        max_template_date=max_template_date,
        max_subsequence_ratio=max_subsequence_ratio,
        hmmbuild_binary_path=hmmbuild_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
    )


def process_a3m_directory(
    input_dir: Path,
    output_dir: Path,
    includetemplates: bool,
    savehmmsto: bool,
    pdb_database_path: str | os.PathLike[str] | None,
    seqres_database_path: str | os.PathLike[str] | None,
    max_template_date: datetime.date,
    max_subsequence_ratio: float | None,
    hmmbuild_binary_path: str | None,
    hmmsearch_binary_path: str | None,
) -> None:
    """Process all A3M files in a directory.

    Args:
        input_dir: Input directory containing A3M files.
        output_dir: Output directory for JSON files.
    """
    if output_dir.suffix == ".json":
        raise ValueError(
            "Now the input is directory, so output name must be a directory."
        )
    logger.info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    a3m_files = list(input_dir.glob("*.a3m"))
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                _process_a3m_file,
                a3m_file,
                output_dir,
                includetemplates,
                savehmmsto,
                pdb_database_path,
                seqres_database_path,
                max_template_date,
                max_subsequence_ratio,
                hmmbuild_binary_path,
                hmmsearch_binary_path,
            )
            for a3m_file in a3m_files
        ]
        concurrent.futures.wait(futures)


def process_single_a3m_file(
    inputmsafile: Path,
    outputjsonfile: Path,
    includetemplates: bool = False,
    savehmmsto: bool = False,
    pdb_database_path: str | os.PathLike[str] | None = None,
    seqres_database_path: str | os.PathLike[str] | None = None,
    max_template_date: datetime.date = datetime.date(2099, 12, 31),
    max_subsequence_ratio: float | None = 0.95,
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
) -> None:
    """Process a single A3M file.

    Args:
        inputmsafile: Input A3M file path.
        outputjsonfile: Output JSON file path.
    """
    name = inputmsafile.stem
    if inputmsafile.suffix != ".a3m":
        raise ValueError("Input file must have .a3m extension.")
    logger.info(f"Input A3M file: {inputmsafile}")
    if outputjsonfile.suffix != ".json":
        raise ValueError("Output file must have .json extension.")
    logger.info(f"Output JSON file: {outputjsonfile}")
    write_input_json_file(
        inputmsafile=inputmsafile,
        name=name,
        outputjsonfile=outputjsonfile,
        includetemplates=includetemplates,
        savehmmsto=savehmmsto,
        pdb_database_path=pdb_database_path,
        seqres_database_path=seqres_database_path,
        max_template_date=max_template_date,
        max_subsequence_ratio=max_subsequence_ratio,
        hmmbuild_binary_path=hmmbuild_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
    )


def to_json(content: dict) -> str:
    """Converts Input to an AlphaFold JSON."""
    alphafold_json = json.dumps(content, indent=2)
    # Remove newlines from the query/template indices arrays. We match the
    # queryIndices/templatesIndices with a non-capturing group. We then match
    # the entire region between the square brackets by looking for lines
    # containing only whitespace, number, or a comma.
    return re.sub(
        r'("(?:queryIndices|templateIndices)": \[)([\s\n\d,]+)(\],?)',
        lambda mtch: mtch[1] + re.sub(r"\n\s+", " ", mtch[2].strip()) + mtch[3],
        alphafold_json,
    )


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Converts a3m-format MSA file to AlphaFold3 input JSON file.",
    )
    add_version_option(parser)
    parser.add_argument(
        "-i",
        "--input",
        help="Input A3M file or directory containing A3M files. e.g. 1bjp.a3m",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Name of the protein complex.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-o", "--out", help="Output directory or JSON file.", type=str, required=True
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
    # includetemplates
    parser.add_argument(
        "--include_templates",
        help="Include template search results in the output JSON file.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_hmmsto",
        help="Save intermediate HMM sto files used for template search.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--pdb_database_path",
        help="Path to the PDB mmCIF database for template search.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seqres_database_path",
        help="Path to the PDB SEQRES database for template search.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_template_date",
        help="Maximum template date for template search in YYYY-MM-DD format.",
        type=lambda s: datetime.date.fromisoformat(s),
        default=datetime.date(2099, 12, 31),
    )
    parser.add_argument(
        "--max_subsequence_ratio",
        help="Maximum subsequence ratio for template search. "
        "If set to 1.0, no templates will be excluded based on subsequence ratio. "
        "Default is 0.95.",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--hmmbuild_binary_path",
        help="Path to the hmmbuild binary. Default is to use the hmmbuild in PATH.",
        type=str,
        default=shutil.which("hmmbuild"),
    )
    parser.add_argument(
        "--hmmsearch_binary_path",
        help="Path to the hmmsearch binary. Default is to use the hmmsearch in PATH.",
        type=str,
        default=shutil.which("hmmsearch"),
    )
    args = parser.parse_args()
    log_setup(args.loglevel)
    # Default name is the input file name without extension
    if args.name == "":
        args.name = os.path.splitext(os.path.basename(args.input))[0]
    logger.info(f"Name of the protein complex: {args.name}")
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")
    out_path = Path(args.out)
    if args.max_subsequence_ratio == 1.0:
        logger.success(
            "No templates will be excluded based on subsequence ratio since max_subsequence_ratio is set to 1.0."
        )
        args.max_subsequence_ratio = None
    if input_path.is_dir():
        logger.info(f"Input directory: {input_path}")
        process_a3m_directory(
            input_dir=input_path,
            output_dir=out_path,
            includetemplates=args.include_templates,
            savehmmsto=args.save_hmmsto,
            pdb_database_path=args.pdb_database_path,
            seqres_database_path=args.seqres_database_path,
            max_template_date=args.max_template_date,
            max_subsequence_ratio=args.max_subsequence_ratio,
            hmmbuild_binary_path=args.hmmbuild_binary_path,
            hmmsearch_binary_path=args.hmmsearch_binary_path,
        )
    else:
        process_single_a3m_file(
            inputmsafile=input_path,
            outputjsonfile=out_path,
            includetemplates=args.include_templates,
            savehmmsto=args.save_hmmsto,
            pdb_database_path=args.pdb_database_path,
            seqres_database_path=args.seqres_database_path,
            max_template_date=args.max_template_date,
            max_subsequence_ratio=args.max_subsequence_ratio,
            hmmbuild_binary_path=args.hmmbuild_binary_path,
            hmmsearch_binary_path=args.hmmsearch_binary_path,
        )


if __name__ == "__main__":
    main()
