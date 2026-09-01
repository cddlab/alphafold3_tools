#!/usr/bin/env python3
import copy
import datetime
import json
import shutil
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from typing import Literal, cast

from loguru import logger

from alphafold3tools import __version__
from alphafold3tools.log import log_setup
from alphafold3tools.utils import add_version_option, int_id_to_str_id, to_json


def read_json_data(jsonpath: str) -> dict:
    """Reads AlphaFold3 json data.
    Args:
        jsonpath (str): Path to the json file.
    Returns:
        dict: Json data.
    """
    with open(jsonpath, "r") as file:
        data = json.load(file)
    return data


def write_json_data(outputfile: str, data: dict):
    """Writes AlphaFold3 json data.
    Args:
        outputfile (str): Path to the json file.
        data (dict): Json data.
    """
    with open(outputfile, "w") as file:
        file.write(to_json(data))


def remove_ccdcodes(data: dict, ccdcodes_to_remove: list[str]) -> dict:
    """Removes ligand entities from AlphaFold3 json data.
    Args:
        data (dict): AlphaFold3 json data.
        ccdcodes_to_remove (list[str]): ccdcodes to remove.
    Returns:
        dict: AlphaFold3 json data with ccdcodes removed.
    """
    new_data = copy.deepcopy(data)
    sequence_contents = new_data["sequences"]
    new_sequence_contents = []

    is_removed = False
    for sequence_content in sequence_contents:
        if "ligand" in sequence_content:
            if "ccdCodes" in sequence_content["ligand"]:
                ccd_codes = sequence_content["ligand"]["ccdCodes"]
                if any(ligand in ccd_codes for ligand in ccdcodes_to_remove):
                    logger.info(
                        f"Removing ligand: {sequence_content['ligand']['ccdCodes']}"
                    )
                    is_removed = True
                else:
                    new_sequence_contents.append(sequence_content)
        else:
            new_sequence_contents.append(sequence_content)

    if not is_removed:
        logger.warning(
            f"No ligand with ccdCodes: {ccdcodes_to_remove} found "
            "in the input JSON file."
        )
    new_data["sequences"] = new_sequence_contents
    return new_data


def purge_ligand(data: dict) -> dict:
    """Purges all ligand entities from AlphaFold3 json data.
    Args:
        data (dict): AlphaFold3 json data.
    Returns:
        dict: AlphaFold3 json data with ligands removed.
    """
    new_data = copy.deepcopy(data)
    sequence_contents = new_data["sequences"]
    new_sequence_contents = []

    for sequence_content in sequence_contents:
        if "ligand" in sequence_content:
            if sequence_content["ligand"].get("smiles"):
                logger.info(f"Purging smiles: {sequence_content['ligand']['smiles']}")
            elif sequence_content["ligand"].get("ccdCodes"):
                logger.info(
                    f"Purging ccdCodes: {sequence_content['ligand']['ccdCodes']}"
                )
            else:
                new_sequence_contents.append(sequence_content)
        else:
            new_sequence_contents.append(sequence_content)
    new_data["sequences"] = new_sequence_contents
    return new_data


def add_ligand(
    data: dict,
    ligand_type: Literal["smiles", "ccdCodes"],
    ligand_name: str,
    num_ligand: int,
) -> dict:
    """Adds ligand entities to AlphaFold3 json data.
    Args:
        data (dict): AlphaFold3 json data.
        ligand_type (Literal["smiles", "ccdCodes"]): Type of ligand to add.
        ligand_name (str): Ligand string to add.
        num_ligand (int): Number of ligand molecules to add.
    Returns:
        dict: AlphaFold3 json data with ligands added.
    """
    logger.info(f"Adding {num_ligand} ligand: {ligand_name} as {ligand_type}")
    new_data = copy.deepcopy(data)
    sequence_contents = new_data["sequences"]

    num_ids = [int_id_to_str_id(num) for num in range(1, num_ligand + 1)]
    if ligand_type == "smiles":
        sequence_contents.append(
            {
                "ligand": {
                    "id": num_ids,
                    "smiles": ligand_name,
                }
            }
        )
    elif ligand_type == "ccdCodes":
        sequence_contents.append(
            {
                "ligand": {
                    "id": num_ids,
                    "ccdCodes": [ligand_name],
                }
            }
        )
    return new_data


def fix_sequence_ids(data: dict) -> dict:
    """Fixes the sequence IDs in the AlphaFold3 JSON data.

    This function updates the IDs in the "sequences" field of the provided
    AlphaFold3 JSON data. It ensures that each ID is unique and follows a
    sequential order using the `int_id_to_str_id` function to convert integers
    to string IDs in a reverse spreadsheet style naming
    (e.g., 1 = A, 2 = B, ..., 27 = AA).

    Args:
        data (dict): The AlphaFold3 JSON data containing sequences with IDs to be fixed.

    Returns:
        dict: A new dictionary with the updated sequence IDs.
    """
    new_data = copy.deepcopy(data)
    sequence_contents = new_data["sequences"]

    id_counter = 1  # 1-based indexing.
    for sequence_content in sequence_contents:
        for key in sequence_content:
            if "id" in sequence_content[key]:
                if isinstance(sequence_content[key]["id"], list):
                    new_ids = []
                    for _ in sequence_content[key]["id"]:
                        new_ids.append(int_id_to_str_id(id_counter))
                        id_counter += 1
                    sequence_content[key]["id"] = new_ids
                elif isinstance(sequence_content[key]["id"], str):
                    new_id = int_id_to_str_id(id_counter)
                    sequence_content[key]["id"] = new_id
                    id_counter += 1

    return new_data


def modify_name(data: dict, new_name: str) -> dict:
    """Modifies the job name in the AlphaFold3 JSON data.

    Args:
        data (dict): The AlphaFold3 JSON data.
        new_name (str): The new job name to set.
    Returns:
        dict: A new dictionary with the updated prediction name.
    """
    new_data = copy.deepcopy(data)
    new_data["name"] = new_name
    return new_data


def add_userccd(data: dict, userccd_files: list[str]) -> dict:
    """Adds user provided ccdCodes to the AlphaFold3 JSON data.

    Args:
        data (dict): The AlphaFold3 JSON data.
        userccd_files (list[str]): The path to the user provided ccdCodes file.
        Multiple files can be provided.
    Returns:
        dict: A new dictionary with the updated ccdCodes.
    """
    new_data = copy.deepcopy(data)
    userccd_as_string = ""
    for userccd_file in userccd_files:
        with open(userccd_file, "r") as file:
            userccd_as_string += file.read()
            userccd_as_string += "## \n"

    new_data["userCCD"] = userccd_as_string
    return new_data


def _resolve_chain_a3m(protein: dict) -> str:
    """Resolve the per-chain a3m MSA string for a protein entity.

    AlphaFold3 stores each protein chain's MSA as a per-chain a3m (query first).
    Prefer the inline ``unpairedMsa`` string; otherwise read the file referenced
    by ``unpairedMsaPath``; if neither is available, fall back to a single-sequence
    a3m built from the query ``sequence``.

    Args:
        protein (dict): A protein entity dict from ``data["sequences"]``.
    Returns:
        str: An a3m-format MSA string whose first record is the query.
    """
    unpaired_msa = protein.get("unpairedMsa")
    if isinstance(unpaired_msa, str) and unpaired_msa.strip():
        return unpaired_msa
    unpaired_msa_path = protein.get("unpairedMsaPath")
    if isinstance(unpaired_msa_path, str) and unpaired_msa_path:
        with open(unpaired_msa_path, "r") as f:
            return f.read()
    return f">query\n{protein['sequence']}\n"


def add_templates(
    data: dict,
    *,
    pdb_database_path: str | None,
    seqres_database_path: str | None,
    max_template_date: datetime.date = datetime.date(2099, 12, 31),
    max_subsequence_ratio: float | None = 0.95,
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
    save_hmmsto: bool = False,
    guess_copies: bool = False,
    overwrite: bool = False,
    name: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Add template search results to protein chains in AlphaFold3 json data.

    For each protein entity, runs the template-search pipeline (HMMER + PDB
    mmCIF/SEQRES databases) reusing the same helpers as ``msatojson`` and stores
    the result under ``protein["templates"]``.

    Chains that already have a non-empty ``templates`` list are preserved (skipped)
    unless ``overwrite`` is True. When ``guess_copies`` is True, the chain's copy
    count (``protein["id"]`` list length) is overridden with the homo-oligomer
    count guessed from the first template's biological assembly; the provisional
    ids are renumbered globally afterwards by ``fix_sequence_ids``.

    Args:
        data (dict): AlphaFold3 json data.
        pdb_database_path (str): Directory of the PDB mmCIF database.
        seqres_database_path (str): Path to the PDB SEQRES database file.
        max_template_date (datetime.date): Maximum template release date.
        max_subsequence_ratio (float | None): Maximum subsequence ratio; None
            disables subsequence-ratio filtering.
        hmmbuild_binary_path (str): Path to the hmmbuild binary.
        hmmsearch_binary_path (str): Path to the hmmsearch binary.
        save_hmmsto (bool): Whether to save intermediate HMM sto files.
        guess_copies (bool): Whether to override each chain's copy count from the
            first template's biological assembly.
        overwrite (bool): Whether to re-search and replace existing templates.
        name (str | None): Base name used for saved sto file names.
        output_dir (Path | None): Directory to write sto files when save_hmmsto.
    Returns:
        dict: A new dictionary with templates added.
    """
    # Lazy imports: the template-search stack pulls in gemmi/pandas etc., which
    # should not be paid by plain modjson invocations (e.g. ligand-only edits).
    from alphafold3tools.msatojson import validate_template_search_paths
    from alphafold3tools.searchtemplates import (
        search_templates,
        search_templates_with_hits,
    )
    from alphafold3tools.structure.oligomer import guess_homomer_count_from_store
    from alphafold3tools.structure_stores import StructureStore

    validate_template_search_paths(
        pdb_database_path, seqres_database_path, hmmbuild_binary_path
    )
    new_data = copy.deepcopy(data)
    for chain_index, sequence_content in enumerate(new_data["sequences"]):
        if "protein" not in sequence_content:
            continue
        protein = sequence_content["protein"]
        existing = protein.get("templates")
        if existing and not overwrite:
            logger.info(
                f"Chain {chain_index + 1} already has {len(existing)} template(s); "
                "preserving them (use -O/--overwrite-templates to replace)."
            )
            continue
        a3m_string = _resolve_chain_a3m(protein)
        chain_id = protein.get("id")
        first_chain_id = chain_id[0] if isinstance(chain_id, list) else chain_id
        if save_hmmsto and output_dir is not None:
            sto_path: Path | None = (
                output_dir / f"{name}_{first_chain_id}.hmmsearch.sto"
            )
        else:
            sto_path = None
        logger.info(
            f"Searching templates for chain {chain_index + 1} "
            f"(sequence length {len(protein.get('sequence', ''))})..."
        )
        if guess_copies:
            templates_list, hits_meta = search_templates_with_hits(
                msa_a3m_string=a3m_string,
                pdb_database_path=pdb_database_path,
                seqres_database_path=seqres_database_path,
                hmmsearch_sto_output_path=sto_path,
                max_template_date=max_template_date,
                max_subsequence_ratio=max_subsequence_ratio,
                hmmbuild_binary_path=hmmbuild_binary_path,
                hmmsearch_binary_path=hmmsearch_binary_path,
            )
            if hits_meta:
                pdb_id, hit_chain_id = hits_meta[0]
                assert pdb_database_path is not None  # validated above.
                copies = guess_homomer_count_from_store(
                    StructureStore(pdb_database_path), pdb_id, hit_chain_id
                )
                logger.info(
                    f"Chain {chain_index + 1}: guessed homo-oligomer count = "
                    f"{copies} from template {pdb_id} chain {hit_chain_id}."
                )
                # Provisional ids; renumbered globally by fix_sequence_ids later.
                protein["id"] = [int_id_to_str_id(j + 1) for j in range(copies)]
        else:
            templates_list = search_templates(
                msa_a3m_string=a3m_string,
                pdb_database_path=pdb_database_path,
                seqres_database_path=seqres_database_path,
                hmmsearch_sto_output_path=sto_path,
                max_template_date=max_template_date,
                max_subsequence_ratio=max_subsequence_ratio,
                hmmbuild_binary_path=hmmbuild_binary_path,
                hmmsearch_binary_path=hmmsearch_binary_path,
            )
        protein["templates"] = templates_list
    return new_data


def modjson(
    input,
    output,
    ligands_to_be_added=None,
    purging=False,
    ligands_to_be_removed=None,
    name=None,
    userccd_to_be_added=None,
    debug="SUCCESS",
    includetemplates=False,
    savehmmsto=False,
    pdb_database_path=None,
    seqres_database_path=None,
    max_template_date=datetime.date(2099, 12, 31),
    max_subsequence_ratio=0.95,
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
    guess_copies=False,
    overwrite_templates=False,
) -> None:
    """Modifies AlphaFold3 JSON file.
    Args:
        input (str): Input AlphaFold3 JSON file.
        output (str): Output JSON file.
        ligands_to_be_added (list[list[str]]): Add ligand to the input JSON file.
        purge_ligand (bool): Purge all ligands from the input JSON file at first.
        ligands_to_be_removed (list): Remove ligands with ccdcodes
                                      from the input JSON file.
        name (str): Set the job name in the input JSON file.
        userccd_to_be_added (list[str]): Add user provided ccdCodes
                                         to the input JSON file.
        debug (str): Print lots of debugging statements.
        includetemplates (bool): Whether to run template search and add templates.
        savehmmsto (bool): Whether to save intermediate HMM sto files.
        pdb_database_path (str): Path to the PDB mmCIF database for template search.
        seqres_database_path (str): Path to the PDB SEQRES database for template search.
        max_template_date (datetime.date): Maximum template date for template search.
        max_subsequence_ratio (float | None): Maximum subsequence ratio for template
                                              search.
        hmmbuild_binary_path (str): Path to the hmmbuild binary.
        hmmsearch_binary_path (str): Path to the hmmsearch binary.
        guess_copies (bool): Whether to guess the homo-oligomer count from templates.
        overwrite_templates (bool): Whether to overwrite existing templates.
    """
    logger.info(f"Reading input JSON file: {input}")
    data = read_json_data(input)
    if purging:
        logger.info("Purging current ligand entities from the input JSON file.")
        data = purge_ligand(data)
    if ligands_to_be_removed:
        logger.info("Removing ligand entities from the input JSON file.")
        data = remove_ccdcodes(data, ligands_to_be_removed)
    if ligands_to_be_added:
        logger.info("Adding ligand entities to the input JSON file.")
        for ligand_input in ligands_to_be_added:
            ligand_type, ligand_name, num_ligand = ligand_input
            if ligand_type not in ["smiles", "ccdCodes"]:
                raise ValueError(
                    f"Invalid ligand type: {ligand_type}. "
                    "The ligand type must be either 'smiles' or 'ccdCodes'."
                )
            ligand_type_literal = cast(Literal["smiles", "ccdCodes"], ligand_type)
            data = add_ligand(data, ligand_type_literal, ligand_name, int(num_ligand))
    if includetemplates:
        logger.info("Searching and adding template information to the input JSON file.")
        template_name = data.get("name") or Path(output).stem
        data = add_templates(
            data,
            pdb_database_path=pdb_database_path,
            seqres_database_path=seqres_database_path,
            max_template_date=max_template_date,
            max_subsequence_ratio=max_subsequence_ratio,
            hmmbuild_binary_path=hmmbuild_binary_path,
            hmmsearch_binary_path=hmmsearch_binary_path,
            save_hmmsto=savehmmsto,
            guess_copies=guess_copies,
            overwrite=overwrite_templates,
            name=template_name,
            output_dir=Path(output).parent,
        )
    data = fix_sequence_ids(data)

    if name:
        logger.info(f"Setting the job name to: {name}")
        data = modify_name(data, name)

    if userccd_to_be_added:
        logger.info("Adding user provided ccdCodes to the input JSON file.")
        data = add_userccd(data, userccd_to_be_added)

    logger.info(f"Output JSON file: {output}")
    write_json_data(output, data)


def main():
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Add or remove ligand entities from AlphaFold3 JSON file.",
    )
    add_version_option(parser)
    parser.add_argument(
        "-i",
        "--input_json",
        help="Input AlphaFold3 JSON file. Mandatory.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Output JSON file. Mandatory.",
        type=str,
        required=True,
        metavar="output.json",
    )
    parser.add_argument(
        "-a",
        "--add_ligand",
        help="Add ligand to the input JSON file.\n"
        "Provide 'ligand type', 'ligand name', "
        "and 'number of the ligand molecule'.\n"
        "The 'ligand type' must be either 'smiles' or 'ccdCodes'.\n"
        "Multiple ligands can be added.\n"
        "e.g. -a smiles CCOCCC 1 -a ccdCodes PRD 2",
        dest="ligands_to_be_added",
        type=str,
        nargs=3,
        action="append",
        metavar=("ligand_type", "ligand_name", "number_of_ligand"),
    )
    parser.add_argument(
        "-p",
        "--purge_ligand",
        dest="purging",
        help="Purge all ligands from the input JSON file at first.",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--remove_ccdcodes",
        help="Remove ligands with ccdcodes from the input JSON file. Multiple ccdcodes "
        "can be provided.\n"
        "e.g. -r PRD ATP",
        dest="ligands_to_be_removed",
        type=str,
        nargs="*",
        metavar="ccdcode",
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Set the job name in the input JSON file. i.e. data['name'] = name",
        type=str,
        metavar="new prediction name",
    )
    parser.add_argument(
        "-u",
        "--add_userccd",
        help="Add user provided ccdCodes to the input JSON file.\n"
        "Multiple files can be provided.\n"
        "e.g. -u userccd1.cif userccd2.cif",
        type=str,
        dest="userccd_to_be_added",
        nargs="*",
        metavar="userccd_file",
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
    parser.add_argument(
        "--include_templates",
        help="Search and add template information to each protein chain.\n"
        "Requires --pdb_database_path, --seqres_database_path and HMMER binaries.\n"
        "Chains that already have templates are preserved unless "
        "-O/--overwrite-templates is given.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-O",
        "--overwrite-templates",
        dest="overwrite_templates",
        help="Overwrite existing template information when --include_templates is set.\n"
        "Only affects template overwriting; other entities are unaffected.",
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
        help="Maximum template date for template search in YYYY-MM-DD format. "
        "Default is 2099-12-31.",
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
    parser.add_argument(
        "--guess_copies",
        "--guess-copies",
        dest="guess_copies",
        help="Guess the homo-oligomer count of each protein chain from the "
        "biological assembly of its first template (PDB ID + chain ID) and set the "
        "number of chain copies (id list length) accordingly, overriding the "
        "existing copy count. Requires --include_templates and --pdb_database_path.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()
    log_setup(args.loglevel)
    if args.guess_copies and not args.include_templates:
        parser.error("--guess_copies requires --include_templates.")
    if args.guess_copies and args.pdb_database_path is None:
        parser.error("--guess_copies requires --pdb_database_path.")
    if args.overwrite_templates and not args.include_templates:
        logger.warning(
            "-O/--overwrite-templates has no effect without --include_templates; "
            "ignoring."
        )
    if args.include_templates:
        # Fail fast (before mutating anything) if any template-search path is missing.
        from alphafold3tools.msatojson import validate_template_search_paths

        validate_template_search_paths(
            args.pdb_database_path,
            args.seqres_database_path,
            args.hmmbuild_binary_path,
        )
    max_subsequence_ratio = args.max_subsequence_ratio
    if max_subsequence_ratio == 1.0:
        logger.success(
            "No templates will be excluded based on subsequence ratio since "
            "max_subsequence_ratio is set to 1.0."
        )
        max_subsequence_ratio = None
    modjson(
        args.input_json,
        args.out,
        args.ligands_to_be_added,
        args.purging,
        args.ligands_to_be_removed,
        args.name,
        args.userccd_to_be_added,
        args.loglevel,
        includetemplates=args.include_templates,
        savehmmsto=args.save_hmmsto,
        pdb_database_path=args.pdb_database_path,
        seqres_database_path=args.seqres_database_path,
        max_template_date=args.max_template_date,
        max_subsequence_ratio=max_subsequence_ratio,
        hmmbuild_binary_path=args.hmmbuild_binary_path,
        hmmsearch_binary_path=args.hmmsearch_binary_path,
        guess_copies=args.guess_copies,
        overwrite_templates=args.overwrite_templates,
    )


if __name__ == "__main__":
    main()
