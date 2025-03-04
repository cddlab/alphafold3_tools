# %%
#!/usr/bin/env python3
import copy
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Literal, cast

from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.utils import int_id_to_str_id

log_setup()


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
        json.dump(data, file, indent=4)


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


def purge_ligands(data: dict) -> dict:
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


def modjson(
    input,
    output,
    add_ligand,
    purge_ligands,
    remove_ccdcodes=None,
    name=None,
    userccdfiles=None,
    debug="SUCCESS",
) -> None:
    """Modifies AlphaFold3 JSON file.
    Args:
        input (str): Input AlphaFold3 JSON file.
        output (str): Output JSON file.
        add_ligand (list[str]): Add ligand to the input JSON file.
        purge_ligands (bool): Purge all ligands from the input JSON file at first.
        remove_ccdcodes (list): Remove ligands with ccdcodes from the input JSON file.
        name (str): Set the job name in the input JSON file.
        userccdfiles (list[str]): Add user provided ccdCodes to the input JSON file.
        debug (str): Print lots of debugging statements.
    """
    data = read_json_data(input)
    log_setup(debug)

    if purge_ligands:
        logger.info("Purging current ligand entities from the input JSON file.")
        data = purge_ligands(data)
    if remove_ccdcodes:
        data = remove_ccdcodes(data, remove_ccdcodes)
    if add_ligand:
        ligand_type, ligand_name, num_ligand = add_ligand
        if ligand_type not in ["smiles", "ccdCodes"]:
            raise ValueError(
                f"Invalid ligand type: {ligand_type}. "
                "The ligand type must be either 'smiles' or 'ccdCodes'."
            )
        ligand_type_literal = cast(Literal["smiles", "ccdCodes"], ligand_type)
        data = add_ligand(data, ligand_type_literal, ligand_name, int(num_ligand))
    data = fix_sequence_ids(data)

    if name:
        logger.info(f"Setting the job name to: {name}")
        data = modify_name(data, name)

    if userccdfiles:
        data = add_userccd(data, userccdfiles)

    write_json_data(output, data)


def main():
    parser = ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Add or remove ligand entities from AlphaFold3 JSON file.",
    )
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
        type=str,
        nargs=3,
        action="append",
        metavar=("ligand_type", "ligand_name", "number_of_ligand"),
    )
    parser.add_argument(
        "-p",
        "--purge_ligands",
        help="Purge all ligands from the input JSON file at first.",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--remove_ccdcodes",
        help="Remove ligands with ccdcodes from the input JSON file. Multiple ccdcodes "
        "can be provided.\n"
        "e.g. -r PRD ATP",
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
    args = parser.parse_args()
    modjson(
        args.input_json,
        args.out,
        args.add_ligand,
        args.purge_ligands,
        args.remove_ccdcodes,
        args.name,
        args.add_userccd,
        args.loglevel,
    )


if __name__ == "__main__":
    main()
