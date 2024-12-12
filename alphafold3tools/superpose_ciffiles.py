import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import gemmi
from loguru import logger

from alphafold3tools.log import log_setup


def superpose_ciffiles(
    inputdir: str, outciffile: str = "", chain_id: str = "A"
) -> None:
    """Superpose all models in the input directory and write to the output CIF file.
    Args:
        inputdir (str): input directory containing model.cif files
        outciffile (str): output CIF file. If not specified,
                          the output file will be named "_superposed.cif"
                          in the input directory.
    """
    if outciffile == "":
        # if inputdir ends with "/", remove it.
        if inputdir.endswith("/"):
            inputdir = inputdir[:-1]
        outciffile = os.path.join(
            inputdir, os.path.basename(inputdir) + "_superposed.cif"
        )
    logger.info(f"Output file: {outciffile}")
    # get all model.cif files in the subdirectories of the input directory
    # remove the best model from the list
    ciffiles = [
        ciffile
        for ciffile in sorted(list(gemmi.CifWalk(inputdir)))
        if ciffile.endswith("/model.cif")
    ]
    outdoc = gemmi.cif.Document()
    for ciffile in ciffiles:
        subdirname = os.path.basename(os.path.dirname(ciffile))
        block = gemmi.cif.read(str(ciffile)).sole_block()
        # pass the first model as a reference
        if ciffile == ciffiles[0]:
            block.name = subdirname
            ref_name = subdirname
            logger.info(f"First model: {subdirname} as a reference")
            firstmodel = gemmi.make_structure_from_block(block)[0]
            outdoc.add_copied_block(block)
        else:
            # obtain a Structure / Model object from cif.Block
            structure = gemmi.make_structure_from_block(block)
            model = structure[0]
            # translate and rotate the model to align with the first model
            # see https://gemmi.readthedocs.io/en/latest/analysis.html#superposition
            if model.find_chain(chain_id) is None:
                raise ValueError(f"Chain {chain_id} not found in {ciffile}")
            ptype = model[chain_id].get_polymer().check_polymer_type()
            logger.info(f"Polymer type: {ptype}")
            if ptype == gemmi.PolymerType.Unknown:
                raise ValueError(f"Unknown polymer type: chain {chain_id} in {ciffile}")
            sup = gemmi.calculate_superposition(
                firstmodel[chain_id].get_polymer(),
                model[chain_id].get_polymer(),
                ptype,
                gemmi.SupSelect.MainChain,
                trim_cycles=3,
                trim_cutoff=2.0,
            )
            logger.info(
                f"{subdirname}: Superposition onto {ref_name}. RMSD: {sup.rmsd:.2f} Å"
            )
            model.transform_pos_and_adp(sup.transform)
            new_block = structure.make_mmcif_block()
            new_block.name = subdirname
            outdoc.add_copied_block(new_block)

            outdoc.write_file(outciffile)


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Converts a3m-format MSA file to AlphaFold3 input JSON file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="The input directory. Specify the directory generated by AlphaFold3. "
        "This command reads 'model.cif' in the subdirectories of the "
        "specified directory. '<input_directory>_model.cif' placed "
        "directly in the input directory will not be read.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--out",
        help="Output CIF file name for superposed structures. "
        "If not specified, it will be '<input_directory>_superposed.cif'.",
        default="",
        type=str,
    )
    parser.add_argument(
        "-c",
        "--chain",
        help="Chain ID to superpose. Default: A",
        default="A",
        type=str,
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
    if os.path.isdir(args.input):
        superpose_ciffiles(args.input, args.out, args.chain)
    else:
        raise ValueError(f"Input directory not found: {args.input}")


if __name__ == "__main__":
    main()
