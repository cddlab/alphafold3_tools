from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass

import gemmi
import rdkit.Chem.rdMolDescriptors
from loguru import logger

from alphafold3tools.log import log_setup


def get_ccd_cif_chiral_type(atom):
    """Translate atom chiral from rdkit to the CCD CIF format.
    Controlled dictionary from:
    http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp_atom.pdbx_stereo_config.html

    Args:
        atom (rdkit.Chem.rdchem.Atom): rdkit atom

    Returns:
        str: chiral type for the '_chem_comp_atom.pdbx_stereo_config'
            field. If none of the rdkit atom types can be matched 'N' is
            returned.
    """
    chiral_type = atom.GetChiralTag()

    if chiral_type == rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:
        return "R"

    if chiral_type == rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return "S"

    return "N"


def get_atom_name(atom):
    """Gets atom name. If not set ElementSymbol + Id is used.

    Args:
        atom (rdkit.Chem.rdchem.Atom): rdkit atom.

    Returns:
        str: Name of the atom.
    """
    return (
        atom.GetProp("name")
        if atom.HasProp("name")
        else atom.GetSymbol() + str(atom.GetIdx() + 1)
    )


def get_ccd_cif_bond_type(bond):
    """Translate bond type from rdkit to the CCD CIF format.
    Controlled dictionary from:
    http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp_bond.value_order.html

    Args:
        bond_order (rdkit.Chem.rdchem.Bond): rdkit molecular bond.

    Returns:
        str: bond type for the '_chem_comp_bond.value_order' field. If
            none of the rdkit bond types can be matched SING is returned.
    """
    bond_order = bond.GetBondType()

    if bond_order == rdkit.Chem.rdchem.BondType.SINGLE:
        return "SING"

    if bond_order == rdkit.Chem.rdchem.BondType.DOUBLE:
        return "DOUB"

    if bond_order == rdkit.Chem.rdchem.BondType.TRIPLE:
        return "TRIP"

    if bond_order == rdkit.Chem.rdchem.BondType.AROMATIC:
        return "AROM"

    if bond_order == rdkit.Chem.rdchem.BondType.QUADRUPLE:
        return "QUAD"

    return "SING"


def get_ccd_cif_bond_stereo(bond):
    """Get bond stereochemistry information to be used in CCD CIF file.
    Controlled dictionary from: http://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_chem_comp_bond.pdbx_stereo_config.html

    Args:
        bond (rdkit.Chem.rdchem.Bond): Molecular bond.

    Returns:
        str: bond stereochemistry information for the field
            '_chem_comp_bond.pdbx_stereo_config'.
    """

    stereo = bond.GetStereo()

    if stereo in (
        rdkit.Chem.rdchem.BondStereo.STEREOE,
        rdkit.Chem.rdchem.BondStereo.STEREOCIS,
    ):
        return "E"

    if stereo in (
        rdkit.Chem.rdchem.BondStereo.STEREOZ,
        rdkit.Chem.rdchem.BondStereo.STEREOTRANS,
    ):
        return "Z"

    return "N"


def write_info_block(
    mol: rdkit.Chem.rdchem.Mol,
    cif_block: gemmi.cif.Block,
    mol_name: str,
    compname: str | None = None,
) -> None:
    """Write the _chem_comp block of the CIF file.

    Args:
        cif_block (gemmi.cif.Block): CIF block to write the information.
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule.
        mol_name (str): Name of the molecule.
    """
    calc_formula = rdkit.Chem.rdMolDescriptors.CalcMolFormula(mol)
    calc_weight = rdkit.Chem.rdMolDescriptors.CalcExactMolWt(mol)

    label = "_chem_comp."
    if compname is None:
        comp_name = mol_name
    cif_block.set_pairs(
        label,
        {
            "id": mol_name,
            "name": f"'{comp_name}'",
            "type": "NON-POLYMER",
            "formula": calc_formula,
            "mon_nstd_parent_comp_id": "?",
            "pdbx_synonyms": "?",
            "formula_weight": f"{calc_weight:.3f}",
        },
        raw=True,
    )


def write_atom_block(mol, cif_block: gemmi.cif.Block, mol_name: str) -> None:
    """Write the _chem_comp_atom block of the CIF file.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule.
        cif_block (gemmi.cif.Block): CIF block to write the information.
        mol_name (str): Name of the molecule.
    """
    conf = mol.GetConformer()
    label = "_chem_comp_atom."
    atom_fields = [
        "comp_id",
        "atom_id",
        "type_symbol",
        "charge",
        "pdbx_aromatic_flag",
        "pdbx_leaving_atom_flag",
        "pdbx_model_Cartn_x_ideal",
        "pdbx_model_Cartn_y_ideal",
        "pdbx_model_Cartn_z_ideal",
        "pdbx_component_atom_id",
        "pdbx_component_comp_id",
        "pdbx_ordinal",  # 1-origin index
    ]
    atom_loop = cif_block.init_loop(label, atom_fields)

    for atom in mol.GetAtoms():
        at_id = atom.GetIdx()
        model_atom = conf.GetAtomPosition(at_id)

        new_row = [
            mol_name,
            get_atom_name(atom),
            atom.GetSymbol(),
            str(atom.GetFormalCharge()),
            "Y" if atom.GetIsAromatic() else "N",
            "N",
            f"{model_atom.x:.3f}",
            f"{model_atom.y:.3f}",
            f"{model_atom.z:.3f}",
            get_atom_name(atom),
            mol_name,
            str(atom.GetIdx() + 1),
        ]
        atom_loop.add_row(gemmi.cif.quote_list(new_row))


def write_bond_block(mol, cif_block: gemmi.cif.Block, mol_name: str) -> None:
    """Write the _chem_comp_bond block of the CIF file.

    Args:
        cif_block (gemmi.cif.Block): CIF block to write the information.
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule.
        mol_name (str): Name of the molecule.
    """
    label = "_chem_comp_bond."
    bond_fields = [
        "comp_id",
        "atom_id_1",
        "atom_id_2",
        "value_order",
        "pdbx_aromatic_flag",
        "pdbx_stereo_config",
        "pdbx_ordinal",
    ]
    bond_loop = cif_block.init_loop(label, bond_fields)

    for b in mol.GetBonds():
        atom_a = b.GetBeginAtom()
        atom_b = b.GetEndAtom()

        new_row = [
            mol_name,
            get_atom_name(atom_a),
            get_atom_name(atom_b),
            get_ccd_cif_bond_type(b),
            "Y" if b.GetIsAromatic() else "N",
            get_ccd_cif_bond_stereo(b),
            str(b.GetIdx() + 1),
        ]
        bond_loop.add_row(gemmi.cif.quote_list(new_row))


def write_pdb_ccd_cif_descriptor(mol, cif_block) -> None:
    """Writes the _pdbx_chem_comp_descriptor namespace with details.

    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit molecule.
        cif_block (cif.Block): mmcif Block object from gemmi.
    """

    smiles = rdkit.Chem.MolToSmiles(mol, allHsExplicit=False)
    label = "_pdbx_chem_comp_descriptor."
    cif_block.set_pairs(
        label,
        {
            "type": "SMILES_CANONICAL",
            "descriptor": f"'{smiles}'",
        },
        raw=True,
    )


def convert_sdf_to_ccd(sdffile: str, mol_name: str, removeHs: bool = True) -> str:
    """Converts an SDF file to a CCD CIF file.

    Args:
        sdffile (str): Path to the SDF file.
        outciffile (str): Path to the output CIF file.
        mol_name (str): Name of the molecule.
        removeHs (bool): Remove hydrogens from the molecule.
    Returns:
        str: as_string of the CIF file.
    """
    doc = gemmi.cif.Document()
    mol = rdkit.Chem.SDMolSupplier(
        sdffile, sanitize=True, removeHs=removeHs, strictParsing=False
    )[0]
    cif_block = doc.add_new_block(mol_name)
    write_info_block(mol, cif_block, mol_name)
    write_atom_block(mol, cif_block, mol_name)
    write_bond_block(mol, cif_block, mol_name)
    write_pdb_ccd_cif_descriptor(mol, cif_block)
    options = gemmi.cif.WriteOptions()
    # Add '# ' after each block
    cif_string = doc.as_string(options).replace("\n\n", "\n# \n")
    return cif_string


def main():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description="Converts a3m-format MSA file to AlphaFold3 input JSON file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Input SDF file. e.g. strigolactone.sdf",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--name",
        help="3-letter code of the ligand (e.g. STR), but more than 3 letters "
        "are allowed.",
        type=str,
        required=True,
    )
    parser.add_argument("-o", "--out", help="Output CIF file.", type=str, required=True)
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
    with open(args.out, "w") as f:
        f.write(
            convert_sdf_to_ccd(
                args.input,
                args.name,
            )
        )


if __name__ == "__main__":
    main()
