from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import gemmi
import rdkit.Chem.rdMolDescriptors
from loguru import logger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from alphafold3tools.log import log_setup
from alphafold3tools.utils import add_version_option


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


def get_atom_name(atom: rdkit.Chem.rdchem.Atom) -> str:
    """Gets atom name. If not set ElementSymbol + Id is used.

    Args:
        atom (rdkit.Chem.rdchem.Atom): rdkit atom.

    Returns:
        str: Name of the atom.
    """
    return atom.GetProp("name") if atom.HasProp("name") else atom.GetSymbol() + str(atom.GetIdx() + 1)


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


def get_ccd_cif_bond_stereo(bond: rdkit.Chem.rdchem.Bond) -> str:
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
    comp_name = mol_name if compname is None else compname
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


def candidate_charges(initial_charge: int, max_shift: int = 4) -> list[int]:
    """List total charges to try, ordered by distance from initial_charge.

    Args:
        initial_charge (int): Total formal charge written in the SDF file.
        max_shift (int): Maximum deviation from initial_charge to try.

    Returns:
        list[int]: e.g. [0, -1, 1, -2, 2, ...] for initial_charge=0.
    """
    charges = [initial_charge]
    for shift in range(1, max_shift + 1):
        charges.extend([initial_charge - shift, initial_charge + shift])
    return charges


def reassign_bond_orders(mol: rdkit.Chem.rdchem.Mol, initial_charge: int | None = None) -> rdkit.Chem.rdchem.Mol:
    """Re-derive bond orders and formal charges from connectivity.

    Some programs (e.g. GaussView) write SDF files that RDKit cannot sanitize,
    such as carboxylates drawn as C(=O)=O or conjugated chains marked with
    aromatic (type 4) bonds outside rings. Since such files carry explicit
    hydrogens, the bond orders can be determined from the connectivity alone.
    The total charge is unknown, so it is searched starting from
    initial_charge.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Unsanitized molecule with explicit
            hydrogens and 3D coordinates.
        initial_charge (int | None): Total charge to try first. If None, the
            sum of the formal charges in mol is used.

    Returns:
        rdkit.Chem.rdchem.Mol: Sanitized molecule with the same atom order.

    Raises:
        ValueError: If no consistent bond ordering could be found.
    """
    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        raise ValueError(
            "Could not determine bond orders because the input file has no "
            "hydrogens. Add all hydrogens explicitly and try again."
        )
    if initial_charge is None:
        initial_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    for charge in candidate_charges(initial_charge):
        fixed = rdkit.Chem.Mol(mol)
        for atom in fixed.GetAtoms():
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(True)
            atom.SetIsAromatic(False)
        for bond in fixed.GetBonds():
            bond.SetIsAromatic(False)
        try:
            rdDetermineBonds.DetermineBondOrders(fixed, charge=charge)
            rdkit.Chem.SanitizeMol(fixed)
        except (ValueError, RuntimeError):
            continue
        # Charged carbons indicate missing hydrogens rather than a real solution.
        if any(atom.GetSymbol() == "C" and atom.GetFormalCharge() != 0 for atom in fixed.GetAtoms()):
            continue
        logger.warning(
            f"Bond orders were re-assigned from connectivity (total charge: {charge:+d}). Please check the output."
        )
        return fixed
    raise ValueError(
        "Could not determine valid bond orders. Make sure that all hydrogens are explicitly included in the input file."
    )


def read_sdf_molecule(sdffile: str, removeHs: bool = True) -> rdkit.Chem.rdchem.Mol:
    """Read the first molecule of an SDF file.

    If RDKit fails to sanitize the molecule, the bond orders are re-assigned
    from connectivity (see reassign_bond_orders).

    Args:
        sdffile (str): Path to the SDF file.
        removeHs (bool): Remove hydrogens from the molecule.

    Returns:
        rdkit.Chem.rdchem.Mol: Sanitized molecule.

    Raises:
        ValueError: If no molecule could be read from the file.
    """
    supplier = rdkit.Chem.SDMolSupplier(sdffile, sanitize=True, removeHs=removeHs, strictParsing=False)
    if len(supplier) == 0:
        raise ValueError(f"No molecule found in {sdffile}")
    mol = supplier[0]
    if mol is not None:
        return mol
    logger.warning(f"RDKit could not sanitize {sdffile}. Trying to fix bond orders.")
    raw_mol = rdkit.Chem.SDMolSupplier(sdffile, sanitize=False, removeHs=False, strictParsing=False)[0]
    if raw_mol is None:
        raise ValueError(f"Could not parse the molecule in {sdffile}")
    mol = reassign_bond_orders(raw_mol)
    rdkit.Chem.AssignStereochemistryFrom3D(mol)
    return rdkit.Chem.RemoveHs(mol) if removeHs else mol


MOL2_BOND_TYPES = {
    "1": rdkit.Chem.rdchem.BondType.SINGLE,
    "2": rdkit.Chem.rdchem.BondType.DOUBLE,
    "3": rdkit.Chem.rdchem.BondType.TRIPLE,
    "ar": rdkit.Chem.rdchem.BondType.AROMATIC,
}


def mol2_element(atom_type: str) -> str:
    """Get the element symbol from a SYBYL or GAFF atom type in a mol2 file.

    Args:
        atom_type (str): Atom type, e.g. 'C.ar', 'O.co2', 'Cl' (SYBYL) or
            'c3', 'ha', 'cl' (GAFF/GAFF2).

    Returns:
        str: Element symbol, e.g. 'C', 'O', 'Cl'.

    Raises:
        ValueError: If no element can be derived from the atom type.
    """
    if "." in atom_type or atom_type[:1].isupper():
        # SYBYL type, e.g. 'C.ar' or 'Cl'
        element = atom_type.split(".")[0]
    elif atom_type[:2] in ("cl", "br"):
        element = atom_type[:2].capitalize()
    else:
        # GAFF type: lowercase element followed by a hybridization label
        element = atom_type[:1].upper()
    try:
        rdkit.Chem.GetPeriodicTable().GetAtomicNumber(element)
    except RuntimeError:
        raise ValueError(f"Unknown atom type in mol2 file: {atom_type}") from None
    return element


def parse_mol2_sections(mol2file: str) -> dict[str, list[list[str]]]:
    """Split the first molecule of a mol2 file into its record sections.

    Args:
        mol2file (str): Path to the mol2 file.

    Returns:
        dict[str, list[list[str]]]: Section name (e.g. 'ATOM', 'BOND') to the
            whitespace-separated fields of each non-empty line.
    """
    sections: dict[str, list[list[str]]] = {}
    current = None
    with open(mol2file) as f:
        for line in f:
            if line.startswith("@<TRIPOS>"):
                current = line.removeprefix("@<TRIPOS>").strip()
                if current == "MOLECULE" and current in sections:
                    break
                sections[current] = []
            elif current is not None and line.strip() and not line.startswith("#"):
                sections[current].append(line.split())
    return sections


def read_mol2_molecule(mol2file: str, removeHs: bool = True) -> rdkit.Chem.rdchem.Mol:
    """Read the first molecule of a mol2 file, keeping its atom names.

    The atom names, coordinates, atom order and bond order of the mol2 file
    are kept. If the bond orders are not chemically valid (e.g. carboxylates
    written with two single bonds), they are re-assigned from connectivity
    (see reassign_bond_orders).

    Args:
        mol2file (str): Path to the mol2 file.
        removeHs (bool): Remove hydrogens from the molecule.

    Returns:
        rdkit.Chem.rdchem.Mol: Sanitized molecule. Each atom has the mol2
            atom name as the 'name' property.

    Raises:
        ValueError: If the file has no atoms or the atom names are not unique.
    """
    sections = parse_mol2_sections(mol2file)
    atom_records = sections.get("ATOM", [])
    if not atom_records:
        raise ValueError(f"No @<TRIPOS>ATOM records found in {mol2file}")
    names = [fields[1] for fields in atom_records]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Atom names must be unique in {mol2file}: {', '.join(duplicates)}")

    rwmol = rdkit.Chem.RWMol()
    conf = rdkit.Chem.Conformer(len(atom_records))
    index_of = {}  # mol2 atom id -> rdkit atom index
    partial_charge = 0.0
    for fields in atom_records:
        atom = rdkit.Chem.Atom(mol2_element(fields[5]))
        atom.SetProp("name", fields[1])
        atom.SetNoImplicit(True)
        idx = rwmol.AddAtom(atom)
        index_of[fields[0]] = idx
        conf.SetAtomPosition(idx, Point3D(*(float(v) for v in fields[2:5])))
        if len(fields) > 8:
            partial_charge += float(fields[8])
    rwmol.AddConformer(conf, assignId=True)

    for fields in sections.get("BOND", []):
        begin, end = index_of[fields[1]], index_of[fields[2]]
        # 'am', 'du', 'un' and 'nc' bonds are treated as single bonds
        bond_type = MOL2_BOND_TYPES.get(fields[3], rdkit.Chem.rdchem.BondType.SINGLE)
        rwmol.AddBond(begin, end, bond_type)
        if bond_type == rdkit.Chem.rdchem.BondType.AROMATIC:
            rwmol.GetBondBetweenAtoms(begin, end).SetIsAromatic(True)
            rwmol.GetAtomWithIdx(begin).SetIsAromatic(True)
            rwmol.GetAtomWithIdx(end).SetIsAromatic(True)
    raw_mol = rwmol.GetMol()

    mol = rdkit.Chem.Mol(raw_mol)
    try:
        rdkit.Chem.SanitizeMol(mol)
        # Atoms with radicals have too few bonds for their valence
        is_valid = not any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    except (ValueError, RuntimeError):
        is_valid = False
    if not is_valid:
        logger.warning(f"Bond orders in {mol2file} are not valid. Trying to fix bond orders.")
        mol = reassign_bond_orders(raw_mol, initial_charge=round(partial_charge))
    rdkit.Chem.AssignStereochemistryFrom3D(mol)
    return rdkit.Chem.RemoveHs(mol) if removeHs else mol


def read_molecule(path: str, removeHs: bool = True) -> rdkit.Chem.rdchem.Mol:
    """Read a molecule from an SDF or mol2 file, chosen by the file extension.

    Args:
        path (str): Path to the SDF or mol2 file.
        removeHs (bool): Remove hydrogens from the molecule.

    Returns:
        rdkit.Chem.rdchem.Mol: Sanitized molecule.
    """
    if Path(path).suffix.lower() == ".mol2":
        return read_mol2_molecule(path, removeHs=removeHs)
    return read_sdf_molecule(path, removeHs=removeHs)


def neutralize_oxyanions(mol: rdkit.Chem.rdchem.Mol) -> rdkit.Chem.rdchem.Mol:
    """Represent singly bonded O(-1) atoms as neutral OH.

    e.g. a carboxylate C(=O)[O-] is written as C(=O)O. Oxygens next to a
    positively charged atom (e.g. nitro groups [N+](=O)[O-]) are kept as is.

    Args:
        mol (rdkit.Chem.rdchem.Mol): Sanitized molecule.

    Returns:
        rdkit.Chem.rdchem.Mol: Sanitized molecule with the same atom order.
    """
    rwmol = rdkit.Chem.RWMol(mol)
    neutralized = []
    for atom in rwmol.GetAtoms():
        if (
            atom.GetSymbol() == "O"
            and atom.GetFormalCharge() == -1
            and all(b.GetBondType() == rdkit.Chem.rdchem.BondType.SINGLE for b in atom.GetBonds())
            and all(n.GetFormalCharge() <= 0 for n in atom.GetNeighbors())
        ):
            atom.SetFormalCharge(0)
            atom.SetNoImplicit(False)
            neutralized.append(get_atom_name(atom))
    if not neutralized:
        return mol
    rdkit.Chem.SanitizeMol(rwmol)
    logger.info(f"O(-1) atoms are written as neutral OH: {', '.join(neutralized)}")
    return rwmol.GetMol()


def convert_sdf_to_ccd(sdffile: str, mol_name: str, removeHs: bool = True) -> str:
    """Converts an SDF or mol2 file to a CCD CIF file.

    Args:
        sdffile (str): Path to the SDF or mol2 file.
        mol_name (str): Name of the molecule.
        removeHs (bool): Remove hydrogens from the molecule.
    Returns:
        str: as_string of the CIF file.
    """
    doc = gemmi.cif.Document()
    mol = neutralize_oxyanions(read_molecule(sdffile, removeHs=removeHs))
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
        description="Converts an SDF or mol2 file to AlphaFold3 user-provided CCD (mmCIF) file.",
    )
    add_version_option(parser)
    parser.add_argument(
        "-i",
        "--input",
        help="Input SDF or mol2 file. e.g. strigolactone.sdf. Atom names in mol2 files are kept.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--name",
        help="3-letter code of the ligand (e.g. STR), but more than 3 letters are allowed.",
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
