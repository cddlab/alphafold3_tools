import gemmi


def fix_arginine_residues(residue: gemmi.Residue) -> None:
    """
    Fix arginine residue atom names if NH1 and NH2 are swapped.

    Args:
        residue: gemmi.Residue object, residue to be checked and fixed
    """
    if residue.name != "ARG":
        return
    else:
        # search for indices of relevant atoms
        cd = residue.find_atom("CD", "*")
        nh1 = residue.find_atom("NH1", "*")
        nh2 = residue.find_atom("NH2", "*")
        hh11 = residue.find_atom("HH11", "*")
        hh21 = residue.find_atom("HH21", "*")
        hh12 = residue.find_atom("HH12", "*")
        hh22 = residue.find_atom("HH22", "*")
        # If CD, NH1, or NH2 atoms are missing, no fix is needed
        if cd is None or nh1 is None or nh2 is None:
            return

        def _distance_squared(pos1: gemmi.Position, pos2: gemmi.Position) -> float:
            """Calculate squared distance between two 3D positions.
            Use Gemmi Position class for positions.
            Args:
                pos1: First atom position
                pos2: Second atom position
            Returns:
                Squared distance between pos1 and pos2 (float)
            """
            dx = pos1.x - pos2.x
            dy = pos1.y - pos2.y
            dz = pos1.z - pos2.z
            return dx * dx + dy * dy + dz * dz

        # If NH1 is closer to CD, no fix is needed (correct order)
        if _distance_squared(nh1.pos, cd.pos) <= _distance_squared(nh2.pos, cd.pos):
            return
        # Swap NH1 and NH2 atoms
        nh1.name, nh2.name = nh2.name, nh1.name
        # Swap corresponding hydrogen atoms as well
        if hh11 is not None and hh21 is not None:
            hh11.name, hh21.name = hh21.name, hh11.name
        elif hh11 is not None:
            hh11.name = "HH21"
        elif hh21 is not None:
            hh21.name = "HH11"

        if hh12 is not None and hh22 is not None:
            hh12.name, hh22.name = hh22.name, hh12.name
        elif hh12 is not None:
            hh12.name = "HH22"
        elif hh22 is not None:
            hh22.name = "HH12"


def _process_altloc_groups_whole(group: gemmi.ResidueGroup) -> None:
    """
    Process a residue group where multiple residue types exist at the same position.

    This function handles cases where different residue types (e.g., SEP vs SER,
    HIS vs ARG) occupy the same position with different alternative location IDs.

    Processing flow:
    1. For each residue type in the group:
       - Group atoms by alternative location ID (A, B, C, etc.)
       - Calculate average occupancy for each alt-loc group
       - Sum all average occupancies to get total score for this residue type

    2. Select the best residue type:
       - Choose the residue type with highest total average occupancy
       - Within that residue type, select the alt-loc with highest occupancy
       - Break ties alphabetically (A > B > C)

    3. Remove unselected conformations:
       - Delete all atoms from non-selected residue types
       - Delete atoms with non-selected alt-loc IDs from selected residue type
       - Clear alt-loc markers from remaining atoms

    Args:
        group: gemmi.ResidueGroup containing residues at the same sequence position

    Example:
        If position 58 has SEP(A) with avg occ 0.60 and SER(C) with avg occ 0.40,
        SEP(A) will be selected, SER(C) will be removed, and 'A' markers cleared.
    """
    best_occupancy = -1.0
    for residue in group:
        altloc_groups = {}
        altloc_occupancies = {}
        altloc_counts = {}

        for atom in residue:
            if atom.has_altloc():
                altloc_id = atom.altloc
                if altloc_id not in altloc_groups:
                    altloc_groups[altloc_id] = []
                    altloc_occupancies[altloc_id] = 0.0
                    altloc_counts[altloc_id] = 0

                altloc_groups[altloc_id].append(atom)
                altloc_occupancies[altloc_id] += atom.occ
                altloc_counts[altloc_id] += 1

        if not altloc_groups:
            continue

        avg_occupancy_by_altloc = {
            altloc_id: altloc_occupancies[altloc_id] / altloc_counts[altloc_id]
            for altloc_id in altloc_groups.keys()
        }
        total_avg_occupancy = sum(avg_occupancy_by_altloc.values())

        if total_avg_occupancy > best_occupancy:
            best_occupancy = total_avg_occupancy
            best_group = residue
            best_altloc = max(
                avg_occupancy_by_altloc.items(),
                key=lambda x: (x[1], -ord(x[0])),
            )[0]
    # remove group and altlocs not selected
    for residue in group:
        if residue != best_group:
            # For not best_group, remove all atoms
            atoms_to_remove = [(atom.name, atom.altloc) for atom in residue]
            for atom_name, altloc in atoms_to_remove:
                residue.remove_atom(atom_name, altloc)
        else:
            atoms_to_remove = []
            for atom in residue:
                if atom.has_altloc() and atom.altloc != best_altloc:
                    atoms_to_remove.append((atom.name, atom.altloc))

            for atom_name, altloc in atoms_to_remove:
                residue.remove_atom(atom_name, altloc)

            # clear altloc for selected atoms
            for atom in residue:
                if atom.has_altloc() and atom.altloc == best_altloc:
                    atom.altloc = "\x00"


def _process_altloc_group_partial(group: gemmi.ResidueGroup) -> None:
    """
    process the residues with partially alternative locations.
    Leave the atoms with the highest occupancy for each atom name.

    Args:
        group: gemmi.ResidueGroup object
    """
    if len(group) > 1:
        raise ValueError("This function processes a single residue only.")
    residue = group[0]
    # grouping by atom names
    atom_groups = {}

    for atom in residue:
        atom_name = atom.name
        if atom_name not in atom_groups:
            atom_groups[atom_name] = []

        if atom.has_altloc():
            atom_groups[atom_name].append((atom, atom.occ, atom.altloc))
        else:
            # Keep regular atoms (no alt-loc)
            atom_groups[atom_name].append((atom, atom.occ, "\x00"))

    # leave the atom with the highest occupancy in each group
    atoms_to_remove = []
    for _, atom_list in atom_groups.items():
        if len(atom_list) > 1:
            # Find the atom with the highest occupancy
            best_atom = None
            best_occ = -1.0
            best_altloc = None

            for atom, occ, altloc in sorted(
                atom_list, key=lambda x: (x[1], x[2]), reverse=True
            ):
                if occ > best_occ or (
                    occ == best_occ and (best_altloc is None or altloc < best_altloc)
                ):
                    best_occ = occ
                    best_atom = atom
                    best_altloc = altloc

            # add other atoms to removal list
            for atom, _, altloc in atom_list:
                if atom != best_atom:
                    atoms_to_remove.append((atom.name, altloc))

    # remove the atoms
    for atom_name, altloc in atoms_to_remove:
        residue.remove_atom(atom_name, altloc)
    # clear altloc for selected atoms
    for atom in residue:
        if atom.has_altloc():
            atom.altloc = "\x00"


def resolve_mmcif_altlocs(chain: gemmi.Chain) -> None:
    """
    Resolve alternative locations in mmCIF data for a given residue.

    Selection criteria:
    1. If the entire residue has alternative locations:
       - Calculate the average occupancy for each residue type
       - Select the residue type with the highest occupancy
       - Select the alternative location group with the highest occupancy within that residue type
    2. If only some atoms have alternative locations:
       - Select the atom with the highest occupancy for each atom name

    Args:
        residue: gemmi.Residue object, residue to be processed
    """
    for group in chain.whole().residue_groups():
        if len(group) > 1:
            _process_altloc_groups_whole(group)
        else:
            _process_altloc_group_partial(group)


def mse_to_met(residue: gemmi.Residue) -> None:
    """
    fix MSE residue to MET by changing atom names and element types.

    Args:
        residue: gemmi.Residue object, residue to be fixed
    """
    if residue.name != "MSE":
        return
    else:
        residue.name = "MET"
        for atom in residue:
            if atom.name == "SE":
                atom.name = "SD"
                atom.element = gemmi.Element("S")


def fix_structure(
    struc: gemmi.Structure,
    fix_mse_residues: bool = True,
    fix_arginine: bool = False,
    include_water: bool = False,
) -> None:
    """
    fix residues in mmCIF data.
    1. Treat deuterium as fraction of hydrogen (Deuterium is the same as hydrogen)
    2. MSE -> MET conversion (fix_mse_residues=True)
    3. fix arginine residues with swapped NH1 and NH2 atom names (fix_arginine=True)
    4. resolve alternative locations

    Args:
        struc: gemmi.Structure object, mmCIF data to be fixed
        fix_mse_residues: bool, whether to convert MSE residues to MET.
        fix_arginine: bool, whether to fix arginine residues with swapped NH1 and NH2 atom names
    """
    # treat deuterium as fraction of hydrogen
    struc.store_deuterium_as_fraction(True)
    for model in struc:
        for chain in model:
            resolve_mmcif_altlocs(chain)
            for residue in chain:
                if fix_mse_residues:
                    mse_to_met(residue)
                if fix_arginine:
                    fix_arginine_residues(residue)
