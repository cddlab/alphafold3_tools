import gemmi


def atom_equiv(lhs: str, rhs: str) -> bool:
    """
    Check if two atom names are equivalent.
    Deuterium is the same atom as Hydrogen so keep equivalent for grouping.

    Args:
        lhs: Left atom name
        rhs: Right atom name

    Returns:
        True if equivalent
    """
    if lhs == rhs:
        return True
    if not lhs or not rhs:
        return False

    # Check equivalence of H and D
    if (lhs[0] == "H" and rhs[0] == "D") or (lhs[0] == "D" and rhs[0] == "H"):
        return lhs[1:] == rhs[1:]

    return False


def group_by(
    values: list[str],
    start: int,
    count: int,
    group_callback: Callable[[int, int], None],
    is_equal: Optional[Callable[[str, str], bool]] = None,
) -> None:
    """
    Group consecutive equivalent values and call a callback for each group

    Args:
        values: List of values
        start: Start index
        count: Number of elements to process
        group_callback: Function called for each group (start, count)
        is_equal: Equality check function (if None, == is used)
    """
    if is_equal is None:
        is_equal = lambda a, b: a == b

    if count == 0:
        return

    span_start = start
    for i in range(start + 1, start + count):
        if not is_equal(values[i], values[span_start]):
            group_callback(span_start, i - span_start)
            span_start = i

    group_callback(span_start, start + count - span_start)


def process_alt_loc_groups_whole(
    alt_loc_start: int,
    alt_loc_count: int,
    comp_ids: list[str],
    atom_ids: list[str],
    alt_ids: list[str],
    occupancies: list[str],
    keep_indices: list[int],
) -> None:
    """
    Processing when the entire residue has alternative locations

    The residue type with the highest occupancy and the alternative location group
    with the highest occupancy within it are selected.

    Args:
        alt_loc_start: Start index of the alternative location group
        alt_loc_count: Number of atoms in the group
        comp_ids: List of residue names
        atom_ids: List of atom names
        alt_ids: List of alternative location IDs
        occupancies: List of occupancies
        keep_indices: List of atom indices to keep (output)
    """
    best_split = (alt_loc_start, alt_loc_count)
    best_occupancy = float("-inf")
    best_group = alt_ids[alt_loc_start][0]

    # 残基タイプごとにグループ化
    def process_residue_group(start: int, count: int):
        nonlocal best_split, best_occupancy, best_group

        # 代替位置グループごとの統計を計算
        alt_loc_groups = []
        occupancy_stats = []  # [(count, total_occupancy), ...]

        for i in range(count):
            alt_loc_id = alt_ids[start + i][0]
            occupancy = occupancy_to_float(occupancies[start + i])

            if alt_loc_id in alt_loc_groups:
                idx = alt_loc_groups.index(alt_loc_id)
                occupancy_stats[idx] = (
                    occupancy_stats[idx][0] + 1,
                    occupancy_stats[idx][1] + occupancy,
                )
            else:
                alt_loc_groups.append(alt_loc_id)
                occupancy_stats.append((1, occupancy))

        # 平均占有率の合計を計算
        total_occupancy = sum(total_occ / cnt for cnt, total_occ in occupancy_stats)

        group = min(alt_loc_groups)

        # より良い残基タイプか判定
        if total_occupancy > best_occupancy or (
            total_occupancy == best_occupancy and group < best_group
        ):
            # 最良のサブグループを選択
            best_sub_group = alt_loc_groups[0]
            best_amount = occupancy_stats[0][1] / occupancy_stats[0][0]

            for i in range(1, len(occupancy_stats)):
                amount = occupancy_stats[i][1] / occupancy_stats[i][0]
                group_char = alt_loc_groups[i]

                if amount > best_amount or (
                    amount == best_amount and group_char < best_sub_group
                ):
                    best_amount = amount
                    best_sub_group = group_char

            best_occupancy = total_occupancy
            best_group = best_sub_group
            best_split = (start, count)

    group_by(comp_ids, alt_loc_start, alt_loc_count, process_residue_group)

    # Add atoms of the selected residue type
    split_start, split_count = best_split

    def add_best_atom(start: int, count: int):
        # Find atoms with the selected alternative location ID
        best_index = start
        for i in range(1, count):
            if alt_ids[start + i][0] == best_group:
                best_index = start + i
                break
        keep_indices.append(best_index)

    group_by(atom_ids, split_start, split_count, add_best_atom, atom_equiv)


def process_alt_loc_group_partial(
    alt_loc_start: int,
    alt_loc_count: int,
    atom_ids: list[str],
    alt_ids: list[str],
    occupancies: list[str],
    keep_indices: list[int],
) -> None:
    """
    Processing when the residue has partial alternative locations

    The atom with the highest occupancy is selected for each atom group.

    Args:
        alt_loc_start: Start index of the alternative location group
        alt_loc_count: Number of atoms in the group
        atom_ids: List of atom names
        alt_ids: List of alternative location IDs
        occupancies: List of occupancies
        keep_indices: List of atom indices to keep (output)
    """

    def process_atom_group(start: int, count: int):
        if count == 1:
            keep_indices.append(start)
        else:
            # 最も占有率の高い原子を選択
            best_occ = occupancy_to_float(occupancies[start])
            best_index = start
            best_group = alt_ids[start][0]

            for i in range(count):
                occ = occupancy_to_float(occupancies[start + i])
                group = alt_ids[start + i][0]

                if occ > best_occ or (occ == best_occ and group < best_group):
                    best_group = group
                    best_index = start + i
                    best_occ = occ

            keep_indices.append(best_index)

    group_by(atom_ids, alt_loc_start, alt_loc_count, process_atom_group, atom_equiv)


def resolve_mmcif_altlocs(
    layout: MmcifLayout,
    comp_ids: list[str],
    atom_ids: list[str],
    alt_ids: list[str],
    occupancies: list[str],
    chain_indices: list[int],
) -> list[int]:
    """
    Resolve alternative locations (alt-loc) in mmCIF and return the indices of atoms to keep

    An altloc is a mechanism to record multiple conformations of the same residue
    in a protein structure. This function selects the most appropriate conformation.

    Selection criteria:
    1. When the entire residue has alternative locations:
       - Calculate the average occupancy for each residue type
       - Select the residue type with the highest occupancy
       - Select the alternative location group with the highest occupancy within that residue type

    2. When the residue has partial alternative locations:
       - Select the atom with the highest occupancy for each atom group

    Args:
        layout: mmCIF layout information
        comp_ids: List of residue names (_atom_site.label_comp_id)
        atom_ids: List of atom names (_atom_site.label_atom_id)
        alt_ids: List of alternative location IDs (_atom_site.label_alt_id)
        occupancies: List of occupancies (_atom_site.occupancy)
        chain_indices: List of chain indices to process

    Returns:
        List of indices of atoms to keep
    """
    keep_indices = []
    alt_loc_start = 0

    # Process each chain
    for chain_index in chain_indices:
        residues_start, residues_end = layout.residue_range(chain_index)

        # Process each residue
        for residue in range(residues_start, residues_end):
            alt_loc_count = 0
            atom_start, atom_end = layout.atom_range(residue)

            # Process each atom
            for i in range(atom_start, atom_end):
                alt_loc_id = alt_ids[i][0] if alt_ids[i] else "."

                # No alternative location ('.' or '?')
                if alt_loc_id == "." or alt_loc_id == "?":
                    # Process previous alternative location group if any
                    if alt_loc_count > 0:
                        process_alt_loc_group_partial(
                            alt_loc_start,
                            alt_loc_count,
                            atom_ids,
                            alt_ids,
                            occupancies,
                            keep_indices,
                        )
                        alt_loc_count = 0

                    # Keep regular atoms
                    keep_indices.append(i)

                # Alternative location present (A, B, C, etc.)
                else:
                    if alt_loc_count == 0:
                        alt_loc_start = i
                    alt_loc_count += 1

            # Process previous alternative location group if any
            if alt_loc_count > 0:
                # When the entire residue has alternative locations
                if atom_end - atom_start == alt_loc_count:
                    process_alt_loc_groups_whole(
                        alt_loc_start,
                        alt_loc_count,
                        comp_ids,
                        atom_ids,
                        alt_ids,
                        occupancies,
                        keep_indices,
                    )
                # When the residue has partial alternative locations
                else:
                    process_alt_loc_group_partial(
                        alt_loc_start,
                        alt_loc_count,
                        atom_ids,
                        alt_ids,
                        occupancies,
                        keep_indices,
                    )

    return keep_indices


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


def fix_struc(
    struc: gemmi.Structure,
    fix_arginine: bool = False,
) -> None:
    """
    fix residues in mmCIF data.

    Args:
        struc: gemmi.Structure object, mmCIF data to be fixed
    """
    for model in struc:
        for chain in model:
            for residue in chain:
                if fix_arginine:
                    fix_arginine_residues(residue)
