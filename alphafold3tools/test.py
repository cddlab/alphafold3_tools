# %%
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


def resolve_mmcif_altlocs(
    struc: gemmi.Structure,
) -> None:
    """
    mmCIFの代替位置（alt-loc）を解決し、保持する原子のインデックスを返す

    代替位置とは、タンパク質構造で同じ残基が複数の配座を取る場合に
    各配座を記録する仕組みです。この関数は最も適切な配座を選択します。

    選択基準:
    1. 残基全体が代替位置を持つ場合:
       - 残基タイプごとに平均占有率を計算
       - 最も高い占有率の残基タイプを選択
       - その中で最も占有率の高い代替位置グループを選択

    2. 部分的に代替位置を持つ場合:
       - 各原子ごとに最も占有率の高いものを選択

    Args:
        struc: gemmi.Structureオブジェクト、mmCIFデータ
    """
    # 各チェインについて処理
    for chain_index in chain_indices:
        residues_start, residues_end = layout.residue_range(chain_index)

        # 各残基について処理
        for residue in range(residues_start, residues_end):
            alt_loc_count = 0
            atom_start, atom_end = layout.atom_range(residue)

            # 各原子について処理
            for i in range(atom_start, atom_end):
                alt_loc_id = alt_ids[i][0] if alt_ids[i] else "."

                # 代替位置なし（'.'または'?'）
                if alt_loc_id == "." or alt_loc_id == "?":
                    # 前に代替位置グループがあれば処理
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

                    # 通常の原子を保持
                    keep_indices.append(i)

                # 代替位置あり（A, B, Cなど）
                else:
                    if alt_loc_count == 0:
                        alt_loc_start = i
                    alt_loc_count += 1

            # 残基の最後に代替位置グループがあれば処理
            if alt_loc_count > 0:
                # 残基全体が代替位置を持つ場合
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
                # 部分的に代替位置を持つ場合
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


# correct NH1 and NH2 atom names in arginine residues
ciffile = "/Users/YoshitakaM/Desktop/mmcif_files/6W81.cif"
struc = gemmi.read_structure(ciffile)
fix_struc(struc, fix_arginine=True)

print(struc.make_mmcif_block().as_string())

# %%
for model in struc:
    for chain in model:
        for residue in chain:
            for atom in residue:
                if atom.has_altloc():
                    print(
                        f"Residue: {residue.name}-{residue.seqid.num} Atom: {atom.name} has altloc {atom.altloc}"
                    )


# %%
