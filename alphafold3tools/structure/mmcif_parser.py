# %%
import math
from typing import Any, Mapping

import gemmi
import pandas as pd
from loguru import logger

import alphafold3tools.structure.mmcif_utils as mmcif_utils
import alphafold3tools.structure_stores as structure_stores


def _sort_dict_by_keys(d):
    return {key: d[key] for key in sorted(d)}


def _get_entity_id_for_chain(df_entity_polys, target_chain_id: str) -> str:
    """Get entity_id for a given chain ID from entity_poly DataFrame (vectorized)."""
    mask = df_entity_polys["pdbx_strand_id"].str.split(",").apply(lambda chains: target_chain_id in chains)
    matching_rows = df_entity_polys[mask]

    if len(matching_rows) == 0:
        raise ValueError(f"No entity found for chain ID: {target_chain_id}")
    return matching_rows.iloc[0]["entity_id"]


def _is_missing(value: Any) -> bool:
    """Whether ``value`` is a pandas missing value (None, NaN or pd.NA)."""
    return value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value))


def _column_to_cif_list(df: pd.DataFrame, column: str) -> list:
    """Converts a DataFrame column to a list of values for gemmi to write.

    gemmi reads the mmCIF null "?" as None and "." as False. pandas turns None
    into NaN in string columns, which gemmi would write as "nan", so missing
    values are turned back into None ("?"). False is kept as it is (".").
    """
    return [None if _is_missing(value) else value for value in df[column].to_list()]


def _select_heterogeneous_residues(
    df: pd.DataFrame, seq_id_column: str, resolved_names: Mapping[str, str]
) -> pd.DataFrame:
    """Keeps one row per residue of a polymer sequence table.

    With microheterogeneity, ``_entity_poly_seq`` and ``_pdbx_poly_seq_scheme`` list
    several residue names (mon_id) at the same residue number. As AlphaFold3 does,
    keep the residue that is left in ``_atom_site`` after the alt-loc resolution, or
    the first one for unresolved residues.

    Args:
        df: Rows of ``_entity_poly_seq`` or ``_pdbx_poly_seq_scheme`` of one chain.
        seq_id_column: Column of the residue number ("num" or "seq_id").
        resolved_names: Residue name in ``_atom_site`` by ``label_seq_id``.

    Returns:
        The rows of ``df`` with a single residue name per residue number.
    """
    positions_by_seq_id: dict[str, list[int]] = {}
    for position, seq_id in enumerate(df[seq_id_column].to_list()):
        positions_by_seq_id.setdefault(seq_id, []).append(position)
    mon_ids = df["mon_id"].to_list()
    selected = []
    for seq_id, positions in positions_by_seq_id.items():
        matching = [position for position in positions if mon_ids[position] == resolved_names.get(seq_id)]
        selected.append(matching[0] if matching else positions[0])
    return df.iloc[sorted(selected)]


def _get_mmcif_category_as_df(block: gemmi.cif.Block, category_name: str) -> pd.DataFrame:
    """Get mmCIF category as pandas DataFrame.

    Args:
        block: gemmi.cif.Block object.
        category_name: Category name (e.g., "_entity", "_entity_poly_seq").

    Returns:
        DataFrame containing the category data.
    """
    category_data = block.get_mmcif_category(category_name)
    return pd.DataFrame(category_data)


def _filter_df_by_column(df: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    """Filter DataFrame by a single column value.

    Args:
        df: Input DataFrame.
        column: Column name to filter by.
        value: Value to match.

    Returns:
        Filtered DataFrame.

    Example:
        >>> filtered = _filter_df_by_column(df_entities, "id", "1")
    """
    return df.loc[df[column] == value]


def _filter_df_by_columns(df: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    """Filter DataFrame by multiple column values.

    Args:
        df: Input DataFrame.
        filters: Dictionary mapping column names to target values.

    Returns:
        Filtered DataFrame.

    Example:
        >>> filtered = _filter_df_by_columns(
        ...     df, {"entity_id": "1", "pdb_strand_id": "A"}
        ... )
    """
    result = df
    for column, value in filters.items():
        result = result.loc[result[column] == value]
    return result


def _safe_to_list(
    df: pd.DataFrame,
    column: str,
    default_value: str = "",
) -> list:
    """Safely convert a DataFrame column to list.

    If the column does not exist, return a list of default values with the same
    number of rows as the DataFrame. Missing values of existing columns are
    returned as None, which gemmi writes as "?".
    """
    if column in df.columns:
        return _column_to_cif_list(df, column)
    return [default_value] * len(df)


_ATOM_SITE_COLUMNS = (
    "group_PDB",
    "id",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "pdbx_PDB_ins_code",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "occupancy",
    "B_iso_or_equiv",
    "auth_seq_id",
    "auth_asym_id",
    "pdbx_PDB_model_num",
)


def format_atom_site_dict(target_atom_site_df: pd.DataFrame) -> dict:
    """Format atom site DataFrame and convert to dictionary.

    Args:
        target_atom_site_df: DataFrame containing atom site information.

    Returns:
        Dictionary formatted for gemmi mmCIF output.
    """
    # Make a copy to avoid modifying the original DataFrame
    df = target_atom_site_df.copy()

    ## group_PDB should be "ATOM", not "HETATM"
    df.loc[df["group_PDB"] == "HETATM", "group_PDB"] = "ATOM"

    # Cartn_x, Cartn_y, Cartn_z should have 3 decimal places and 0 padding
    df["Cartn_x"] = df["Cartn_x"].astype(float).round(3).map(lambda x: f"{x:.3f}")
    df["Cartn_y"] = df["Cartn_y"].astype(float).round(3).map(lambda x: f"{x:.3f}")
    df["Cartn_z"] = df["Cartn_z"].astype(float).round(3).map(lambda x: f"{x:.3f}")

    # occupancy should have 2 decimal places
    df["occupancy"] = df["occupancy"].astype(float).round(2).map(lambda x: f"{x:.2f}")

    # B_iso_or_equiv should have 2 decimal places
    df["B_iso_or_equiv"] = df["B_iso_or_equiv"].astype(float).round(2).map(lambda x: f"{x:.2f}")

    # Convert DataFrame to dict format for gemmi
    return {column: _column_to_cif_list(df, column) for column in _ATOM_SITE_COLUMNS}


def polymer_auth_asym_id_to_label_asym_id(
    block: gemmi.cif.Block,
    protein: bool = True,
    rna: bool = True,
    dna: bool = True,
    other: bool = True,
) -> Mapping[str, str]:
    """Mapping from author chain ID to internal chain ID, polymers only.

    This mapping is well defined only for polymers (protein, DNA, RNA), but not
    for ligands or water.

    E.g. if a structure had the following internal chain IDs (label_asym_id):
        A (protein), B (DNA), C (ligand bound to A), D (ligand bound to A),
        E (ligand bound to B).

    Such structure would have this internal chain ID (label_asym_id) -> author
    chain ID (auth_asym_id) mapping:
        A -> A, B -> B, C -> A, D -> A, E -> B

    This is a bijection only for polymers (A, B), but not for ligands.

    Args:
        protein: Whether to include protein (polypeptide(L)) chains.
        rna: Whether to include RNA chains.
        dna: Whether to include DNA chains.
        other: Whether to include other polymer chains, e.g. RNA/DNA hybrid or
        polypeptide(D). Note that include_other=True must be set in from_mmcif.

    Returns:
        A mapping from author chain ID to the internal (label) chain ID for the
        given polymer types in the Structure, ligands/water are ignored.

    Raises:
        ValueError: If the mapping from internal chain IDs to author chain IDs is
        not a bijection for polymer chains.
    """
    allowed_types = set()
    if protein:
        allowed_types.add(gemmi.PolymerType.PeptideL)
    if rna:
        allowed_types.add(gemmi.PolymerType.Rna)
    if dna:
        allowed_types.add(gemmi.PolymerType.Dna)
    if other:
        allowed_types |= {
            gemmi.PolymerType.PeptideD,
            gemmi.PolymerType.DnaRnaHybrid,
            gemmi.PolymerType.CyclicPseudoPeptide,
        }

    auth_asym_id_to_label_asym_id = {}
    struc = gemmi.make_structure_from_block(block)
    model = struc[0]
    polymer_label_asym_ids = [
        subchain for entity in struc.entities if entity.polymer_type in allowed_types for subchain in entity.subchains
    ]
    for chain in model:
        for subchain in chain.subchains():
            auth_asym_id = chain.name
            label_asym_id = subchain.subchain_id()
            if label_asym_id not in polymer_label_asym_ids:
                continue
            # The mapping from author chain id to label chain id is only one-to-one if
            # we restrict our attention to polymers. But check nevertheless.
            if auth_asym_id in auth_asym_id_to_label_asym_id:
                raise ValueError(
                    f'Author chain ID "{auth_asym_id}" does not have a unique mapping '
                    f'to internal chain ID "{label_asym_id}", it is already mapped to '
                    f'"{auth_asym_id_to_label_asym_id[auth_asym_id]}".'
                )
            auth_asym_id_to_label_asym_id[auth_asym_id] = label_asym_id
    return auth_asym_id_to_label_asym_id


def _normalize_polymer_residues(df: pd.DataFrame, fix_mse_residues: bool) -> pd.DataFrame:
    """Returns a copy of polymer sequence rows with MSE -> MET and hetero = "n"."""
    df = df.copy()
    if fix_mse_residues:
        df.loc[df["mon_id"] == "MSE", "mon_id"] = "MET"
    df.loc[:, "hetero"] = "n"  # A template has a single residue per residue number.
    return df


def _fixed_chain_atom_site(
    block: gemmi.cif.Block,
    label_asym_id: str,
    *,
    fix_mse_residues: bool,
    fix_arginines: bool,
    include_water: bool,
) -> pd.DataFrame:
    """Returns the fixed ``_atom_site`` rows of one chain in the first model.

    The structure in ``block`` is fixed (water, alt-locs, MSE, arginines) and
    written back to ``block``. Only the first model is kept, as AlphaFold3 reads
    the first model of a template (e.g. of an NMR entry).
    """
    gemmi_structure = gemmi.make_structure_from_block(block)
    mmcif_utils.fix_structure(
        gemmi_structure,
        fix_mse_residues=fix_mse_residues,
        fix_arginine=fix_arginines,
        include_water=include_water,
    )
    gemmi_structure.update_mmcif_block(block)

    df_atom_site = _get_mmcif_category_as_df(block, "_atom_site")
    df_chain = _filter_df_by_column(df_atom_site, "label_asym_id", label_asym_id)
    if df_chain.empty:
        return df_chain.copy()
    first_model = df_chain["pdbx_PDB_model_num"].iloc[0]
    return _filter_df_by_column(df_chain, "pdbx_PDB_model_num", first_model).copy()


def mmcifcontent(
    pdbid: str,
    auth_asym_id: str,
    structure_store: structure_stores.StructureStore,
    fix_mse_residues: bool = True,
    fix_arginines: bool = True,
    include_water: bool = False,
    include_bonds: bool = False,
    include_other: bool = True,
) -> gemmi.cif.Block:
    """Write a new mmCIF file for the specified chain from the original mmCIF file.

    Args:
      pdbid: PDB ID of the structure.
      auth_asym_id: Chain ID to extract from the mmCIF file.
      structurestore: StructureStore instance to retrieve mmCIF data.
      fix_mse_residues: If True, selenium atom sites (SE) in selenomethionine
        (MSE) residues will be changed to sulfur atom sites (SD). This is because
        methionine (MET) residues are often replaced with MSE to aid X-Ray
        crystallography. If False, the SE MSE atom sites won't be modified.
      fix_arginines: If True, NH1 and NH2 in arginine will be swapped if needed so
        that NH1 is always closer to CD than NH2. If False, no atom sites in
        arginine will be touched. Note that HH11, HH12, HH21, HH22 are fixed too.
      fix_unknown_dna: If True, residues with name N in DNA chains will have their
        res_name replaced with DN. Atoms are not changed.
      include_water: If True, water molecules are kept when the structure is
        fixed. The template contains only the polymer chain, so water is never
        written, as in AlphaFold3.
      include_other: If True, all other atoms that are not included by any of the
        above parameters will be included. This covers e.g. "polypeptide(D)" and
        "macrolide" entities, as well as all other non-standard types.
    Returns:
      gemmi.cif.Block: New mmCIF block for the specified chain.
    Raises:
      ValueError: If a non-water entity has water molecules, which AlphaFold3
        cannot parse, or no entity is found for the chain.
    """
    logger.info(f"Processing {pdbid} chain {auth_asym_id}...")
    cif_str = structure_store.get_mmcif_str(pdbid)
    block = gemmi.cif.read_string(cif_str)[0]
    mmcif_utils.check_no_water_in_non_water_entities(block)
    target_chain_id = polymer_auth_asym_id_to_label_asym_id(
        block,
        protein=True,
        rna=True,
        dna=True,
        other=include_other,
    )[auth_asym_id]
    # entry_id
    entry_id = block.find_value("_entry.id")
    # _chem_comp
    chem_comps = block.get_mmcif_category("_chem_comp")
    sorted_chem_comps = _sort_dict_by_keys(chem_comps)
    # _entity
    df_entities = _get_mmcif_category_as_df(block, "_entity")
    df_entity_polys = _get_mmcif_category_as_df(block, "_entity_poly")
    target_entity_id = _get_entity_id_for_chain(df_entity_polys, auth_asym_id)
    target_entity = _filter_df_by_column(df_entities, "id", target_entity_id)
    entity_dict = {
        "id": _column_to_cif_list(target_entity, "id")[0],
        "pdbx_description": _column_to_cif_list(target_entity, "pdbx_description")[0],
        "type": _column_to_cif_list(target_entity, "type")[0],
    }
    target_poly = _filter_df_by_column(df_entity_polys, "entity_id", target_entity_id)
    poly_dict = {
        "entity_id": _column_to_cif_list(target_poly, "entity_id")[0],
        "pdbx_strand_id": auth_asym_id,
        "type": _column_to_cif_list(target_poly, "type")[0],
    }

    df_entity_poly_seqs = _get_mmcif_category_as_df(block, "_entity_poly_seq")
    target_poly_seqs = _normalize_polymer_residues(
        _filter_df_by_column(df_entity_poly_seqs, "entity_id", target_entity_id),
        fix_mse_residues=fix_mse_residues,
    )
    # _exptl.method
    if block.find_value("_exptl.method") is None:
        exptl_method = "unknown"
    else:
        exptl_method = block.find_value("_exptl.method")
    # _pdbx_audit_revision_history.revision_date: only the first row
    df_revision_histories = _get_mmcif_category_as_df(block, "_pdbx_audit_revision_history")
    first_revision = df_revision_histories.iloc[0]
    first_revision_date = first_revision["revision_date"]
    # _pdbx_database_status.recvd_initial_deposition_date
    initial_deposition_date = block.find_value("_pdbx_database_status.recvd_initial_deposition_date")

    # _pdbx_poly_seq_scheme loop
    df_pdbx_poly_seq_scheme = _get_mmcif_category_as_df(block, "_pdbx_poly_seq_scheme")
    # Filter by the internal (label) chain ID: pdb_strand_id is the author chain
    # ID, which can differ from label_asym_id (e.g. label A is author B in 7YF5).
    target_pdbx_poly_seq_scheme = _normalize_polymer_residues(
        _filter_df_by_columns(
            df_pdbx_poly_seq_scheme,
            {"entity_id": target_entity_id, "asym_id": target_chain_id},
        ),
        fix_mse_residues=fix_mse_residues,
    )
    # _pdbx_struct_assembly loop
    df_pdbx_struct_assembly = _get_mmcif_category_as_df(block, "_pdbx_struct_assembly")
    pdbx_struct_assembly_dict = {
        "details": _safe_to_list(df_pdbx_struct_assembly, "details"),
        "id": _safe_to_list(df_pdbx_struct_assembly, "id"),
        "method_details": _safe_to_list(df_pdbx_struct_assembly, "method_details"),
        "oligomeric_count": _safe_to_list(df_pdbx_struct_assembly, "oligomeric_count"),
        "oligomeric_details": _safe_to_list(df_pdbx_struct_assembly, "oligomeric_details"),
    }
    # _pdbx_struct_assembly_gen
    df_pdbx_struct_assembly_gen = _get_mmcif_category_as_df(block, "_pdbx_struct_assembly_gen")
    pdbx_structure_assembly_gen_dict = {
        "assembly_id": _safe_to_list(df_pdbx_struct_assembly_gen, "assembly_id"),
        "asym_id_list": _safe_to_list(df_pdbx_struct_assembly_gen, "asym_id_list"),
        "oper_expression": _safe_to_list(df_pdbx_struct_assembly_gen, "oper_expression"),
    }
    # _pdbx_struct_oper_list
    pdbx_struct_oper_list = _sort_dict_by_keys(block.get_mmcif_category("_pdbx_struct_oper_list"))
    # _refine.ls_d_res_high
    # This block doesn't exist in Solution NMR. If so, pass None.
    if block.find_value("_refine.ls_d_res_high") is None:
        refine_ls_d_res_high = None
    else:
        refine_ls_d_res_high = block.find_value("_refine.ls_d_res_high")
    # _software category addition
    software_dict = {
        "classification": "other",
        "name": "Gemmi and alphafold3_tools mmcif_parser",
        "pdbx_ordinal": "1",
        "version": "2.0.0",
    }
    # _struct_asym
    df_struct_asym = _get_mmcif_category_as_df(block, "_struct_asym")
    target_struct_asym = _filter_df_by_column(df_struct_asym, "id", target_chain_id)
    struct_asym_dict = {
        "entity_id": _column_to_cif_list(target_struct_asym, "entity_id")[0],
        "id": _column_to_cif_list(target_struct_asym, "id")[0],
    }
    # _atom_site
    target_atom_site_df = _fixed_chain_atom_site(
        block,
        target_chain_id,
        fix_mse_residues=fix_mse_residues,
        fix_arginines=fix_arginines,
        include_water=include_water,
    )
    target_atom_site_dict = format_atom_site_dict(target_atom_site_df)

    # Keep one residue per residue number, consistent with _atom_site.
    resolved_names = dict(
        zip(
            target_atom_site_df["label_seq_id"].to_list(),
            target_atom_site_df["label_comp_id"].to_list(),
            strict=True,
        )
    )
    target_poly_seqs = _select_heterogeneous_residues(target_poly_seqs, "num", resolved_names)
    polyseq_loop_dict = {
        column: _column_to_cif_list(target_poly_seqs, column) for column in ("entity_id", "hetero", "mon_id", "num")
    }
    target_pdbx_poly_seq_scheme = _select_heterogeneous_residues(target_pdbx_poly_seq_scheme, "seq_id", resolved_names)
    pdbx_polyseq_loop_dict = {
        column: _column_to_cif_list(target_pdbx_poly_seq_scheme, column)
        for column in (
            "asym_id",
            "auth_seq_num",
            "entity_id",
            "hetero",
            "mon_id",
            "pdb_ins_code",
            "pdb_seq_num",
            "pdb_strand_id",
            "seq_id",
        )
    }

    newdoc = gemmi.cif.Document()
    newblock = newdoc.add_new_block(block.name)
    newblock.set_pair("_entry.id", entry_id)
    newblock.set_mmcif_category("_chem_comp", sorted_chem_comps)
    newblock.set_pairs("_entity.", entity_dict)
    newblock.set_pairs("_entity_poly.", poly_dict)
    newblock.set_mmcif_category("_entity_poly_seq", polyseq_loop_dict)
    newblock.set_pair("_exptl.method", exptl_method)
    newblock.set_pair("_pdbx_audit_revision_history.revision_date", first_revision_date)
    newblock.set_pair("_pdbx_database_status.recvd_initial_deposition_date", initial_deposition_date)
    newblock.set_mmcif_category("_pdbx_poly_seq_scheme", pdbx_polyseq_loop_dict)
    newblock.set_mmcif_category("_pdbx_struct_assembly.", pdbx_struct_assembly_dict)
    newblock.set_mmcif_category("_pdbx_struct_assembly_gen.", pdbx_structure_assembly_gen_dict)
    newblock.set_mmcif_category("_pdbx_struct_oper_list", pdbx_struct_oper_list)
    if refine_ls_d_res_high is not None:
        newblock.set_pair("_refine.ls_d_res_high", refine_ls_d_res_high)
    newblock.set_pairs("_software.", software_dict)
    newblock.set_pairs("_struct_asym.", struct_asym_dict)
    newblock.set_mmcif_category("_atom_site", target_atom_site_dict)
    return newblock
