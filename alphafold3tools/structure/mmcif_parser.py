# %%
from typing import Mapping

import gemmi
import pandas as pd
from loguru import logger

import alphafold3tools.structure.mmcif_utils as mmcif_utils
import alphafold3tools.structure_stores as structure_stores


def _sort_dict_by_keys(d):
    return {key: d[key] for key in sorted(d)}


def _get_entity_id_for_chain(df_entity_polys, target_chain_id: str) -> str:
    """Get entity_id for a given chain ID from entity_poly DataFrame (vectorized)."""
    mask = (
        df_entity_polys["pdbx_strand_id"]
        .str.split(",")
        .apply(lambda chains: target_chain_id in chains)
    )
    matching_rows = df_entity_polys[mask]

    if len(matching_rows) == 0:
        ValueError(f"No entity found for chain ID: {target_chain_id}")
    return matching_rows.iloc[0]["entity_id"]


def _get_mmcif_category_as_df(
    block: gemmi.cif.Block, category_name: str
) -> pd.DataFrame:
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
    number of rows as the DataFrame. Existing columns are returned as-is and may
    include missing values.
    """
    if column in df.columns:
        return df[column].to_list()
    return [default_value] * len(df)


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
    df["B_iso_or_equiv"] = (
        df["B_iso_or_equiv"].astype(float).round(2).map(lambda x: f"{x:.2f}")
    )

    # Convert DataFrame to dict format for gemmi
    return {
        "group_PDB": df["group_PDB"].to_list(),
        "id": df["id"].to_list(),
        "type_symbol": df["type_symbol"].to_list(),
        "label_atom_id": df["label_atom_id"].to_list(),
        "label_alt_id": df["label_alt_id"].to_list(),
        "label_comp_id": df["label_comp_id"].to_list(),
        "label_asym_id": df["label_asym_id"].to_list(),
        "label_entity_id": df["label_entity_id"].to_list(),
        "label_seq_id": df["label_seq_id"].to_list(),
        "pdbx_PDB_ins_code": df["pdbx_PDB_ins_code"].to_list(),
        "Cartn_x": df["Cartn_x"].to_list(),
        "Cartn_y": df["Cartn_y"].to_list(),
        "Cartn_z": df["Cartn_z"].to_list(),
        "occupancy": df["occupancy"].to_list(),
        "B_iso_or_equiv": df["B_iso_or_equiv"].to_list(),
        "auth_seq_id": df["auth_seq_id"].to_list(),
        "auth_asym_id": df["auth_asym_id"].to_list(),
        "pdbx_PDB_model_num": df["pdbx_PDB_model_num"].to_list(),
    }


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
        subchain
        for entity in struc.entities
        if entity.polymer_type in allowed_types
        for subchain in entity.subchains
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
      include_water: If True, water (HOH) molecules will be parsed. Water
        molecules may be grouped into chains, where number of residues > 1. Water
        molecules are usually grouped into chains but do not necessarily all share
        the same chain ID.
      include_other: If True, all other atoms that are not included by any of the
        above parameters will be included. This covers e.g. "polypeptide(D)" and
        "macrolide" entities, as well as all other non-standard types.
    Returns:
      gemmi.cif.Block: New mmCIF block for the specified chain.
    """
    logger.info(f"Processing {pdbid} chain {auth_asym_id}...")
    cif_str = structure_store.get_mmcif_str(pdbid)
    block = gemmi.cif.read_string(cif_str)[0]
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
        "id": target_entity["id"].values[0],
        "pdbx_description": target_entity["pdbx_description"].values[0],
        "type": target_entity["type"].values[0],
    }
    target_poly = _filter_df_by_column(df_entity_polys, "entity_id", target_entity_id)
    poly_dict = {
        "entity_id": target_poly["entity_id"].values[0],
        "pdbx_strand_id": auth_asym_id,
        "type": target_poly["type"].values[0],
    }

    df_entity_poly_seqs = _get_mmcif_category_as_df(block, "_entity_poly_seq")
    target_poly_seqs = _filter_df_by_column(
        df_entity_poly_seqs, "entity_id", target_entity_id
    ).copy()
    if fix_mse_residues:
        # MSE -> MET
        target_poly_seqs.loc[target_poly_seqs["mon_id"] == "MSE", "mon_id"] = "MET"
    target_poly_seqs.loc[:, "hetero"] = "n"  # always set hetero to "n"

    polyseq_loop_dict = {
        "entity_id": target_poly_seqs["entity_id"].to_list(),
        "hetero": target_poly_seqs["hetero"].to_list(),
        "mon_id": target_poly_seqs["mon_id"].to_list(),
        "num": target_poly_seqs["num"].to_list(),
    }
    # _exptl.method
    if block.find_value("_exptl.method") is None:
        exptl_method = "unknown"
    else:
        exptl_method = block.find_value("_exptl.method")
    # _pdbx_audit_revision_history.revision_date: only the first row
    df_revision_histories = _get_mmcif_category_as_df(
        block, "_pdbx_audit_revision_history"
    )
    first_revision = df_revision_histories.iloc[0]
    first_revision_date = first_revision["revision_date"]
    # _pdbx_database_status.recvd_initial_deposition_date
    initial_deposition_date = block.find_value(
        "_pdbx_database_status.recvd_initial_deposition_date"
    )

    # _pdbx_poly_seq_scheme loop
    df_pdbx_poly_seq_scheme = _get_mmcif_category_as_df(block, "_pdbx_poly_seq_scheme")
    # filter by target_entity_id and target_chain_id
    target_pdbx_poly_seq_scheme = _filter_df_by_columns(
        df_pdbx_poly_seq_scheme,
        {"entity_id": target_entity_id, "pdb_strand_id": target_chain_id},
    )
    if fix_mse_residues:
        # MSE -> MET
        target_pdbx_poly_seq_scheme.loc[
            target_pdbx_poly_seq_scheme["mon_id"] == "MSE", "mon_id"
        ] = "MET"
    target_pdbx_poly_seq_scheme.loc[:, "hetero"] = "n"  # always set hetero to "n"
    pdbx_polyseq_loop_dict = {
        "asym_id": target_pdbx_poly_seq_scheme["asym_id"].to_list(),
        "auth_seq_num": target_pdbx_poly_seq_scheme["auth_seq_num"].to_list(),
        "entity_id": target_pdbx_poly_seq_scheme["entity_id"].to_list(),
        "hetero": target_pdbx_poly_seq_scheme["hetero"].to_list(),
        "mon_id": target_pdbx_poly_seq_scheme["mon_id"].to_list(),
        "pdb_ins_code": target_pdbx_poly_seq_scheme["pdb_ins_code"].to_list(),
        "pdb_seq_num": target_pdbx_poly_seq_scheme["pdb_seq_num"].to_list(),
        "pdb_strand_id": target_pdbx_poly_seq_scheme["pdb_strand_id"].to_list(),
        "seq_id": target_pdbx_poly_seq_scheme["seq_id"].to_list(),
    }
    # _pdbx_struct_assembly loop
    df_pdbx_struct_assembly = _get_mmcif_category_as_df(block, "_pdbx_struct_assembly")
    pdbx_struct_assembly_dict = {
        "details": _safe_to_list(df_pdbx_struct_assembly, "details"),
        "id": _safe_to_list(df_pdbx_struct_assembly, "id"),
        "method_details": _safe_to_list(df_pdbx_struct_assembly, "method_details"),
        "oligomeric_count": _safe_to_list(df_pdbx_struct_assembly, "oligomeric_count"),
        "oligomeric_details": _safe_to_list(
            df_pdbx_struct_assembly, "oligomeric_details"
        ),
    }
    # _pdbx_struct_assembly_gen
    df_pdbx_struct_assembly_gen = _get_mmcif_category_as_df(
        block, "_pdbx_struct_assembly_gen"
    )
    pdbx_structure_assembly_gen_dict = {
        "assembly_id": _safe_to_list(df_pdbx_struct_assembly_gen, "assembly_id"),
        "asym_id_list": _safe_to_list(df_pdbx_struct_assembly_gen, "asym_id_list"),
        "oper_expression": _safe_to_list(
            df_pdbx_struct_assembly_gen, "oper_expression"
        ),
    }
    # _pdbx_struct_oper_list
    pdbx_struct_oper_list = _sort_dict_by_keys(
        block.get_mmcif_category("_pdbx_struct_oper_list")
    )
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
        "entity_id": target_struct_asym["entity_id"].values[0],
        "id": target_struct_asym["id"].values[0],
    }
    # _atom_site
    # Use gemmi.Structure to modify structural information
    gemmi_structure = gemmi.make_structure_from_block(block)
    mmcif_utils.fix_structure(
        gemmi_structure,
        fix_mse_residues=fix_mse_residues,
        fix_arginine=fix_arginines,
        include_water=include_water,
    )
    gemmi_structure.update_mmcif_block(block)

    df_atom_site = _get_mmcif_category_as_df(block, "_atom_site")
    target_atom_site_df = _filter_df_by_column(
        df_atom_site, "label_asym_id", target_chain_id
    ).copy()
    target_atom_site_dict = format_atom_site_dict(target_atom_site_df)

    newdoc = gemmi.cif.Document()
    newblock = newdoc.add_new_block(block.name)
    newblock.set_pair("_entry.id", entry_id)
    newblock.set_mmcif_category("_chem_comp", sorted_chem_comps)
    newblock.set_pairs("_entity.", entity_dict)
    newblock.set_pairs("_entity_poly.", poly_dict)
    newblock.set_mmcif_category("_entity_poly_seq", polyseq_loop_dict)
    newblock.set_pair("_exptl.method", exptl_method)
    newblock.set_pair("_pdbx_audit_revision_history.revision_date", first_revision_date)
    newblock.set_pair(
        "_pdbx_database_status.recvd_initial_deposition_date", initial_deposition_date
    )
    newblock.set_mmcif_category("_pdbx_poly_seq_scheme", pdbx_polyseq_loop_dict)
    newblock.set_mmcif_category("_pdbx_struct_assembly.", pdbx_struct_assembly_dict)
    newblock.set_mmcif_category(
        "_pdbx_struct_assembly_gen.", pdbx_structure_assembly_gen_dict
    )
    newblock.set_mmcif_category("_pdbx_struct_oper_list", pdbx_struct_oper_list)
    if refine_ls_d_res_high is not None:
        newblock.set_pair("_refine.ls_d_res_high", refine_ls_d_res_high)
    newblock.set_pairs("_software.", software_dict)
    newblock.set_pairs("_struct_asym.", struct_asym_dict)
    newblock.set_mmcif_category("_atom_site", target_atom_site_dict)
    return newblock
