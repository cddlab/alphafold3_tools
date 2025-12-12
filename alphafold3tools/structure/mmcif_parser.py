import gemmi
import pandas as pd
from loguru import logger

import alphafold3tools.structure.mmcif_utils as mmcif_utils
import alphafold3tools.structure_stores as structure_stores


def _sort_dict_by_keys(d):
    return {key: d[key] for key in sorted(d)}


def _get_entity_id_for_chain(df_entity_polys, target_chain: str) -> str:
    """Get entity_id for a given chain ID from entity_poly DataFrame (vectorized)."""
    mask = (
        df_entity_polys["pdbx_strand_id"]
        .str.split(",")
        .apply(lambda chains: target_chain in chains)
    )
    matching_rows = df_entity_polys[mask]

    if len(matching_rows) == 0:
        ValueError(f"No entity found for chain ID: {target_chain}")
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


def ciffilecontent(
    pdbid: str,
    structure_store: structure_stores.StructureStore,
    target_chain: str,
    options: gemmi.cif.WriteOptions,
    fix_mse_residues: bool = True,
    fix_arginines: bool = True,
    include_water: bool = False,
    include_bonds: bool = False,
    include_other=True,
) -> str:
    """Write a new mmCIF file for the specified chain from the original mmCIF file.

    Args:
      pdbid: PDB ID of the structure.
      structurestore: StructureStore instance to retrieve mmCIF data.
      target_chain: Target chain ID to extract.
      options: gemmi.cif.WriteOptions for mmCIF writing.
      fix_mse_residues: If True, selenium atom sites (SE) in selenomethionine
        (MSE) residues will be changed to sulphur atom sites (SD). This is because
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
      mmcif string content.
    """
    cif_str = structure_store.get_mmcif_str(pdbid)
    doc = gemmi.cif.read_string(cif_str)
    block = doc.sole_block()

    # _chem_comp
    chem_comps = block.get_mmcif_category("_chem_comp")
    sorted_chem_comps = _sort_dict_by_keys(chem_comps)
    # _entity
    df_entities = _get_mmcif_category_as_df(block, "_entity")
    df_entity_polys = _get_mmcif_category_as_df(block, "_entity_poly")
    target_entity_id = _get_entity_id_for_chain(df_entity_polys, target_chain)
    target_entity = _filter_df_by_column(df_entities, "id", target_entity_id)
    entity_dict = {
        "id": target_entity["id"].values[0],
        "pdbx_description": target_entity["pdbx_description"].values[0],
        "type": target_entity["type"].values[0],
    }
    target_poly = _filter_df_by_column(df_entity_polys, "entity_id", target_entity_id)
    poly_dict = {
        "entity_id": target_poly["entity_id"].values[0],
        "pdbx_strand_id": target_chain,
        "type": target_poly["type"].values[0],
    }

    df_entity_poly_seqs = _get_mmcif_category_as_df(block, "_entity_poly_seq")
    target_poly_seqs = _filter_df_by_column(
        df_entity_poly_seqs, "entity_id", target_entity_id
    )

    target_poly_seqs.loc[:, "hetero"] = "n"  # always set hetero to "n"
    if fix_mse_residues:
        target_poly_seqs.loc[target_poly_seqs["mon_id"] == "MSE", "mon_id"] = "MET"
    polyseq_loop_dict = {
        "entity_id": target_poly_seqs["entity_id"].to_list(),
        "hetero": target_poly_seqs["hetero"].to_list(),
        "mon_id": target_poly_seqs["mon_id"].to_list(),
        "num": target_poly_seqs["num"].to_list(),
    }
    # _exptl.method
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
    # filter by target_entity_id and target_chain
    target_pdbx_poly_seq_scheme = _filter_df_by_columns(
        df_pdbx_poly_seq_scheme,
        {"entity_id": target_entity_id, "pdb_strand_id": target_chain},
    )
    if fix_mse_residues:
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
    # _pdbx_struct_assembly
    df_pdbx_struct_assembly = _get_mmcif_category_as_df(block, "_pdbx_struct_assembly")
    target_pdbx_struct_assembly = _filter_df_by_column(
        df_pdbx_struct_assembly, "id", "1"
    )
    pdbx_struct_assembly_dict = {
        "details": target_pdbx_struct_assembly["details"].values[0],
        "id": target_pdbx_struct_assembly["id"].values[0],
        "method_details": target_pdbx_struct_assembly["method_details"].values[0],
        "oligomeric_count": target_pdbx_struct_assembly["oligomeric_count"].values[0],
        "oligomeric_details": target_pdbx_struct_assembly["oligomeric_details"].values[
            0
        ],
    }
    # _pdbx_struct_assembly_gen
    df_pdbx_struct_assembly_gen = _get_mmcif_category_as_df(
        block, "_pdbx_struct_assembly_gen"
    )
    target_pdbx_struct_assembly_gen = _filter_df_by_column(
        df_pdbx_struct_assembly_gen, "assembly_id", "1"
    )
    pdbx_structure_assembly_gen_dict = {
        "assembly_id": target_pdbx_struct_assembly_gen["assembly_id"].values[0],
        "asym_id_list": target_pdbx_struct_assembly_gen["asym_id_list"].values[0],
        "oper_expression": target_pdbx_struct_assembly_gen["oper_expression"].values[0],
    }
    # _pdbx_struct_oper_list
    pdbx_struct_oper_list = _sort_dict_by_keys(
        block.get_mmcif_category("_pdbx_struct_oper_list")
    )
    # _refine.ls_d_res_high
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
    target_struct_asym = _filter_df_by_column(df_struct_asym, "id", target_chain)
    struct_asym_dict = {
        "entity_id": target_struct_asym["entity_id"].values[0],
        "id": target_struct_asym["id"].values[0],
    }
    # _atom_site
    # Use gemmi.Structure to modify arginine residues if needed
    if fix_arginines:
        gemmi_structure = gemmi.make_structure_from_block(block)
        mmcif_utils.fix_arginine_residues(gemmi_structure)
        # Update block after fixing arginines
        block = gemmi_structure.make_mmcif_block()
    df_atom_site = _get_mmcif_category_as_df(block, "_atom_site")
    target_atom_site_df = _filter_df_by_column(
        df_atom_site, "label_asym_id", target_chain
    )
    ## group_DPB should be "ATOM", not "HETATM"
    target_atom_site_df.loc[
        target_atom_site_df["group_PDB"] == "HETATM", "group_PDB"
    ] = "ATOM"

    if fix_mse_residues:
        target_atom_site_df.loc[
            target_atom_site_df["label_comp_id"] == "MSE", "label_comp_id"
        ] = "MET"
        target_atom_site_df.loc[
            target_atom_site_df["type_symbol"] == "SE", "type_symbol"
        ] = "S"
        target_atom_site_df.loc[
            target_atom_site_df["label_atom_id"] == "SE", "label_atom_id"
        ] = "SD"

    # Convert DataFrame to dict format for gemmi
    target_atom_site_dict = {
        "group_PDB": target_atom_site_df["group_PDB"].to_list(),
        "id": target_atom_site_df["id"].to_list(),
        "type_symbol": target_atom_site_df["type_symbol"].to_list(),
        "label_atom_id": target_atom_site_df["label_atom_id"].to_list(),
        "label_alt_id": target_atom_site_df["label_alt_id"].to_list(),
        "label_comp_id": target_atom_site_df["label_comp_id"].to_list(),
        "label_asym_id": target_atom_site_df["label_asym_id"].to_list(),
        "label_entity_id": target_atom_site_df["label_entity_id"].to_list(),
        "label_seq_id": target_atom_site_df["label_seq_id"].to_list(),
        "pdbx_PDB_ins_code": target_atom_site_df["pdbx_PDB_ins_code"].to_list(),
        "Cartn_x": target_atom_site_df["Cartn_x"].to_list(),
        "Cartn_y": target_atom_site_df["Cartn_y"].to_list(),
        "Cartn_z": target_atom_site_df["Cartn_z"].to_list(),
        "occupancy": target_atom_site_df["occupancy"].to_list(),
        "B_iso_or_equiv": target_atom_site_df["B_iso_or_equiv"].to_list(),
        "auth_seq_id": target_atom_site_df["auth_seq_id"].to_list(),
        "auth_asym_id": target_atom_site_df["auth_asym_id"].to_list(),
        "pdbx_PDB_model_num": target_atom_site_df["pdbx_PDB_model_num"].to_list(),
    }

    newdoc = gemmi.cif.Document()
    newblock = newdoc.add_new_block(block.name)
    newblock.set_mmcif_category("_chem_comp", sorted_chem_comps)
    newblock.set_pairs("_entity.", entity_dict)
    newblock.set_pairs("_entity_poly", poly_dict)
    newblock.set_mmcif_category("_entity_poly_seq", polyseq_loop_dict)
    newblock.set_pair("_exptl.method", exptl_method)
    newblock.set_pair("_pdbx_audit_revision_history.revision_date", first_revision_date)
    newblock.set_pair(
        "_pdbx_database_status.recvd_initial_deposition_date", initial_deposition_date
    )
    newblock.set_mmcif_category("_pdbx_poly_seq_scheme", pdbx_polyseq_loop_dict)
    newblock.set_pairs("_pdbx_struct_assembly.", pdbx_struct_assembly_dict)
    newblock.set_pairs("_pdbx_struct_assembly_gen.", pdbx_structure_assembly_gen_dict)
    newblock.set_mmcif_category("_pdbx_struct_oper_list", pdbx_struct_oper_list)
    newblock.set_pair("_refine.ls_d_res_high", refine_ls_d_res_high)
    newblock.set_pairs("_software.", software_dict)
    newblock.set_pairs("_struct_asym.", struct_asym_dict)
    newblock.set_mmcif_category("_atom_site", target_atom_site_dict)

    return newblock.as_string(options=options)


# %%
options = gemmi.cif.WriteOptions()
options.misuse_hash = True
options.align_loops = 20
options.prefer_pairs = True
pdbid = "6W81"
target_chain = "A"
mmcif_dir = "/Users/YoshitakaM/Desktop/mmcif_files"
structure_store = structure_stores.StructureStore(mmcif_dir)
mmcif_string = ciffilecontent(
    pdbid=pdbid,
    structure_store=structure_store,
    target_chain=target_chain,
    options=options,
    fix_mse_residues=True,
    fix_arginines=True,
    include_water=False,
    include_bonds=False,
    include_other=True,
)
print(mmcif_string)

# %%
