# %%
import dataclasses
import datetime
import functools
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Final, Self, TypeAlias

import gemmi
import numpy as np
import pandas as pd
from loguru import logger


def _sort_dict_by_keys(d):
    return {key: d[key] for key in sorted(d)}


def get_entity_id_for_chain(df_entity_polys, target_chain: str) -> str | None:
    """Get entity_id for a given chain ID from entity_poly DataFrame (vectorized)."""
    mask = (
        df_entity_polys["pdbx_strand_id"]
        .str.split(",")
        .apply(lambda chains: target_chain in chains)
    )
    matching_rows = df_entity_polys[mask]

    if len(matching_rows) == 0:
        return None
    return matching_rows.iloc[0]["entity_id"]


ciffile = "/Users/YoshitakaM/Desktop/mmcif_files/4HHB.cif"
outcif = "/Users/YoshitakaM/Desktop/mmcif_files/4HHB_out.cif"
target_chain = "A"
doc = gemmi.cif.read_file(ciffile)
block = doc.sole_block()
names = block.get_mmcif_category_names()
# _chem_compを抜き出し
chem_comps = block.get_mmcif_category("_chem_comp")
sorted_chem_comps = _sort_dict_by_keys(chem_comps)
# _entityを抜き出し
entities = block.get_mmcif_category("_entity")
entity_polys = block.get_mmcif_category("_entity_poly")
entity_poly_seqs = block.get_mmcif_category("_entity_poly_seq")
df_entity_polys = pd.DataFrame(entity_polys)
target_entity_id = get_entity_id_for_chain(df_entity_polys, target_chain)
df_entities = pd.DataFrame(entities)
target_entity = df_entities.loc[df_entities["id"] == target_entity_id]
entity_dict = {
    "id": target_entity["id"].values[0],
    "pdbx_description": target_entity["pdbx_description"].values[0],
    "type": target_entity["type"].values[0],
}
target_poly = df_entity_polys.loc[df_entity_polys["entity_id"] == target_entity_id]
poly_dict = {
    "entity_id": target_poly["entity_id"].values[0],
    "pdbx_strand_id": target_chain,
    "type": target_poly["type"].values[0],
}
df_entity_poly_seqs = pd.DataFrame(entity_poly_seqs)
target_poly_seqs = df_entity_poly_seqs.loc[
    df_entity_poly_seqs["entity_id"] == target_entity_id
]
# always set hetero to "n"
target_poly_seqs.loc[:, "hetero"] = "n"
polyseq_loop_dict = {
    "entity_id": target_poly_seqs["entity_id"].to_list(),
    "hetero": target_poly_seqs["hetero"].to_list(),
    "mon_id": target_poly_seqs["mon_id"].to_list(),
    "num": target_poly_seqs["num"].to_list(),
}

# _exptl.method
exptl_method = block.find_value("_exptl.method")
# _pdbx_audit_revision_history.revision_date: only the first row
revision_histories = block.get_mmcif_category("_pdbx_audit_revision_history")
df_revision_histories = pd.DataFrame(revision_histories)
first_revision = df_revision_histories.iloc[0]
first_revision_date = first_revision["revision_date"]
# _pdbx_database_status.recvd_initial_deposition_date
initial_deposition_date = block.find_value(
    "_pdbx_database_status.recvd_initial_deposition_date"
)

# _pdbx_poly_seq_scheme loop
pdbx_poly_seq_scheme = block.get_mmcif_category("_pdbx_poly_seq_scheme")
df_pdbx_poly_seq_scheme = pd.DataFrame(pdbx_poly_seq_scheme)
# filter by target_entity_id and target_chain
target_pdbx_poly_seq_scheme = df_pdbx_poly_seq_scheme.loc[
    df_pdbx_poly_seq_scheme["entity_id"] == target_entity_id
].loc[df_pdbx_poly_seq_scheme["pdb_strand_id"] == target_chain]
# always set hetero to "n"
target_pdbx_poly_seq_scheme.loc[:, "hetero"] = "n"
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
# %%
# %%
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
options = gemmi.cif.WriteOptions()
options.misuse_hash = True
newblock.write_file(outcif, options=options)
# %%
