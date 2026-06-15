"""Interaction metrics for protein–protein/nucleic-acid complexes.

Calculates ipSAE, ipTM, pDockQ, pDockQ2, LIS, and LIS-family scores
(iLIS, iLIA, iLISA, actifpTM, cLIS, LIA, cLIA) for AlphaFold2, AlphaFold3,
and Boltz structural models.

References:
    ipSAE: https://www.biorxiv.org/content/10.1101/2025.02.10.637595v2
    LIS family: https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1
Original ipSAE author: Roland Dunbrack, Fox Chase Cancer Center

Usage as module:
    from alphafold3tools.metrics import run_ipsae
    paths = run_ipsae("model.json", "model.cif", pae_cutoff=10, dist_cutoff=15)

Usage as CLI:
    af3tools metrics -i /path/to/output_directory --json
"""

import json
import math
import os
import sys
from argparse import ArgumentParser
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from alphafold3tools import __version__
from alphafold3tools.log import log_setup
from alphafold3tools.utils import add_version_option

np.set_printoptions(threshold=sys.maxsize)

# ── Constants ─────────────────────────────────────────────────────────────────

RESIDUE_SET: frozenset[str] = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "DA",
        "DC",
        "DT",
        "DG",
        "A",
        "C",
        "U",
        "G",
    }
)
NUC_RESIDUE_SET: frozenset[str] = frozenset(
    {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
)

CHAIN_COLORS: dict[str, str] = {
    "A": "magenta",
    "B": "marine",
    "C": "lime",
    "D": "orange",
    "E": "yellow",
    "F": "cyan",
    "G": "lightorange",
    "H": "pink",
    "I": "deepteal",
    "J": "forest",
    "K": "lightblue",
    "L": "slate",
    "M": "violet",
    "N": "arsenic",
    "O": "iodine",
    "P": "silver",
    "Q": "red",
    "R": "sulfur",
    "S": "purple",
    "T": "olive",
    "U": "palegreen",
    "V": "green",
    "W": "blue",
    "X": "palecyan",
    "Y": "limon",
    "Z": "chocolate",
}

# ── Math helpers ──────────────────────────────────────────────────────────────


def ptm_func(x: float, d0: float) -> float:
    return 1.0 / (1 + (x / d0) ** 2.0)


_ptm_vec = np.vectorize(ptm_func)


def calc_d0(L: float, pair_type: str) -> float:
    L = float(L)
    min_val = 2.0 if pair_type == "nucleic_acid" else 1.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8 if L > 27 else 1.0
    return max(min_val, d0)


def calc_d0_array(L: Any, pair_type: str) -> np.ndarray:
    L = np.maximum(26, np.array(L, dtype=float))
    min_val = 2.0 if pair_type == "nucleic_acid" else 1.0
    return np.maximum(min_val, 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8)


def contiguous_ranges(numbers: set[int]) -> str:
    if not numbers:
        return ""
    sorted_nums = sorted(numbers)
    start = end = sorted_nums[0]
    ranges: list[str] = []

    def _fmt(s: int, e: int) -> str:
        return f"{s}" if s == e else f"{s}-{e}"

    for n in sorted_nums[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(_fmt(start, end))
            start = end = n
    ranges.append(_fmt(start, end))
    return "+".join(ranges)


# ── Dict initializers ─────────────────────────────────────────────────────────


def _pair_zeros(chains: list[str]) -> dict:
    return {c1: {c2: 0 for c2 in chains if c1 != c2} for c1 in chains}


def _pair_npzeros(chains: list[str], size: int) -> dict:
    return {c1: {c2: np.zeros(size) for c2 in chains if c1 != c2} for c1 in chains}


def _pair_sets(chains: list[str]) -> dict:
    return {c1: {c2: set() for c2 in chains if c1 != c2} for c1 in chains}


# ── Structure utilities ───────────────────────────────────────────────────────


def classify_chains(chains: np.ndarray, residue_types: np.ndarray) -> dict[str, str]:
    _, first_idx = np.unique(chains, return_index=True)
    unique_chains = chains[np.sort(first_idx)]
    result: dict[str, str] = {}
    for chain in unique_chains:
        idxs = np.where(chains == chain)[0]
        nuc_count = sum(r in NUC_RESIDUE_SET for r in residue_types[idxs])
        result[chain] = "nucleic_acid" if nuc_count > 0 else "protein"
    return result


def parse_pdb_atom_line(line: str) -> dict | None:
    residue_name = line[17:20].strip()
    if residue_name == "LIG":
        return None
    return {
        "atom_num": int(line[6:11]),
        "atom_name": line[12:16].strip(),
        "residue_name": residue_name,
        "chain_id": line[21],
        "residue_seq_num": int(line[22:26]),
        "x": float(line[30:38]),
        "y": float(line[38:46]),
        "z": float(line[46:54]),
    }


def parse_cif_atom_line(line: str, fielddict: dict[str, int]) -> dict | None:
    parts = line.split()
    seq_id = parts[fielddict["label_seq_id"]]
    if seq_id == ".":
        return None
    chain_id = (
        parts[fielddict["auth_asym_id"]]
        if "auth_asym_id" in fielddict
        else parts[fielddict["label_asym_id"]]
    )
    return {
        "atom_num": int(parts[fielddict["id"]]),
        "atom_name": parts[fielddict["label_atom_id"]],
        "residue_name": parts[fielddict["label_comp_id"]],
        "chain_id": chain_id,
        "residue_seq_num": int(seq_id),
        "x": float(parts[fielddict["Cartn_x"]]),
        "y": float(parts[fielddict["Cartn_y"]]),
        "z": float(parts[fielddict["Cartn_z"]]),
    }


def parse_structure(
    struct_path: Path, is_cif: bool
) -> tuple[list[dict], list[dict], list[str], list[int]]:
    """Parse PDB or mmCIF file.

    Returns:
        (residues, cb_residues, chains, token_mask)
    """
    residues: list[dict] = []
    cb_residues: list[dict] = []
    chains: list[str] = []
    token_mask: list[int] = []
    fielddict: dict[str, int] = {}
    field_num = 0

    with open(struct_path) as fh:
        for line in fh:
            if line.startswith("_atom_site."):
                _, fieldname = line.strip().split(".")
                fielddict[fieldname] = field_num
                field_num += 1
                continue

            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom = (
                parse_cif_atom_line(line, fielddict)
                if is_cif
                else parse_pdb_atom_line(line)
            )
            if atom is None:
                token_mask.append(0)
                continue

            is_ca = atom["atom_name"] == "CA" or "C1" in atom["atom_name"]
            is_cb = (
                atom["atom_name"] == "CB"
                or "C3" in atom["atom_name"]
                or (atom["residue_name"] == "GLY" and atom["atom_name"] == "CA")
            )

            if is_ca:
                token_mask.append(1)
                residues.append(
                    {
                        "atom_num": atom["atom_num"],
                        "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                        "res": atom["residue_name"],
                        "chainid": atom["chain_id"],
                        "resnum": atom["residue_seq_num"],
                        "residue": (
                            f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}"
                        ),
                    }
                )
                chains.append(atom["chain_id"])

            if is_cb:
                cb_residues.append(
                    {
                        "atom_num": atom["atom_num"],
                        "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                        "res": atom["residue_name"],
                        "chainid": atom["chain_id"],
                        "resnum": atom["residue_seq_num"],
                        "residue": (
                            f"{atom['residue_name']:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}"
                        ),
                    }
                )

            if (
                not is_ca
                and "C1" not in atom["atom_name"]
                and atom["residue_name"] not in RESIDUE_SET
            ):
                token_mask.append(0)

    return residues, cb_residues, chains, token_mask


# ── PAE / pLDDT loaders ───────────────────────────────────────────────────────


def load_af2_data(
    pae_path: Path, numres: int
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Returns (plddt, pae_matrix, iptm, ptm)."""
    if pae_path.suffix == ".pkl":
        data = np.load(str(pae_path), allow_pickle=True)
    else:
        with open(pae_path) as fh:
            data = json.load(fh)

    iptm = float(data.get("iptm", -1.0))
    ptm_val = float(data.get("ptm", -1.0))
    plddt = np.array(data["plddt"]) if "plddt" in data else np.zeros(numres)

    if "pae" in data:
        pae_matrix = np.array(data["pae"])
    else:
        pae_matrix = np.array(data["predicted_aligned_error"])

    return plddt, pae_matrix, iptm, ptm_val


def load_boltz_data(
    pae_path: Path,
    token_mask: np.ndarray,
    unique_chains: np.ndarray,
    ntokens: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (plddt, pae_matrix, iptm_boltz)."""
    plddt_path = Path(str(pae_path).replace("pae", "plddt"))
    if plddt_path.exists():
        raw = np.load(str(plddt_path))["plddt"]
        full_plddt = 100.0 * raw if np.max(raw) <= 1.0 else raw.copy()
        plddt = full_plddt[token_mask.astype(bool)]
    else:
        plddt = np.zeros(ntokens)

    data_pae = np.load(str(pae_path))
    pae_full = np.array(data_pae["pae"])
    pae_matrix = pae_full[np.ix_(token_mask.astype(bool), token_mask.astype(bool))]

    iptm_boltz = _pair_zeros(list(unique_chains))
    summary_path = Path(
        str(pae_path).replace("pae", "confidence").replace(".npz", ".json")
    )
    if summary_path.exists():
        with open(summary_path) as fh:
            data_summary = json.load(fh)
        pair_data = data_summary.get("pair_chains_iptm", {})
        for ni, c1 in enumerate(unique_chains):
            for nj, c2 in enumerate(unique_chains):
                if c1 != c2:
                    iptm_boltz[c1][c2] = pair_data.get(str(ni), {}).get(str(nj), 0)
    else:
        print(f"Boltz summary file does not exist: {summary_path}")

    return plddt, pae_matrix, iptm_boltz


def load_af3_data(
    pae_path: Path,
    ca_atom_num: np.ndarray,
    cb_atom_num: np.ndarray,
    token_mask: np.ndarray,
    unique_chains: np.ndarray,
    numres: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Returns (plddt, cb_plddt, pae_matrix, iptm_af3)."""
    with open(pae_path) as fh:
        data = json.load(fh)

    if "atom_plddts" in data:
        atom_plddts = np.array(data["atom_plddts"])
        plddt = atom_plddts[ca_atom_num]
        cb_plddt = atom_plddts[cb_atom_num]
    else:
        plddt = np.zeros(numres)
        cb_plddt = np.zeros(numres)

    if "pae" not in data:
        raise ValueError(f"No PAE data in AF3 json file: {pae_path}")

    pae_full = np.array(data["pae"])
    pae_matrix = pae_full[np.ix_(token_mask.astype(bool), token_mask.astype(bool))]

    iptm_af3 = _pair_zeros(list(unique_chains))
    pae_str = str(pae_path)
    if "confidences" in pae_str:
        summary_path: Path | None = Path(
            pae_str.replace("confidences", "summary_confidences")
        )
    elif "full_data" in pae_str:
        summary_path = Path(pae_str.replace("full_data", "summary_confidences"))
    else:
        summary_path = None

    if summary_path is not None and summary_path.exists():
        with open(summary_path) as fh:
            data_summary = json.load(fh)
        pair_data = data_summary["chain_pair_iptm"]
        for ni, c1 in enumerate(unique_chains):
            for nj, c2 in enumerate(unique_chains):
                if c1 != c2:
                    iptm_af3[c1][c2] = pair_data[ni][nj]
    else:
        print(f"AF3 summary file does not exist: {summary_path}")

    return plddt, cb_plddt, pae_matrix, iptm_af3


# ── Score computations ────────────────────────────────────────────────────────


def compute_pdockq(
    unique_chains: np.ndarray,
    chains: np.ndarray,
    distances: np.ndarray,
    cb_plddt: np.ndarray,
    numres: int,
) -> tuple[dict, dict]:
    """Returns (pdockq_residues, pdockq)."""
    cutoff = 8.0
    pdockq_res = _pair_sets(list(unique_chains))
    pdockq = _pair_zeros(list(unique_chains))

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs = 0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid = (chains == chain2) & (distances[i] <= cutoff)
                npairs += int(np.sum(valid))
                if valid.any():
                    pdockq_res[chain1][chain2].add(i)
                    for j in np.where(valid)[0]:
                        pdockq_res[chain1][chain2].add(int(j))

            if npairs > 0:
                res_list = list(pdockq_res[chain1][chain2])
                mean_plddt = cb_plddt[res_list].mean()
                x = mean_plddt * math.log10(npairs)
                pdockq[chain1][chain2] = (
                    0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
                )

    return pdockq_res, pdockq


def compute_pdockq2(
    unique_chains: np.ndarray,
    chains: np.ndarray,
    distances: np.ndarray,
    cb_plddt: np.ndarray,
    pae_matrix: np.ndarray,
    pdockq_res: dict,
    numres: int,
) -> dict:
    """Returns pdockq2 dict."""
    cutoff = 8.0
    pdockq2 = _pair_zeros(list(unique_chains))

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs = 0
            total_ptm = 0.0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid = (chains == chain2) & (distances[i] <= cutoff)
                if valid.any():
                    npairs += int(np.sum(valid))
                    total_ptm += _ptm_vec(pae_matrix[i][valid], 10.0).sum()

            if npairs > 0:
                res_list = list(pdockq_res[chain1][chain2])
                mean_plddt = cb_plddt[res_list].mean()
                x = mean_plddt * (total_ptm / npairs)
                pdockq2[chain1][chain2] = (
                    1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
                )

    return pdockq2


def compute_lis(
    unique_chains: np.ndarray,
    chains: np.ndarray,
    pae_matrix: np.ndarray,
) -> dict:
    """Returns LIS dict."""
    lis = _pair_zeros(list(unique_chains))
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            mask = (chains[:, None] == chain1) & (chains[None, :] == chain2)
            selected = pae_matrix[mask]
            if selected.size > 0:
                valid = selected[selected < 12]
                lis[chain1][chain2] = (
                    float(np.mean((12 - valid) / 12)) if valid.size > 0 else 0.0
                )
    return lis


# ── LIS-family metric computations (AFM-LIS) ──────────────────────────────────
#
# Reimplements the LIS/cLIS/iLIS/LIA/cLIA/iLIA/iLISA/actifpTM scores from
# AFM-LIS (https://github.com/flyark/AFM-LIS) -- see also
# https://doi.org/10.1101/2024.02.19.580970 and
# https://doi.org/10.1101/2025.03.24.644977 for actifpTM.
#
# Conventions:
#   * Per-direction LIS_{A->B} = mean( (cutoff - PAE)/cutoff ) over PAE < cutoff.
#   * Symmetric pair-level: LIS = (LIS_{A->B} + LIS_{B->A}) / 2; same for cLIS.
#   * LIA / cLIA are *counts*, summed over both directions.
#   * iLIS = sqrt(LIS * cLIS); iLIA = sqrt(LIA * cLIA); iLISA = iLIS * iLIA.
#   * actifpTM uses TM-score(d0 from full complex length) * binary CB contact,
#     and takes the max over residues and both directions.


def _transform_pae_matrix_lis(pae: np.ndarray, pae_cutoff: float) -> np.ndarray:
    """Transform PAE to per-residue-pair confidence in [0, 1).

    Returns 1 - PAE/cutoff where PAE < cutoff, else 0. Stays asymmetric.
    """
    pae = np.asarray(pae, dtype=np.float64)
    transformed = np.zeros_like(pae)
    mask = pae < pae_cutoff
    transformed[mask] = 1.0 - pae[mask] / pae_cutoff
    return transformed


def _compute_lis_contact_map(
    distances: np.ndarray,
    residue_types: np.ndarray,
    cb_cutoff: float,
) -> np.ndarray:
    """Binary residue-residue contact map (Cβ–Cβ, with +4Å reach for nucleic acids).

    AFM-LIS uses ``distance - 4Å`` for residue pairs that include a P atom
    (i.e. nucleic-acid residues whose pseudo-Cβ is C3'); the equivalent here is
    extending the cutoff by 4Å whenever either residue is a nucleic acid.
    """
    is_nuc = np.array([r in NUC_RESIDUE_SET for r in residue_types])
    p_mask = is_nuc[:, None] | is_nuc[None, :]
    effective_cutoff = np.where(p_mask, cb_cutoff + 4.0, cb_cutoff)
    return (distances < effective_cutoff).astype(np.uint8)


def _calc_actifptm_pair(
    pae: np.ndarray,
    contact: np.ndarray,
    si: int,
    ei: int,
    sj: int,
    ej: int,
) -> float:
    """Approximate actifpTM (Varga & Ovchinnikov 2025) for a chain pair.

    Per-residue weighted TM-score with binary CB contact as the weight, where
    d0 is derived from the full-complex length. Returns the max per-residue
    score across both interfacial directions (A→B and B→A).
    """
    n_total = int(pae.shape[0])
    clipped = max(n_total, 19)
    d0 = 1.24 * (clipped - 15) ** (1.0 / 3.0) - 1.8
    d0sq = d0 * d0

    def _one_direction(r0: int, r1: int, s0: int, s1: int) -> float:
        if r1 <= r0 or s1 <= s0:
            return 0.0
        contact_block = contact[r0:r1, s0:s1]
        weight_sums = contact_block.sum(axis=1)
        has_contact = weight_sums > 0
        if not has_contact.any():
            return 0.0
        pae_block = pae[r0:r1, s0:s1].astype(np.float64)
        tm_vals = np.where(
            contact_block.astype(bool),
            1.0 / (1.0 + (pae_block * pae_block) / d0sq),
            0.0,
        )
        scores = tm_vals[has_contact].sum(axis=1) / weight_sums[has_contact]
        return float(scores.max())

    return max(_one_direction(si, ei, sj, ej), _one_direction(sj, ej, si, ei))


def compute_lis_metrics(
    unique_chains: np.ndarray,
    chains: np.ndarray,
    residues: list[dict],
    pae_matrix: np.ndarray,
    distances: np.ndarray,
    pae_cutoff_lis: float = 12.0,
    cb_cutoff_lis: float = 8.0,
) -> dict[str, dict]:
    """Compute LIS-family metrics for every (chain_a, chain_b) pair with a<b.

    Returns: ``{"A-B": {"LIS": ..., "cLIS": ..., "LIA": ..., "cLIA": ...,
    "iLIS": ..., "iLIA": ..., "iLISA": ..., "actifpTM": ...}, ...}``
    """
    transformed = _transform_pae_matrix_lis(pae_matrix, pae_cutoff_lis)
    residue_types = np.array([r["res"] for r in residues])
    contact = _compute_lis_contact_map(distances, residue_types, cb_cutoff_lis)

    # Contiguous chain ranges (parse_structure preserves chain order)
    chain_ranges: dict[str, tuple[int, int]] = {}
    for c in unique_chains:
        idxs = np.where(chains == c)[0]
        if idxs.size > 0:
            chain_ranges[str(c)] = (int(idxs[0]), int(idxs[-1]) + 1)

    asym_lis: dict[str, dict[str, float]] = {}
    asym_clis: dict[str, dict[str, float]] = {}
    asym_lia: dict[str, dict[str, int]] = {}
    asym_clia: dict[str, dict[str, int]] = {}
    asym_actifptm: dict[str, dict[str, float]] = {}
    for c in unique_chains:
        key = str(c)
        asym_lis[key] = {}
        asym_clis[key] = {}
        asym_lia[key] = {}
        asym_clia[key] = {}
        asym_actifptm[key] = {}

    for c1 in unique_chains:
        c1s = str(c1)
        if c1s not in chain_ranges:
            continue
        for c2 in unique_chains:
            c2s = str(c2)
            if c1s == c2s or c2s not in chain_ranges:
                continue
            si, ei = chain_ranges[c1s]
            sj, ej = chain_ranges[c2s]

            t_block = transformed[si:ei, sj:ej]
            t_pos = t_block > 0
            lis_count = int(t_pos.sum())
            asym_lis[c1s][c2s] = (
                float(t_block[t_pos].sum()) / lis_count if lis_count > 0 else 0.0
            )

            contact_block = contact[si:ei, sj:ej].astype(bool)
            ct_pos = t_pos & contact_block
            clis_count = int(ct_pos.sum())
            asym_clis[c1s][c2s] = (
                float(t_block[ct_pos].sum()) / clis_count if clis_count > 0 else 0.0
            )

            pae_block = pae_matrix[si:ei, sj:ej]
            asym_lia[c1s][c2s] = int((pae_block < pae_cutoff_lis).sum())
            asym_clia[c1s][c2s] = int(
                ((pae_block < pae_cutoff_lis) & contact_block).sum()
            )

            asym_actifptm[c1s][c2s] = _calc_actifptm_pair(
                pae_matrix, contact, si, ei, sj, ej
            )

    pair_metrics: dict[str, dict] = {}
    chain_list = [str(c) for c in unique_chains]
    for i, c1 in enumerate(chain_list):
        for c2 in chain_list[i + 1 :]:
            if c1 not in chain_ranges or c2 not in chain_ranges:
                continue
            lis = (asym_lis[c1][c2] + asym_lis[c2][c1]) / 2.0
            clis = (asym_clis[c1][c2] + asym_clis[c2][c1]) / 2.0
            lia = asym_lia[c1][c2] + asym_lia[c2][c1]
            clia = asym_clia[c1][c2] + asym_clia[c2][c1]
            ilis = math.sqrt(lis * clis)
            ilia = math.sqrt(lia * clia)
            ilisa = ilis * ilia
            actifptm = max(asym_actifptm[c1][c2], asym_actifptm[c2][c1])

            pair_metrics[f"{c1}-{c2}"] = {
                "LIS": round(lis, 4),
                "cLIS": round(clis, 4),
                "LIA": int(lia),
                "cLIA": int(clia),
                "iLIS": round(ilis, 4),
                "iLIA": round(ilia, 1),
                "iLISA": round(ilisa, 1),
                "actifpTM": round(actifptm, 4),
            }

    return pair_metrics


def compute_ipsae(
    unique_chains: np.ndarray,
    chains: np.ndarray,
    pae_matrix: np.ndarray,
    distances: np.ndarray,
    residues: list[dict],
    chain_pair_type: dict,
    pae_cutoff: float,
    dist_cutoff: float,
) -> dict:
    """Compute all ipTM/ipSAE variants.

    Returns a dict containing all score arrays and metadata needed for output.
    """
    numres = len(residues)
    uc = list(unique_chains)

    iptm_d0chn_byres = _pair_npzeros(uc, numres)
    ipsae_d0chn_byres = _pair_npzeros(uc, numres)
    ipsae_d0dom_byres = _pair_npzeros(uc, numres)
    ipsae_d0res_byres = _pair_npzeros(uc, numres)

    iptm_d0chn_asym = _pair_zeros(uc)
    ipsae_d0chn_asym = _pair_zeros(uc)
    ipsae_d0dom_asym = _pair_zeros(uc)
    ipsae_d0res_asym = _pair_zeros(uc)

    iptm_d0chn_max = _pair_zeros(uc)
    ipsae_d0chn_max = _pair_zeros(uc)
    ipsae_d0dom_max = _pair_zeros(uc)
    ipsae_d0res_max = _pair_zeros(uc)

    iptm_d0chn_asymres = _pair_zeros(uc)
    ipsae_d0chn_asymres = _pair_zeros(uc)
    ipsae_d0dom_asymres = _pair_zeros(uc)
    ipsae_d0res_asymres = _pair_zeros(uc)

    iptm_d0chn_maxres = _pair_zeros(uc)
    ipsae_d0chn_maxres = _pair_zeros(uc)
    ipsae_d0dom_maxres = _pair_zeros(uc)
    ipsae_d0res_maxres = _pair_zeros(uc)

    iptm_d0chn_min = _pair_zeros(uc)
    ipsae_d0chn_min = _pair_zeros(uc)
    ipsae_d0dom_min = _pair_zeros(uc)
    ipsae_d0res_min = _pair_zeros(uc)

    iptm_d0chn_minres = _pair_zeros(uc)
    ipsae_d0chn_minres = _pair_zeros(uc)
    ipsae_d0dom_minres = _pair_zeros(uc)
    ipsae_d0res_minres = _pair_zeros(uc)

    n0chn = _pair_zeros(uc)
    n0dom = _pair_zeros(uc)
    n0dom_max = _pair_zeros(uc)
    n0dom_min = _pair_zeros(uc)
    n0res = _pair_zeros(uc)
    n0res_max = _pair_zeros(uc)
    n0res_min = _pair_zeros(uc)
    n0res_byres = _pair_npzeros(uc, numres)

    d0chn = _pair_zeros(uc)
    d0dom = _pair_zeros(uc)
    d0dom_max = _pair_zeros(uc)
    d0dom_min = _pair_zeros(uc)
    d0res = _pair_zeros(uc)
    d0res_max = _pair_zeros(uc)
    d0res_min = _pair_zeros(uc)
    d0res_byres = _pair_npzeros(uc, numres)

    valid_pair_counts = _pair_zeros(uc)
    dist_valid_pair_counts = _pair_zeros(uc)
    unique_res_chain1 = _pair_sets(uc)
    unique_res_chain2 = _pair_sets(uc)
    dist_unique_res_chain1 = _pair_sets(uc)
    dist_unique_res_chain2 = _pair_sets(uc)

    # Pass 1: d0chn-based scores and interface residue tracking
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            n0chn[chain1][chain2] = int(
                np.sum(chains == chain1) + np.sum(chains == chain2)
            )
            d0chn[chain1][chain2] = calc_d0(
                n0chn[chain1][chain2], chain_pair_type[chain1][chain2]
            )
            ptm_d0chn = _ptm_vec(pae_matrix, d0chn[chain1][chain2])
            valid_iptm = chains == chain2
            valid_matrix = np.outer(chains == chain1, chains == chain2) & (
                pae_matrix < pae_cutoff
            )

            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_ipsae = valid_matrix[i]
                iptm_d0chn_byres[chain1][chain2][i] = (
                    ptm_d0chn[i, valid_iptm].mean() if valid_iptm.any() else 0.0
                )
                ipsae_d0chn_byres[chain1][chain2][i] = (
                    ptm_d0chn[i, valid_ipsae].mean() if valid_ipsae.any() else 0.0
                )
                valid_pair_counts[chain1][chain2] += int(np.sum(valid_ipsae))
                if valid_ipsae.any():
                    unique_res_chain1[chain1][chain2].add(residues[i]["resnum"])
                    for j in np.where(valid_ipsae)[0]:
                        unique_res_chain2[chain1][chain2].add(residues[j]["resnum"])

                valid_dist = (
                    (chains == chain2)
                    & (pae_matrix[i] < pae_cutoff)
                    & (distances[i] < dist_cutoff)
                )
                dist_valid_pair_counts[chain1][chain2] += int(np.sum(valid_dist))
                if valid_dist.any():
                    dist_unique_res_chain1[chain1][chain2].add(residues[i]["resnum"])
                    for j in np.where(valid_dist)[0]:
                        dist_unique_res_chain2[chain1][chain2].add(
                            residues[j]["resnum"]
                        )

    # Pass 2: d0dom and d0res scores (require interface residue counts from pass 1)
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            r1 = len(unique_res_chain1[chain1][chain2])
            r2 = len(unique_res_chain2[chain1][chain2])
            n0dom[chain1][chain2] = r1 + r2
            d0dom[chain1][chain2] = calc_d0(
                n0dom[chain1][chain2], chain_pair_type[chain1][chain2]
            )
            ptm_d0dom = _ptm_vec(pae_matrix, d0dom[chain1][chain2])
            valid_matrix = np.outer(chains == chain1, chains == chain2) & (
                pae_matrix < pae_cutoff
            )
            n0res_all = np.sum(valid_matrix, axis=1)
            d0res_all = calc_d0_array(n0res_all, chain_pair_type[chain1][chain2])
            n0res_byres[chain1][chain2] = n0res_all
            d0res_byres[chain1][chain2] = d0res_all

            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid = valid_matrix[i]
                ipsae_d0dom_byres[chain1][chain2][i] = (
                    ptm_d0dom[i, valid].mean() if valid.any() else 0.0
                )
                ptm_row_d0res = _ptm_vec(pae_matrix[i], d0res_byres[chain1][chain2][i])
                ipsae_d0res_byres[chain1][chain2][i] = (
                    ptm_row_d0res[valid].mean() if valid.any() else 0.0
                )

    # Pass 3: asymmetric and max scores per chain pair
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            def _set_asym(
                byres: dict,
                asym: dict,
                asymres: dict,
                source_chain: str,
                target_chain: str,
            ) -> int:
                vals = byres[source_chain][target_chain]
                idx = int(np.argmax(vals))
                asym[source_chain][target_chain] = float(vals[idx])
                asymres[source_chain][target_chain] = residues[idx]["residue"]
                return idx

            _set_asym(
                iptm_d0chn_byres,
                iptm_d0chn_asym,
                iptm_d0chn_asymres,
                chain1,
                chain2,
            )
            _set_asym(
                ipsae_d0chn_byres,
                ipsae_d0chn_asym,
                ipsae_d0chn_asymres,
                chain1,
                chain2,
            )
            _set_asym(
                ipsae_d0dom_byres,
                ipsae_d0dom_asym,
                ipsae_d0dom_asymres,
                chain1,
                chain2,
            )
            best_idx = _set_asym(
                ipsae_d0res_byres,
                ipsae_d0res_asym,
                ipsae_d0res_asymres,
                chain1,
                chain2,
            )

            n0res[chain1][chain2] = n0res_byres[chain1][chain2][best_idx]
            d0res[chain1][chain2] = d0res_byres[chain1][chain2][best_idx]

            if chain1 > chain2:

                def _set_max(
                    asym_d: dict,
                    max_d: dict,
                    asymres_d: dict,
                    maxres_d: dict,
                    source_chain: str,
                    target_chain: str,
                ) -> None:
                    v1 = asym_d[source_chain][target_chain]
                    v2 = asym_d[target_chain][source_chain]
                    mv = max(v1, v2)
                    mr = (
                        asymres_d[source_chain][target_chain]
                        if mv == v1
                        else asymres_d[target_chain][source_chain]
                    )
                    max_d[source_chain][target_chain] = mv
                    max_d[target_chain][source_chain] = mv
                    maxres_d[source_chain][target_chain] = mr
                    maxres_d[target_chain][source_chain] = mr

                _set_max(
                    iptm_d0chn_asym,
                    iptm_d0chn_max,
                    iptm_d0chn_asymres,
                    iptm_d0chn_maxres,
                    chain1,
                    chain2,
                )
                _set_max(
                    ipsae_d0chn_asym,
                    ipsae_d0chn_max,
                    ipsae_d0chn_asymres,
                    ipsae_d0chn_maxres,
                    chain1,
                    chain2,
                )

                v1 = ipsae_d0dom_asym[chain1][chain2]
                v2 = ipsae_d0dom_asym[chain2][chain1]
                mv = max(v1, v2)
                if mv == v1:
                    mr = ipsae_d0dom_asymres[chain1][chain2]
                    mn0 = n0dom[chain1][chain2]
                    md0 = d0dom[chain1][chain2]
                else:
                    mr = ipsae_d0dom_asymres[chain2][chain1]
                    mn0 = n0dom[chain2][chain1]
                    md0 = d0dom[chain2][chain1]
                for c1, c2 in ((chain1, chain2), (chain2, chain1)):
                    ipsae_d0dom_max[c1][c2] = mv
                    ipsae_d0dom_maxres[c1][c2] = mr
                    n0dom_max[c1][c2] = mn0
                    d0dom_max[c1][c2] = md0

                v1 = ipsae_d0res_asym[chain1][chain2]
                v2 = ipsae_d0res_asym[chain2][chain1]
                mv = max(v1, v2)
                if mv == v1:
                    mr = ipsae_d0res_asymres[chain1][chain2]
                    mn0 = n0res[chain1][chain2]
                    md0 = d0res[chain1][chain2]
                else:
                    mr = ipsae_d0res_asymres[chain2][chain1]
                    mn0 = n0res[chain2][chain1]
                    md0 = d0res[chain2][chain1]
                for c1, c2 in ((chain1, chain2), (chain2, chain1)):
                    ipsae_d0res_max[c1][c2] = mv
                    ipsae_d0res_maxres[c1][c2] = mr
                    n0res_max[c1][c2] = mn0
                    d0res_max[c1][c2] = md0

                def _set_min(
                    asym_d: dict,
                    min_d: dict,
                    asymres_d: dict,
                    minres_d: dict,
                    source_chain: str,
                    target_chain: str,
                ) -> None:
                    v1 = asym_d[source_chain][target_chain]
                    v2 = asym_d[target_chain][source_chain]
                    mv = min(v1, v2)
                    mr = (
                        asymres_d[source_chain][target_chain]
                        if mv == v1
                        else asymres_d[target_chain][source_chain]
                    )
                    min_d[source_chain][target_chain] = mv
                    min_d[target_chain][source_chain] = mv
                    minres_d[source_chain][target_chain] = mr
                    minres_d[target_chain][source_chain] = mr

                _set_min(
                    iptm_d0chn_asym,
                    iptm_d0chn_min,
                    iptm_d0chn_asymres,
                    iptm_d0chn_minres,
                    chain1,
                    chain2,
                )
                _set_min(
                    ipsae_d0chn_asym,
                    ipsae_d0chn_min,
                    ipsae_d0chn_asymres,
                    ipsae_d0chn_minres,
                    chain1,
                    chain2,
                )

                v1 = ipsae_d0dom_asym[chain1][chain2]
                v2 = ipsae_d0dom_asym[chain2][chain1]
                mv = min(v1, v2)
                if mv == v1:
                    mr = ipsae_d0dom_asymres[chain1][chain2]
                    mn0 = n0dom[chain1][chain2]
                    md0 = d0dom[chain1][chain2]
                else:
                    mr = ipsae_d0dom_asymres[chain2][chain1]
                    mn0 = n0dom[chain2][chain1]
                    md0 = d0dom[chain2][chain1]
                for c1, c2 in ((chain1, chain2), (chain2, chain1)):
                    ipsae_d0dom_min[c1][c2] = mv
                    ipsae_d0dom_minres[c1][c2] = mr
                    n0dom_min[c1][c2] = mn0
                    d0dom_min[c1][c2] = md0

                v1 = ipsae_d0res_asym[chain1][chain2]
                v2 = ipsae_d0res_asym[chain2][chain1]
                mv = min(v1, v2)
                if mv == v1:
                    mr = ipsae_d0res_asymres[chain1][chain2]
                    mn0 = n0res[chain1][chain2]
                    md0 = d0res[chain1][chain2]
                else:
                    mr = ipsae_d0res_asymres[chain2][chain1]
                    mn0 = n0res[chain2][chain1]
                    md0 = d0res[chain2][chain1]
                for c1, c2 in ((chain1, chain2), (chain2, chain1)):
                    ipsae_d0res_min[c1][c2] = mv
                    ipsae_d0res_minres[c1][c2] = mr
                    n0res_min[c1][c2] = mn0
                    d0res_min[c1][c2] = md0

    return {
        "iptm_d0chn_byres": iptm_d0chn_byres,
        "ipsae_d0chn_byres": ipsae_d0chn_byres,
        "ipsae_d0dom_byres": ipsae_d0dom_byres,
        "ipsae_d0res_byres": ipsae_d0res_byres,
        "iptm_d0chn_asym": iptm_d0chn_asym,
        "ipsae_d0chn_asym": ipsae_d0chn_asym,
        "ipsae_d0dom_asym": ipsae_d0dom_asym,
        "ipsae_d0res_asym": ipsae_d0res_asym,
        "iptm_d0chn_max": iptm_d0chn_max,
        "ipsae_d0chn_max": ipsae_d0chn_max,
        "ipsae_d0dom_max": ipsae_d0dom_max,
        "ipsae_d0res_max": ipsae_d0res_max,
        "iptm_d0chn_asymres": iptm_d0chn_asymres,
        "ipsae_d0chn_asymres": ipsae_d0chn_asymres,
        "ipsae_d0dom_asymres": ipsae_d0dom_asymres,
        "ipsae_d0res_asymres": ipsae_d0res_asymres,
        "iptm_d0chn_maxres": iptm_d0chn_maxres,
        "ipsae_d0chn_maxres": ipsae_d0chn_maxres,
        "ipsae_d0dom_maxres": ipsae_d0dom_maxres,
        "ipsae_d0res_maxres": ipsae_d0res_maxres,
        "iptm_d0chn_min": iptm_d0chn_min,
        "ipsae_d0chn_min": ipsae_d0chn_min,
        "ipsae_d0dom_min": ipsae_d0dom_min,
        "ipsae_d0res_min": ipsae_d0res_min,
        "iptm_d0chn_minres": iptm_d0chn_minres,
        "ipsae_d0chn_minres": ipsae_d0chn_minres,
        "ipsae_d0dom_minres": ipsae_d0dom_minres,
        "ipsae_d0res_minres": ipsae_d0res_minres,
        "n0chn": n0chn,
        "n0dom": n0dom,
        "n0dom_max": n0dom_max,
        "n0dom_min": n0dom_min,
        "n0res": n0res,
        "n0res_max": n0res_max,
        "n0res_min": n0res_min,
        "n0res_byres": n0res_byres,
        "d0chn": d0chn,
        "d0dom": d0dom,
        "d0dom_max": d0dom_max,
        "d0dom_min": d0dom_min,
        "d0res": d0res,
        "d0res_max": d0res_max,
        "d0res_min": d0res_min,
        "d0res_byres": d0res_byres,
        "valid_pair_counts": valid_pair_counts,
        "dist_valid_pair_counts": dist_valid_pair_counts,
        "unique_res_chain1": unique_res_chain1,
        "unique_res_chain2": unique_res_chain2,
        "dist_unique_res_chain1": dist_unique_res_chain1,
        "dist_unique_res_chain2": dist_unique_res_chain2,
    }


# ── Output writers ────────────────────────────────────────────────────────────


def write_byres(
    out_path: Path,
    unique_chains: np.ndarray,
    chains: np.ndarray,
    residues: list[dict],
    plddt: np.ndarray,
    scores: dict,
) -> None:
    numres = len(residues)
    with open(out_path, "w") as fh:
        fh.write(
            "i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT"
            "      n0chn  n0dom  n0res    d0chn     d0dom     d0res"
            "   ipTM_pae  ipSAE_d0chn ipSAE_d0dom    ipSAE \n"
        )
        for chain1 in unique_chains:
            for chain2 in unique_chains:
                if chain1 == chain2:
                    continue
                for i in range(numres):
                    if chains[i] != chain1:
                        continue
                    fh.write(
                        f"{i + 1:<4d}    "
                        f"{chain1:4}      "
                        f"{chain2:4}      "
                        f"{residues[i]['resnum']:4d}           "
                        f"{residues[i]['res']:3}        "
                        f"{plddt[i]:8.2f}         "
                        f"{int(scores['n0chn'][chain1][chain2]):5d}  "
                        f"{int(scores['n0dom'][chain1][chain2]):5d}  "
                        f"{int(scores['n0res_byres'][chain1][chain2][i]):5d}  "
                        f"{scores['d0chn'][chain1][chain2]:8.3f}  "
                        f"{scores['d0dom'][chain1][chain2]:8.3f}  "
                        f"{scores['d0res_byres'][chain1][chain2][i]:8.3f}   "
                        f"{scores['iptm_d0chn_byres'][chain1][chain2][i]:8.4f}    "
                        f"{scores['ipsae_d0chn_byres'][chain1][chain2][i]:8.4f}    "
                        f"{scores['ipsae_d0dom_byres'][chain1][chain2][i]:8.4f}    "
                        f"{scores['ipsae_d0res_byres'][chain1][chain2][i]:8.4f}\n"
                    )


def _fmt_summary_line(
    chain1: str,
    chain2: str,
    pae_string: str,
    dist_string: str,
    score_type: str,
    ipsae: float,
    ipsae_d0chn: float,
    ipsae_d0dom: float,
    iptm_af: float,
    iptm_d0chn: float,
    pdockq_val: float,
    pdockq2_val: float,
    lis_val: float,
    n0res_val: int,
    n0chn_val: int,
    n0dom_val: int,
    d0res_val: float,
    d0chn_val: float,
    d0dom_val: float,
    nres1: int,
    nres2: int,
    dist1: int,
    dist2: int,
    model_stem: str,
) -> str:
    return (
        f"{chain1}    {chain2}     {pae_string:3}  {dist_string:3}  {score_type:5} "
        f"{ipsae:8.6f}    "
        f"{ipsae_d0chn:8.6f}    "
        f"{ipsae_d0dom:8.6f}    "
        f"{iptm_af:5.3f}    "
        f"{iptm_d0chn:8.6f}    "
        f"{pdockq_val:8.4f}   "
        f"{pdockq2_val:8.4f}   "
        f"{lis_val:8.4f}   "
        f"{n0res_val:5d}  "
        f"{n0chn_val:5d}  "
        f"{n0dom_val:5d}  "
        f"{d0res_val:6.2f}  "
        f"{d0chn_val:6.2f}  "
        f"{d0dom_val:6.2f}  "
        f"{nres1:5d}   "
        f"{nres2:5d}   "
        f"{dist1:5d}   "
        f"{dist2:5d}   "
        f"{model_stem}\n"
    )


def _make_score_record(
    chain1: str,
    chain2: str,
    ipsae: float,
    ipsae_d0chn: float,
    ipsae_d0dom: float,
    iptm_af: float,
    iptm_d0chn: float,
    pdockq_val: float,
    pdockq2_val: float,
    lis_val: float,
    n0res_val: int,
    n0chn_val: int,
    n0dom_val: int,
    d0res_val: float,
    d0chn_val: float,
    d0dom_val: float,
    nres1: int,
    nres2: int,
    dist1: int,
    dist2: int,
) -> dict:
    return {
        "chain1": chain1,
        "chain2": chain2,
        "ipSAE": round(float(ipsae), 6),
        "ipSAE_d0chn": round(float(ipsae_d0chn), 6),
        "ipSAE_d0dom": round(float(ipsae_d0dom), 6),
        "ipTM_af": round(float(iptm_af), 3),
        "ipTM_d0chn": round(float(iptm_d0chn), 6),
        "pDockQ": round(float(pdockq_val), 4),
        "pDockQ2": round(float(pdockq2_val), 4),
        "LIS": round(float(lis_val), 4),
        "n0res": int(n0res_val),
        "n0chn": int(n0chn_val),
        "n0dom": int(n0dom_val),
        "d0res": round(float(d0res_val), 2),
        "d0chn": round(float(d0chn_val), 2),
        "d0dom": round(float(d0dom_val), 2),
        "nres1": int(nres1),
        "nres2": int(nres2),
        "dist1": int(dist1),
        "dist2": int(dist2),
    }


def scores_to_json(
    model_stem: str,
    unique_chains: np.ndarray,
    scores: dict,
    pdockq: dict,
    pdockq2: dict,
    lis: dict,
    iptm_per_pair: dict,
    pae_cutoff: float,
    dist_cutoff: float,
    lis_metrics: dict[str, dict] | None = None,
    lis_pae_cutoff: float | None = None,
    lis_cb_cutoff: float | None = None,
) -> dict:
    """Build JSON-serialisable dict from computed scores."""
    model_name = Path(model_stem).name

    chainpairs: set[str] = set()
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 < c2:
                chainpairs.add(f"{c1}-{c2}")

    pair_data: dict = {}
    for pair in sorted(chainpairs):
        chain_a, chain_b = pair.split("-")  # chain_a < chain_b alphabetically

        asym_records = []
        for chain1, chain2 in ((chain_a, chain_b), (chain_b, chain_a)):
            nres1 = len(scores["unique_res_chain1"][chain1][chain2])
            nres2 = len(scores["unique_res_chain2"][chain1][chain2])
            dist1 = len(scores["dist_unique_res_chain1"][chain1][chain2])
            dist2 = len(scores["dist_unique_res_chain2"][chain1][chain2])
            asym_records.append(
                _make_score_record(
                    chain1,
                    chain2,
                    scores["ipsae_d0res_asym"][chain1][chain2],
                    scores["ipsae_d0chn_asym"][chain1][chain2],
                    scores["ipsae_d0dom_asym"][chain1][chain2],
                    iptm_per_pair[chain1][chain2],
                    scores["iptm_d0chn_asym"][chain1][chain2],
                    pdockq[chain1][chain2],
                    pdockq2[chain1][chain2],
                    lis[chain1][chain2],
                    int(scores["n0res"][chain1][chain2]),
                    int(scores["n0chn"][chain1][chain2]),
                    int(scores["n0dom"][chain1][chain2]),
                    scores["d0res"][chain1][chain2],
                    scores["d0chn"][chain1][chain2],
                    scores["d0dom"][chain1][chain2],
                    nres1,
                    nres2,
                    dist1,
                    dist2,
                )
            )

        # max record — chain_b > chain_a; max values stored at [chain_b][chain_a]
        nres1_max = max(
            len(scores["unique_res_chain2"][chain_b][chain_a]),
            len(scores["unique_res_chain1"][chain_a][chain_b]),
        )
        nres2_max = max(
            len(scores["unique_res_chain1"][chain_b][chain_a]),
            len(scores["unique_res_chain2"][chain_a][chain_b]),
        )
        dist1_max = max(
            len(scores["dist_unique_res_chain2"][chain_b][chain_a]),
            len(scores["dist_unique_res_chain1"][chain_a][chain_b]),
        )
        dist2_max = max(
            len(scores["dist_unique_res_chain1"][chain_b][chain_a]),
            len(scores["dist_unique_res_chain2"][chain_a][chain_b]),
        )
        pdockq2_max = max(pdockq2[chain_b][chain_a], pdockq2[chain_a][chain_b])
        lis_avg = (lis[chain_b][chain_a] + lis[chain_a][chain_b]) / 2.0
        iptm_af_max = max(
            iptm_per_pair[chain_b][chain_a], iptm_per_pair[chain_a][chain_b]
        )

        max_record = _make_score_record(
            chain_a,
            chain_b,
            scores["ipsae_d0res_max"][chain_b][chain_a],
            scores["ipsae_d0chn_max"][chain_b][chain_a],
            scores["ipsae_d0dom_max"][chain_b][chain_a],
            iptm_af_max,
            scores["iptm_d0chn_max"][chain_b][chain_a],
            pdockq[chain_b][chain_a],
            pdockq2_max,
            lis_avg,
            int(scores["n0res_max"][chain_b][chain_a]),
            int(scores["n0chn"][chain_b][chain_a]),
            int(scores["n0dom_max"][chain_b][chain_a]),
            scores["d0res_max"][chain_b][chain_a],
            scores["d0chn"][chain_b][chain_a],
            scores["d0dom_max"][chain_b][chain_a],
            nres1_max,
            nres2_max,
            dist1_max,
            dist2_max,
        )

        # min record — min values stored at [chain_b][chain_a] (same indexing as max)
        nres1_min = min(
            len(scores["unique_res_chain2"][chain_b][chain_a]),
            len(scores["unique_res_chain1"][chain_a][chain_b]),
        )
        nres2_min = min(
            len(scores["unique_res_chain1"][chain_b][chain_a]),
            len(scores["unique_res_chain2"][chain_a][chain_b]),
        )
        dist1_min = min(
            len(scores["dist_unique_res_chain2"][chain_b][chain_a]),
            len(scores["dist_unique_res_chain1"][chain_a][chain_b]),
        )
        dist2_min = min(
            len(scores["dist_unique_res_chain1"][chain_b][chain_a]),
            len(scores["dist_unique_res_chain2"][chain_a][chain_b]),
        )
        pdockq2_min = min(pdockq2[chain_b][chain_a], pdockq2[chain_a][chain_b])
        iptm_af_min = min(
            iptm_per_pair[chain_b][chain_a], iptm_per_pair[chain_a][chain_b]
        )

        min_record = _make_score_record(
            chain_a,
            chain_b,
            scores["ipsae_d0res_min"][chain_b][chain_a],
            scores["ipsae_d0chn_min"][chain_b][chain_a],
            scores["ipsae_d0dom_min"][chain_b][chain_a],
            iptm_af_min,
            scores["iptm_d0chn_min"][chain_b][chain_a],
            pdockq[chain_b][chain_a],
            pdockq2_min,
            lis_avg,
            int(scores["n0res_min"][chain_b][chain_a]),
            int(scores["n0chn"][chain_b][chain_a]),
            int(scores["n0dom_min"][chain_b][chain_a]),
            scores["d0res_min"][chain_b][chain_a],
            scores["d0chn"][chain_b][chain_a],
            scores["d0dom_min"][chain_b][chain_a],
            nres1_min,
            nres2_min,
            dist1_min,
            dist2_min,
        )

        entry: dict = {
            "asym": asym_records,
            "max": max_record,
            "min": min_record,
        }
        if lis_metrics is not None and pair in lis_metrics:
            entry["lis_metrics"] = lis_metrics[pair]
        pair_data[pair] = entry

    header: dict = {
        "pae_cutoff": int(pae_cutoff),
        "dist_cutoff": int(dist_cutoff),
    }
    if lis_metrics is not None:
        header["lis_pae_cutoff"] = (
            int(lis_pae_cutoff) if lis_pae_cutoff is not None else 12
        )
        header["lis_cb_cutoff"] = int(lis_cb_cutoff) if lis_cb_cutoff is not None else 8

    return {model_name: {**header, **pair_data}}


def write_json(
    json_path: Path,
    model_stem: str,
    unique_chains: np.ndarray,
    scores: dict,
    pdockq: dict,
    pdockq2: dict,
    lis: dict,
    iptm_per_pair: dict,
    pae_cutoff: float,
    dist_cutoff: float,
    lis_metrics: dict[str, dict] | None = None,
    lis_pae_cutoff: float | None = None,
    lis_cb_cutoff: float | None = None,
) -> None:
    data = scores_to_json(
        model_stem,
        unique_chains,
        scores,
        pdockq,
        pdockq2,
        lis,
        iptm_per_pair,
        pae_cutoff,
        dist_cutoff,
        lis_metrics=lis_metrics,
        lis_pae_cutoff=lis_pae_cutoff,
        lis_cb_cutoff=lis_cb_cutoff,
    )
    with open(json_path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")


_TXT_HEADER = (
    "\nChn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom"
    "  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS"
    "       n0res  n0chn  n0dom   d0res   d0chn   d0dom"
    "  nres1   nres2   dist1   dist2  Model\n"
)
_PML_HEADER = (
    "# Chn1 Chn2  PAE Dist  Type   ipSAE    ipSAE_d0chn ipSAE_d0dom"
    "  ipTM_af  ipTM_d0chn     pDockQ     pDockQ2    LIS"
    "      n0res  n0chn  n0dom   d0res   d0chn   d0dom"
    "  nres1   nres2   dist1   dist2  Model\n"
)


def _iter_pair_lines(
    unique_chains: np.ndarray,
    scores: dict,
    pdockq: dict,
    pdockq2: dict,
    lis: dict,
    iptm_per_pair: dict,
    pae_string: str,
    dist_string: str,
    model_stem: str,
) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (txt_lines, pml_lines) for each chain pair."""
    chainpairs: set[str] = set()
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 < c2:
                chainpairs.add(f"{c1}-{c2}")

    for pair in sorted(chainpairs):
        chain_a, chain_b = pair.split("-")
        txt_lines: list[str] = []
        pml_lines: list[str] = []

        for chain1, chain2 in ((chain_a, chain_b), (chain_b, chain_a)):
            color1 = CHAIN_COLORS.get(chain1, "magenta")
            color2 = CHAIN_COLORS.get(chain2, "marine")

            nres1 = len(scores["unique_res_chain1"][chain1][chain2])
            nres2 = len(scores["unique_res_chain2"][chain1][chain2])
            dist1 = len(scores["dist_unique_res_chain1"][chain1][chain2])
            dist2 = len(scores["dist_unique_res_chain2"][chain1][chain2])

            asym_line = _fmt_summary_line(
                chain1,
                chain2,
                pae_string,
                dist_string,
                "asym",
                scores["ipsae_d0res_asym"][chain1][chain2],
                scores["ipsae_d0chn_asym"][chain1][chain2],
                scores["ipsae_d0dom_asym"][chain1][chain2],
                iptm_per_pair[chain1][chain2],
                scores["iptm_d0chn_asym"][chain1][chain2],
                pdockq[chain1][chain2],
                pdockq2[chain1][chain2],
                lis[chain1][chain2],
                int(scores["n0res"][chain1][chain2]),
                int(scores["n0chn"][chain1][chain2]),
                int(scores["n0dom"][chain1][chain2]),
                scores["d0res"][chain1][chain2],
                scores["d0chn"][chain1][chain2],
                scores["d0dom"][chain1][chain2],
                nres1,
                nres2,
                dist1,
                dist2,
                model_stem,
            )
            txt_lines.append(asym_line)
            pml_lines.append("# " + asym_line)

            if chain1 > chain2:
                nres1_max = max(
                    len(scores["unique_res_chain2"][chain1][chain2]),
                    len(scores["unique_res_chain1"][chain2][chain1]),
                )
                nres2_max = max(
                    len(scores["unique_res_chain1"][chain1][chain2]),
                    len(scores["unique_res_chain2"][chain2][chain1]),
                )
                dist1_max = max(
                    len(scores["dist_unique_res_chain2"][chain1][chain2]),
                    len(scores["dist_unique_res_chain1"][chain2][chain1]),
                )
                dist2_max = max(
                    len(scores["dist_unique_res_chain1"][chain1][chain2]),
                    len(scores["dist_unique_res_chain2"][chain2][chain1]),
                )
                pdockq2_max = max(pdockq2[chain1][chain2], pdockq2[chain2][chain1])
                lis_avg = (lis[chain1][chain2] + lis[chain2][chain1]) / 2.0

                max_line = _fmt_summary_line(
                    chain2,
                    chain1,
                    pae_string,
                    dist_string,
                    "max",
                    scores["ipsae_d0res_max"][chain1][chain2],
                    scores["ipsae_d0chn_max"][chain1][chain2],
                    scores["ipsae_d0dom_max"][chain1][chain2],
                    iptm_per_pair[chain1][chain2],
                    scores["iptm_d0chn_max"][chain1][chain2],
                    pdockq[chain1][chain2],
                    pdockq2_max,
                    lis_avg,
                    int(scores["n0res_max"][chain1][chain2]),
                    int(scores["n0chn"][chain1][chain2]),
                    int(scores["n0dom_max"][chain1][chain2]),
                    scores["d0res_max"][chain1][chain2],
                    scores["d0chn"][chain1][chain2],
                    scores["d0dom_max"][chain1][chain2],
                    nres1_max,
                    nres2_max,
                    dist1_max,
                    dist2_max,
                    model_stem,
                )
                txt_lines.append(max_line)
                pml_lines.append("# " + max_line)

                nres1_min = min(
                    len(scores["unique_res_chain2"][chain1][chain2]),
                    len(scores["unique_res_chain1"][chain2][chain1]),
                )
                nres2_min = min(
                    len(scores["unique_res_chain1"][chain1][chain2]),
                    len(scores["unique_res_chain2"][chain2][chain1]),
                )
                dist1_min = min(
                    len(scores["dist_unique_res_chain2"][chain1][chain2]),
                    len(scores["dist_unique_res_chain1"][chain2][chain1]),
                )
                dist2_min = min(
                    len(scores["dist_unique_res_chain1"][chain1][chain2]),
                    len(scores["dist_unique_res_chain2"][chain2][chain1]),
                )
                pdockq2_min = min(pdockq2[chain1][chain2], pdockq2[chain2][chain1])

                min_line = _fmt_summary_line(
                    chain2,
                    chain1,
                    pae_string,
                    dist_string,
                    "min",
                    scores["ipsae_d0res_min"][chain1][chain2],
                    scores["ipsae_d0chn_min"][chain1][chain2],
                    scores["ipsae_d0dom_min"][chain1][chain2],
                    iptm_per_pair[chain1][chain2],
                    scores["iptm_d0chn_min"][chain1][chain2],
                    pdockq[chain1][chain2],
                    pdockq2_min,
                    lis_avg,
                    int(scores["n0res_min"][chain1][chain2]),
                    int(scores["n0chn"][chain1][chain2]),
                    int(scores["n0dom_min"][chain1][chain2]),
                    scores["d0res_min"][chain1][chain2],
                    scores["d0chn"][chain1][chain2],
                    scores["d0dom_min"][chain1][chain2],
                    nres1_min,
                    nres2_min,
                    dist1_min,
                    dist2_min,
                    model_stem,
                )
                txt_lines.append(min_line)
                pml_lines.append("# " + min_line)

            res1_str = contiguous_ranges(scores["unique_res_chain1"][chain1][chain2])
            res2_str = contiguous_ranges(scores["unique_res_chain2"][chain1][chain2])
            pml_lines.append(
                f"alias color_{chain1}_{chain2}, color gray80, all;"
                f" color {color1}, chain  {chain1} and resi {res1_str};"
                f" color {color2}, chain  {chain2} and resi {res2_str}\n\n"
            )

        txt_lines.append("\n")
        yield txt_lines, pml_lines


def write_pml(
    pml_path: Path,
    unique_chains: np.ndarray,
    scores: dict,
    pdockq: dict,
    pdockq2: dict,
    lis: dict,
    iptm_per_pair: dict,
    pae_string: str,
    dist_string: str,
    model_stem: str,
) -> None:
    with open(pml_path, "w") as pml:
        pml.write(_PML_HEADER)
        for _, pml_lines in _iter_pair_lines(
            unique_chains,
            scores,
            pdockq,
            pdockq2,
            lis,
            iptm_per_pair,
            pae_string,
            dist_string,
            model_stem,
        ):
            pml.writelines(pml_lines)


def write_summary_and_pml(
    txt_path: Path,
    pml_path: Path,
    unique_chains: np.ndarray,
    scores: dict,
    pdockq: dict,
    pdockq2: dict,
    lis: dict,
    iptm_per_pair: dict,
    pae_string: str,
    dist_string: str,
    model_stem: str,
) -> None:
    with open(txt_path, "w") as txt, open(pml_path, "w") as pml:
        txt.write(_TXT_HEADER)
        pml.write(_PML_HEADER)
        for txt_lines, pml_lines in _iter_pair_lines(
            unique_chains,
            scores,
            pdockq,
            pdockq2,
            lis,
            iptm_per_pair,
            pae_string,
            dist_string,
            model_stem,
        ):
            txt.writelines(txt_lines)
            pml.writelines(pml_lines)


# ── Public API ────────────────────────────────────────────────────────────────

_AF2_PAE_GLOB = "*_scores_rank_001_alphafold2_multimer_v3_model_*_seed_*.json"
_AF2_SCORES_MARKER = "_scores_rank_001_"
_AF3_PAE_SUFFIX = "_confidences.json"
_AF3_SUMMARY_SUFFIX = "_summary_confidences.json"


def find_inputs(input_dir: str | Path) -> tuple[Path, Path]:
    """Find PAE and structure files in a directory, auto-detecting AF2 or AF3 format.

    AF2 format:
        PAE:    [prefix]_scores_rank_001_alphafold2_multimer_v3_model_N_seed_NNN.json
        struct: [prefix]_relaxed_rank_001_alphafold2_multimer_v3_model_N_seed_NNN.pdb
                (falls back to _unrelaxed_ if relaxed is absent)

    AF3 format:
        PAE:    [prefix]_confidences.json
        struct: [prefix]_model.cif

    Returns:
        (pae_file, struct_file) as resolved Paths.

    Raises:
        NotADirectoryError: if input_dir is not a directory.
        FileNotFoundError: if required files cannot be found.
        ValueError: if format cannot be determined or files are ambiguous.
    """
    d = Path(input_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    # Try AF2
    af2_pae_files = list(d.glob(_AF2_PAE_GLOB))
    if af2_pae_files:
        if len(af2_pae_files) > 1:
            raise ValueError(
                f"Multiple AF2 rank_001 PAE files found in {d}: {af2_pae_files}"
            )
        pae_file = af2_pae_files[0]
        stem = pae_file.stem
        relaxed_stem = stem.replace(_AF2_SCORES_MARKER, "_relaxed_rank_001_", 1)
        unrelaxed_stem = stem.replace(_AF2_SCORES_MARKER, "_unrelaxed_rank_001_", 1)
        relaxed = d / f"{relaxed_stem}.pdb"
        unrelaxed = d / f"{unrelaxed_stem}.pdb"
        if relaxed.exists():
            return pae_file, relaxed
        if unrelaxed.exists():
            return pae_file, unrelaxed
        raise FileNotFoundError(
            f"No PDB found for '{relaxed_stem}' or '{unrelaxed_stem}' in {d}"
        )

    # Try AF3
    af3_pae_files = [
        f
        for f in d.glob(f"*{_AF3_PAE_SUFFIX}")
        if not f.name.endswith(_AF3_SUMMARY_SUFFIX)
    ]
    if af3_pae_files:
        if len(af3_pae_files) > 1:
            raise ValueError(f"Multiple AF3 PAE files found in {d}: {af3_pae_files}")
        pae_file = af3_pae_files[0]
        prefix = pae_file.name[: -len(_AF3_PAE_SUFFIX)]
        struct_file = d / f"{prefix}_model.cif"
        if not struct_file.exists():
            raise FileNotFoundError(f"Expected CIF file not found: {struct_file}")
        return pae_file, struct_file

    raise ValueError(
        f"Cannot determine format in {d}. "
        f"Expected AF2: {_AF2_PAE_GLOB}, "
        f"or AF3: *{_AF3_PAE_SUFFIX} + *_model.cif"
    )


def _check_colabfold_prefix(d: Path, prefix: str) -> tuple[Path, Path, str] | None:
    """Validate one ColabFold prefix and return its (pae, struct, prefix), or None."""
    if not (d / f"{prefix}_coverage.png").exists():
        return None

    pae_glob = f"{prefix}_scores_rank_001_alphafold2_multimer_v3_model_*_seed_*.json"
    pae_files = list(d.glob(pae_glob))
    if not pae_files:
        logger.warning(f"No PAE file found for prefix '{prefix}' in {d}; skipping")
        return None
    if len(pae_files) > 1:
        logger.warning(
            f"Multiple PAE files for prefix '{prefix}' in {d}: {pae_files}; skipping"
        )
        return None

    pae_file = pae_files[0]
    stem = pae_file.stem
    relaxed_stem = stem.replace(_AF2_SCORES_MARKER, "_relaxed_rank_001_", 1)
    unrelaxed_stem = stem.replace(_AF2_SCORES_MARKER, "_unrelaxed_rank_001_", 1)
    relaxed = d / f"{relaxed_stem}.pdb"
    unrelaxed = d / f"{unrelaxed_stem}.pdb"

    if relaxed.exists():
        return pae_file, relaxed, prefix
    if unrelaxed.exists():
        return pae_file, unrelaxed, prefix

    logger.warning(
        f"No PDB found for prefix '{prefix}' "
        f"(tried '{relaxed_stem}' and '{unrelaxed_stem}'); skipping"
    )
    return None


def find_colabfold_inputs(
    input_dir: str | Path,
) -> list[tuple[Path, Path, str]]:
    """Find all complete ColabFold predictions in a directory.

    A prediction with prefix 'AAA' is considered complete when both
    ``AAA.done.txt`` and ``AAA_coverage.png`` are present in the directory.

    For each complete prefix the function locates the rank-001 PAE JSON and
    the corresponding relaxed (preferred) or unrelaxed PDB file.
    Prefix validation runs in parallel across available CPU cores.

    Returns:
        Sorted list of ``(pae_file, struct_file, prefix)`` tuples.
        Returns an empty list when no valid predictions are found.

    Raises:
        NotADirectoryError: if *input_dir* is not a directory.
    """
    d = Path(input_dir)
    if not d.is_dir():
        raise NotADirectoryError(f"Not a directory: {d}")

    prefixes = [
        name[: -len(".done.txt")]
        for done_file in sorted(d.glob("*.done.txt"))
        if (name := done_file.name).endswith(".done.txt")
    ]
    if not prefixes:
        return []

    workers = min(len(prefixes), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_check_colabfold_prefix, d, p) for p in prefixes]
        raw = [f.result() for f in futures]

    return [r for r in raw if r is not None]


def run_ipsae(
    pae_file: str | Path,
    struct_file: str | Path,
    pae_cutoff: float,
    dist_cutoff: float,
    output_json: bool = False,
    model_name: str | None = None,
    lis_pae_cutoff: float = 12.0,
    lis_cb_cutoff: float = 8.0,
) -> dict[str, Path]:
    """Calculate ipSAE and related scores.

    Writes three output files next to the structure file and returns their paths.

    Args:
        pae_file: PAE file (.json for AF2/AF3, .npz for Boltz).
        struct_file: Structure file (.pdb for AF2/Boltz, .cif for AF3/Boltz).
        pae_cutoff: PAE threshold in Angstroms.
        dist_cutoff: Distance threshold in Angstroms.
        output_json: If True, write JSON instead of txt summary.
        model_name: Top-level key used in JSON output. Defaults to the struct
            file stem when None.
        lis_pae_cutoff: PAE cutoff used for the LIS-family metrics (default 12).
        lis_cb_cutoff: Cβ contact cutoff (Å) for the LIS-family metrics
            (default 8).

    Returns:
        Dict with keys "txt"/"json", "byres", "pml" pointing to the written files.
    """
    pae_path = Path(pae_file)
    struct_path = Path(struct_file)

    pae_str = str(pae_path)
    struct_str = str(struct_path)

    is_cif = ".cif" in struct_str
    is_af2 = ".pdb" in struct_str and pae_str.endswith(".json")
    is_af3 = ".cif" in struct_str and pae_str.endswith(".json")
    is_boltz = pae_str.endswith(".npz")

    if not (is_af2 or is_af3 or is_boltz):
        raise ValueError(f"Unrecognised file combination: {pae_path}, {struct_path}")

    pae_str_int = str(int(pae_cutoff)).zfill(2)
    dist_str_int = str(int(dist_cutoff)).zfill(2)
    stem = struct_path.with_suffix("")
    path_stem = f"{stem}_{pae_str_int}_{dist_str_int}"

    txt_path = Path(path_stem + (".json" if output_json else ".txt"))
    byres_path = Path(path_stem + "_byres.txt")
    pml_path = Path(path_stem + ".pml")

    residues, cb_residues, raw_chains, token_mask = parse_structure(struct_path, is_cif)
    numres = len(residues)
    chains = np.array(raw_chains)

    _, first_idx = np.unique(chains, return_index=True)
    unique_chains = chains[np.sort(first_idx)]
    token_array = np.array(token_mask)
    ntokens = int(np.sum(token_array))

    ca_atom_num = np.array([r["atom_num"] - 1 for r in residues])
    cb_atom_num = np.array([r["atom_num"] - 1 for r in cb_residues])
    cb_coords = np.array([r["coor"] for r in cb_residues])
    residue_types = np.array([r["res"] for r in residues])

    distances = np.sqrt(
        ((cb_coords[:, np.newaxis, :] - cb_coords[np.newaxis, :, :]) ** 2).sum(axis=2)
    )

    chain_type_map = classify_chains(chains, residue_types)
    chain_pair_type = _pair_zeros(list(unique_chains))
    for c1 in unique_chains:
        for c2 in unique_chains:
            if c1 == c2:
                continue
            if (
                chain_type_map[c1] == "nucleic_acid"
                or chain_type_map[c2] == "nucleic_acid"
            ):
                chain_pair_type[c1][c2] = "nucleic_acid"
            else:
                chain_pair_type[c1][c2] = "protein"

    if is_af2:
        plddt, pae_matrix, iptm_scalar, _ = load_af2_data(pae_path, numres)
        cb_plddt = plddt.copy()
        iptm_per_pair = {
            c1: {c2: iptm_scalar for c2 in unique_chains if c1 != c2}
            for c1 in unique_chains
        }
    elif is_af3:
        plddt, cb_plddt, pae_matrix, iptm_per_pair = load_af3_data(
            pae_path, ca_atom_num, cb_atom_num, token_array, unique_chains, numres
        )
    else:
        plddt, pae_matrix, iptm_per_pair = load_boltz_data(
            pae_path, token_array, unique_chains, ntokens
        )
        cb_plddt = plddt.copy()

    pdockq_res, pdockq = compute_pdockq(
        unique_chains, chains, distances, cb_plddt, numres
    )
    pdockq2 = compute_pdockq2(
        unique_chains, chains, distances, cb_plddt, pae_matrix, pdockq_res, numres
    )
    lis = compute_lis(unique_chains, chains, pae_matrix)
    scores = compute_ipsae(
        unique_chains,
        chains,
        pae_matrix,
        distances,
        residues,
        chain_pair_type,
        pae_cutoff,
        dist_cutoff,
    )
    lis_metrics = compute_lis_metrics(
        unique_chains,
        chains,
        residues,
        pae_matrix,
        distances,
        pae_cutoff_lis=lis_pae_cutoff,
        cb_cutoff_lis=lis_cb_cutoff,
    )

    write_byres(byres_path, unique_chains, chains, residues, plddt, scores)
    json_model_stem = model_name if model_name is not None else str(stem)
    if output_json:
        write_json(
            txt_path,
            json_model_stem,
            unique_chains,
            scores,
            pdockq,
            pdockq2,
            lis,
            iptm_per_pair,
            pae_cutoff,
            dist_cutoff,
            lis_metrics=lis_metrics,
            lis_pae_cutoff=lis_pae_cutoff,
            lis_cb_cutoff=lis_cb_cutoff,
        )
        write_pml(
            pml_path,
            unique_chains,
            scores,
            pdockq,
            pdockq2,
            lis,
            iptm_per_pair,
            pae_str_int,
            dist_str_int,
            str(stem),
        )
        return {"json": txt_path, "byres": byres_path, "pml": pml_path}

    write_summary_and_pml(
        txt_path,
        pml_path,
        unique_chains,
        scores,
        pdockq,
        pdockq2,
        lis,
        iptm_per_pair,
        pae_str_int,
        dist_str_int,
        str(stem),
    )

    return {"txt": txt_path, "byres": byres_path, "pml": pml_path}


def main() -> None:
    parser = ArgumentParser(
        description="Calculate ipSAE scores for AlphaFold2/3 and Boltz models."
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        help="Input directory; auto-detects AF2/AF3 PAE and structure files",
    )
    parser.add_argument(
        "-p",
        "--pae_file",
        help="PAE file (.json for AF2/AF3, .npz for Boltz)",
    )
    parser.add_argument(
        "-s",
        "--struct_file",
        help="Structure file (.pdb for AF2/Boltz, .cif for AF3/Boltz)",
    )
    parser.add_argument(
        "-pc", "--pae_cutoff", type=float, help="PAE cutoff in Angstroms", default=10.0
    )
    parser.add_argument(
        "-dc",
        "--dist_cutoff",
        type=float,
        help="Distance cutoff in Angstroms",
        default=10.0,
    )
    parser.add_argument(
        "--lis_pae_cutoff",
        type=float,
        default=12.0,
        help="PAE cutoff (Å) for LIS-family metrics (LIS/cLIS/iLIS/LIA/cLIA/iLIA/iLISA/actifpTM). Default: 12",
    )
    parser.add_argument(
        "--lis_cb_cutoff",
        type=float,
        default=8.0,
        help="Cβ distance cutoff (Å) for LIS-family metrics. Default: 8",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of txt summary",
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

    if args.input_dir:
        input_dir = Path(args.input_dir)
        batch = find_colabfold_inputs(input_dir)
        if batch:
            for pae_file, struct_file, prefix in batch:
                paths = run_ipsae(
                    pae_file,
                    struct_file,
                    args.pae_cutoff,
                    args.dist_cutoff,
                    output_json=args.json,
                    model_name=prefix,
                    lis_pae_cutoff=args.lis_pae_cutoff,
                    lis_cb_cutoff=args.lis_cb_cutoff,
                )
                for path in paths.values():
                    logger.debug(f"Written: {path}")
            return
        pae_file, struct_file = find_inputs(input_dir)
        model_name: str | None = input_dir.name
    elif args.pae_file and args.struct_file:
        pae_file, struct_file = Path(args.pae_file), Path(args.struct_file)
        model_name = None
    else:
        parser.error(
            "Specify either -i INPUT_DIR or both -p PAE_FILE and -s STRUCT_FILE"
        )

    paths = run_ipsae(
        pae_file,
        struct_file,
        args.pae_cutoff,
        args.dist_cutoff,
        output_json=args.json,
        model_name=model_name,
        lis_pae_cutoff=args.lis_pae_cutoff,
        lis_cb_cutoff=args.lis_cb_cutoff,
    )
    for path in paths.values():
        logger.debug(f"Written: {path}")


if __name__ == "__main__":
    main()
