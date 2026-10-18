"""PDB to mmCIF format converter using gemmi.

Replicates the behavior of `maxit -input <pdb> -output <cif> -o 1`.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import re
from collections.abc import Mapping
from pathlib import Path

import gemmi
from loguru import logger

from alphafold3tools.ciftoseqres import cif_to_seqres
from alphafold3tools.log import log_setup


def pdb_to_cif(
    input_path: str,
    output_path: str,
    default_pdb_id: str = "xxxx",
    pdbid: str | None = None,
    write_seqres: bool = True,
) -> None:
    """Convert a PDB file to mmCIF format.

    Args:
        input_path: Path to the input PDB file. A gzip-compressed file is read
            if the name ends with ".gz".
        output_path: Path to the output mmCIF file.
        default_pdb_id: The default PDB ID to use in the output file.
            Defaults to "xxxx" if not provided.
        pdbid: Backward-compatible alias for default_pdb_id.
        write_seqres: Whether to also write a pdb_seqres.txt file with sequences extracted from the CIF.
    """
    if pdbid is not None:
        default_pdb_id = pdbid

    if Path(input_path).suffix.lower() == ".gz":
        with gzip.open(input_path, "rt") as f:
            pdb_text = f.read()
    else:
        pdb_text = Path(input_path).read_text()
    pdb_lines = pdb_text.splitlines()

    st = gemmi.read_pdb_string(pdb_text)
    st.setup_entities()
    _fill_missing_entity_sequences(st)
    st.assign_label_seq_id()

    pdb_id = st.info["_entry.id"] if "_entry.id" in st.info else default_pdb_id

    # Remap entity names to numeric IDs and subchains to A, B, C, D …
    _remap_entities_and_subchains(st)

    # Generate base mmCIF with all output groups enabled
    groups = gemmi.MmcifOutputGroups(True)
    doc = st.make_mmcif_document(groups)
    block = doc.sole_block()

    # Read all existing categories produced by gemmi
    cats: dict[str, dict[str, list]] = {}
    for cat_name in block.get_mmcif_category_names():
        cats[cat_name] = block.get_mmcif_category(cat_name)

    # --- Fix / augment categories ---
    _fix_atom_site(cats)
    _fix_struct_asym(cats)
    _fix_chem_comp(cats, st)
    _fix_refine_ls_restr(cats)
    _fix_pdbx_database_status(cats, pdb_id)
    _fix_struct(cats, pdb_id)
    _fix_struct_keywords(cats, pdb_id)
    _fix_cell(cats, pdb_id)
    _fix_symmetry(cats, pdb_id)
    _fix_entity_poly(cats)
    _fix_entity_poly_order(cats)
    _fix_struct_ref_columns(cats)
    _fix_struct_ref_seq(cats, pdb_id)
    _fix_audit_author(cats)
    _fix_struct_conf(cats, pdb_lines)
    _fix_struct_conf_type(cats)
    _fix_struct_sheet_columns(cats)
    _fix_struct_sheet_order_columns(cats)
    _fix_struct_sheet_range(cats)
    _fix_struct_mon_prot_cis(cats)
    _fix_refine(cats, pdb_id, pdb_lines)
    _fix_reflns(cats, pdb_lines)
    _fix_reflns_shell(cats)
    _fix_exptl_crystal_grow(cats, pdb_lines)
    _fix_exptl_crystal(cats, pdb_lines)
    _fix_pdbx_struct_assembly(cats)
    _fix_pdbx_struct_oper_list(cats)
    _fix_software(cats, pdb_lines)
    _fix_diffrn_radiation(cats, pdb_lines)
    _fix_diffrn_source(cats)
    _fix_diffrn(cats)
    _fix_diffrn_detector(cats)
    _fix_pdbx_struct_sheet_hbond(cats)
    _fix_entity_columns(cats, st, pdb_lines)
    _fix_refine_ls_restr_from_pdb(cats, pdb_lines)

    # --- Add missing categories ---
    _add_database_2(cats, pdb_id)
    # add a fake revision date (1970-01-01) since they lack revision history
    _add_revision_history(cats, pdb_lines, prefix_date=datetime.datetime(1970, 1, 1))
    _add_citation_and_authors(cats, pdb_lines)
    _add_entity_src_gen(cats, pdb_lines)
    _add_struct_ref_seq_dif(cats, pdb_lines, pdb_id)
    _add_pdbx_poly_seq_scheme(cats, st)
    _add_pdbx_nonpoly_scheme(cats, st)
    _add_pdbx_entity_nonpoly(cats, st)
    _add_database_PDB_rev(cats, pdb_lines, pdb_id)
    _add_atom_sites(cats, pdb_lines, pdb_id)
    _add_database_PDB_matrix(cats, pdb_lines, pdb_id)
    _add_diffrn_radiation_wavelength(cats, pdb_lines)
    _add_refine_hist(cats, pdb_lines, st)
    _add_refine_ls_shell(cats, pdb_lines)
    _add_pdbx_unobs_or_zero_occ_residues(cats, pdb_lines, st)
    _add_pdbx_unobs_or_zero_occ_atoms(cats, pdb_lines, st)
    _fix_struct_conn(cats)
    _add_pdbx_validate_close_contact(cats, pdb_lines)
    _add_pdbx_validate_torsion(cats, pdb_lines)
    _fix_atom_type(cats, st)

    # --- Write in maxit category order ---
    _write_output(cats, pdb_id, output_path)
    if write_seqres:
        seqres_output_path = Path(output_path).with_name("pdb_seqres.txt")
        cif_to_seqres(output_path, str(seqres_output_path))


# ---------------------------------------------------------------------------
# Entity / subchain remapping
# ---------------------------------------------------------------------------


def _fill_missing_entity_sequences(st: gemmi.Structure) -> None:
    """Fill empty polymer entity sequences from the ATOM records.

    PDB files without SEQRES records (e.g. predicted models) leave
    ``Entity.full_sequence`` empty, which results in ``?`` for
    ``_entity_poly.pdbx_seq_one_letter_code`` and ``.`` for
    ``_atom_site.label_seq_id``. Use the residues of the first subchain
    of each such entity as its full sequence.
    """
    if len(st) == 0:
        return
    model = st[0]
    for entity in st.entities:
        if entity.entity_type != gemmi.EntityType.Polymer or entity.full_sequence:
            continue
        if not entity.subchains:
            continue
        subchain = model.get_subchain(entity.subchains[0])
        entity.full_sequence = [res.name for res in subchain]
        logger.info(
            f"No SEQRES for entity {entity.name}; using {len(subchain)} residues from ATOM records as its sequence."
        )


def _remap_entities_and_subchains(st: gemmi.Structure) -> None:
    """Remap entity names to 1,2,... and subchain IDs to A,B,C,D,..."""
    subchain_map: dict[str, str] = {}
    next_label = ord("A")

    for model in st:
        for chain in model:
            for res in chain:
                old = res.subchain
                if old and old not in subchain_map:
                    subchain_map[old] = chr(next_label)
                    next_label += 1

    for model in st:
        for chain in model:
            for res in chain:
                if res.subchain in subchain_map:
                    res.subchain = subchain_map[res.subchain]

    for i, ent in enumerate(st.entities, 1):
        ent.subchains = [subchain_map.get(s, s) for s in ent.subchains]
        ent.name = str(i)


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------


def _get(cats: dict, cat: str, key: str) -> list:
    """Get a column from a category, returning [] if missing."""
    return cats.get(cat, {}).get(key, [])


def _nrows(cats: dict, cat: str) -> int:
    d = cats.get(cat, {})
    if not d:
        return 0
    return len(next(iter(d.values())))


def _first(cats: dict, cat: str, key: str) -> str:
    vals = _get(cats, cat, key)
    return vals[0] if vals else "?"


# ---------------------------------------------------------------------------
# Fixes to gemmi-generated categories
# ---------------------------------------------------------------------------


def _fmt_float(v: object, decimals: int) -> str:
    """Format a numeric string to a fixed number of decimal places.

    Handles both scalar values and single-element lists.
    """
    # Unwrap single-element lists
    if isinstance(v, list):
        v = v[0] if v else "?"
    if v is None or v == "?" or v == ".":
        return str(v) if v is not None else "?"
    try:
        f = float(str(v))
        if f in (-float("inf"), float("inf")):
            return str(v)
        return f"{f:.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def _fix_atom_site(cats: dict) -> None:
    """Add auth_comp_id and auth_atom_id columns; format numeric columns."""
    d = cats.get("_atom_site.", {})
    if not d:
        return
    # Format Cartn_x/y/z to 3 decimal places, occupancy and B_iso_or_equiv to 2
    for col, decimals in (
        ("Cartn_x", 3),
        ("Cartn_y", 3),
        ("Cartn_z", 3),
        ("occupancy", 2),
        ("B_iso_or_equiv", 2),
    ):
        if col in d:
            d[col] = [_fmt_float(v, decimals) for v in d[col]]
    # Insert auth_comp_id after auth_seq_id, auth_atom_id after auth_asym_id
    new_d: dict[str, list] = {}
    for key, vals in d.items():
        new_d[key] = vals
        if key == "auth_seq_id":
            new_d["auth_comp_id"] = list(d.get("label_comp_id", vals))
        elif key == "auth_asym_id":
            new_d["auth_atom_id"] = list(d.get("label_atom_id", vals))
    cats["_atom_site."] = new_d


def _fix_struct_asym(cats: dict) -> None:
    """Add pdbx_blank_PDB_chainid_flag, pdbx_modified, details columns; fix order."""
    d = cats.get("_struct_asym.", {})
    if not d:
        return
    n = _nrows(cats, "_struct_asym.")
    d.setdefault("pdbx_blank_PDB_chainid_flag", ["N"] * n)
    d.setdefault("pdbx_modified", ["N"] * n)
    d.setdefault("details", ["?"] * n)
    # Reorder to match maxit: id, pdbx_blank_PDB_chainid_flag, pdbx_modified, entity_id, details
    desired = [
        "id",
        "pdbx_blank_PDB_chainid_flag",
        "pdbx_modified",
        "entity_id",
        "details",
    ]
    cats["_struct_asym."] = {k: d.get(k, ["?"] * n) for k in desired}


def _fix_chem_comp(cats: dict, st: gemmi.Structure) -> None:
    """Replace sparse chem_comp with fully populated table."""
    CHEM_COMP: dict[str, tuple] = {
        "ALA": ("L-peptide linking", "y", "ALANINE", "C3 H7 N O2", "89.093"),
        "ARG": ("L-peptide linking", "y", "ARGININE", "C6 H15 N4 O2 1", "175.209"),
        "ASN": ("L-peptide linking", "y", "ASPARAGINE", "C4 H8 N2 O3", "132.118"),
        "ASP": ("L-peptide linking", "y", "ASPARTIC ACID", "C4 H7 N O4", "133.103"),
        "CYS": ("L-peptide linking", "y", "CYSTEINE", "C3 H7 N O2 S", "121.158"),
        "GLN": ("L-peptide linking", "y", "GLUTAMINE", "C5 H10 N2 O3", "146.144"),
        "GLU": ("L-peptide linking", "y", "GLUTAMIC ACID", "C5 H9 N O4", "147.129"),
        "GLY": ("peptide linking", "y", "GLYCINE", "C2 H5 N O2", "75.067"),
        "HIS": ("L-peptide linking", "y", "HISTIDINE", "C6 H10 N3 O2 1", "156.162"),
        "FE": ("non-polymer", ".", "FE (III) ION", "Fe 3", "55.845"),
        "HOH": ("non-polymer", ".", "WATER", "H2 O", "18.015"),
        "OGA": ("non-polymer", ".", "N-OXALYLGLYCINE", "C4 H5 N O5", "147.086"),
        "ILE": ("L-peptide linking", "y", "ISOLEUCINE", "C6 H13 N O2", "131.173"),
        "LEU": ("L-peptide linking", "y", "LEUCINE", "C6 H13 N O2", "131.173"),
        "LYS": ("L-peptide linking", "y", "LYSINE", "C6 H15 N2 O2 1", "147.195"),
        "MET": ("L-peptide linking", "y", "METHIONINE", "C5 H11 N O2 S", "149.211"),
        "PHE": ("L-peptide linking", "y", "PHENYLALANINE", "C9 H11 N O2", "165.189"),
        "PRO": ("L-peptide linking", "y", "PROLINE", "C5 H9 N O2", "115.130"),
        "SER": ("L-peptide linking", "y", "SERINE", "C3 H7 N O3", "105.093"),
        "THR": ("L-peptide linking", "y", "THREONINE", "C4 H9 N O3", "119.119"),
        "TRP": ("L-peptide linking", "y", "TRYPTOPHAN", "C11 H12 N2 O2", "204.225"),
        "TYR": ("L-peptide linking", "y", "TYROSINE", "C9 H11 N O3", "181.189"),
        "VAL": ("L-peptide linking", "y", "VALINE", "C5 H11 N O2", "117.146"),
    }

    comp_ids: set[str] = set()
    for model in st:
        for chain in model:
            for res in chain:
                comp_ids.add(res.name)
        break

    ids, types, flags, names, syns, formulas, fws = [], [], [], [], [], [], []
    for cid in sorted(comp_ids):
        if cid in CHEM_COMP:
            ct, fl, nm, fm, fw = CHEM_COMP[cid]
        else:
            ct, fl, nm, fm, fw = "?", "?", "?", "?", "?"
        ids.append(cid)
        types.append(ct)
        flags.append(fl)
        names.append(nm)
        syns.append("?")
        formulas.append(fm)
        fws.append(fw)

    cats["_chem_comp."] = {
        "id": ids,
        "type": types,
        "mon_nstd_flag": flags,
        "name": names,
        "pdbx_synonyms": syns,
        "formula": formulas,
        "formula_weight": fws,
    }


def _fix_refine_ls_restr(cats: dict) -> None:
    """Reorder refine_ls_restr columns to match maxit order."""
    d = cats.get("_refine_ls_restr.", {})
    if not d or not _nrows(cats, "_refine_ls_restr."):
        return
    desired = [
        "type",
        "dev_ideal",
        "dev_ideal_target",
        "weight",
        "number",
        "pdbx_refine_id",
        "pdbx_restraint_function",
    ]
    cats["_refine_ls_restr."] = {k: d.get(k, ["?"] * _nrows(cats, "_refine_ls_restr.")) for k in desired}


def _fix_pdbx_database_status(cats: dict, pdb_id: str) -> None:
    # Rebuild with maxit field order; override any gemmi-generated values
    cats["_pdbx_database_status."] = {
        "entry_id": pdb_id,
        "status_code": ".",
        "status_code_sf": "?",
        "status_code_mr": "?",
        "status_code_cs": "?",
        "recvd_initial_deposition_date": "?",
        "status_code_nmr_data": "?",
        "deposit_site": "?",
        "process_site": "?",
        "SG_entry": "?",
        "pdb_format_compatible": "Y",
        "methods_development_category": "?",
    }


def _fix_struct(cats: dict, pdb_id: str) -> None:
    d = cats.setdefault("_struct.", {})
    d.setdefault("entry_id", pdb_id)
    d.setdefault("pdbx_model_details", "?")
    d.setdefault("pdbx_CASP_flag", "?")
    d.setdefault("pdbx_model_type_details", "?")


def _fix_struct_keywords(cats: dict, pdb_id: str) -> None:
    d = cats.setdefault("_struct_keywords.", {})
    d.setdefault("entry_id", pdb_id)


def _fix_cell(cats: dict, pdb_id: str) -> None:
    d = cats.setdefault("_cell.", {})
    d.setdefault("entry_id", pdb_id)
    d.setdefault("pdbx_unique_axis", "?")
    # Format cell lengths to 3 decimal places, angles to 2 decimal places
    # Values from gemmi may be lists; unwrap and reformat as scalars
    for col in ("length_a", "length_b", "length_c"):
        if col in d:
            v = d[col]
            raw = v[0] if isinstance(v, list) else v
            d[col] = _fmt_float(raw, 3)
    for col in ("angle_alpha", "angle_beta", "angle_gamma"):
        if col in d:
            v = d[col]
            raw = v[0] if isinstance(v, list) else v
            d[col] = _fmt_float(raw, 2)


def _fix_symmetry(cats: dict, pdb_id: str) -> None:
    d = cats.get("_symmetry.", {})
    # Rebuild with maxit field order
    cats["_symmetry."] = {
        "entry_id": pdb_id,
        "space_group_name_H-M": d.get("space_group_name_H-M", "?"),
        "pdbx_full_space_group_name_H-M": "?",
        "cell_setting": "?",
        "Int_Tables_number": "?",
    }


def _fix_entity_poly(cats: dict) -> None:
    """Add missing entity_poly fields and fix entity_poly_seq.hetero."""
    d = cats.get("_entity_poly.", {})
    if not d:
        return
    n = _nrows(cats, "_entity_poly.")
    d.setdefault("nstd_linkage", ["no"] * n)
    d.setdefault("nstd_monomer", ["no"] * n)
    d.setdefault(
        "pdbx_seq_one_letter_code_can",
        list(d.get("pdbx_seq_one_letter_code", ["?"] * n)),
    )
    d.setdefault("pdbx_target_identifier", ["?"] * n)

    # Fix entity_poly_seq.hetero: should be "n" not "?"
    ps = cats.get("_entity_poly_seq.", {})
    if ps:
        m = len(ps.get("entity_id", []))
        ps["hetero"] = ["n"] * m


def _fix_struct_ref_seq(cats: dict, pdb_id: str) -> None:
    """Fix struct_ref_seq: use label_seq for align positions, ? for auth positions."""
    d = cats.get("_struct_ref_seq.", {})
    if not d:
        return
    n = _nrows(cats, "_struct_ref_seq.")

    desired_order = [
        "align_id",
        "ref_id",
        "pdbx_PDB_id_code",
        "pdbx_strand_id",
        "seq_align_beg",
        "pdbx_seq_align_beg_ins_code",
        "seq_align_end",
        "pdbx_seq_align_end_ins_code",
        "pdbx_db_accession",
        "db_align_beg",
        "pdbx_db_align_beg_ins_code",
        "db_align_end",
        "pdbx_db_align_end_ins_code",
        "pdbx_auth_seq_align_beg",
        "pdbx_auth_seq_align_end",
    ]
    new_d: dict[str, list] = {}
    for k in desired_order:
        new_d[k] = d.get(k, ["?"] * n)
    if not any(v != "?" for v in new_d.get("pdbx_PDB_id_code", [])):
        new_d["pdbx_PDB_id_code"] = [pdb_id] * n
    # seq_align_beg/end = db_align_beg/end (maxit convention)
    new_d["seq_align_beg"] = list(new_d.get("db_align_beg", ["?"] * n))
    new_d["seq_align_end"] = list(new_d.get("db_align_end", ["?"] * n))
    new_d["pdbx_auth_seq_align_beg"] = ["?"] * n
    new_d["pdbx_auth_seq_align_end"] = ["?"] * n
    # maxit: all struct_ref_seq rows reference the same single struct_ref (ref_id = "1")
    new_d["ref_id"] = ["1"] * n
    cats["_struct_ref_seq."] = new_d


def _seq_wrap(seq: str) -> str:
    """Wrap a sequence at 80 chars with embedded newlines (for loop/set_mmcif_category)."""
    if seq in ("?", "."):
        return seq
    lines = [seq[i : i + 80] for i in range(0, len(seq), 80)]
    return "\n".join(lines)


def _seq_to_semicolon_block(seq: object) -> object:
    """Wrap a sequence string in a CIF semicolon block for set_pair (single-row)."""
    if isinstance(seq, list):
        return [_seq_to_semicolon_block(s) for s in seq]
    s = str(seq) if seq is not None else "?"
    if s in ("?", "."):
        return s
    return ";" + _seq_wrap(s) + "\n;"


def _fix_entity_poly_order(cats: dict) -> None:
    """Reorder entity_poly columns to match maxit."""
    d = cats.get("_entity_poly.", {})
    if not d:
        return
    n = _nrows(cats, "_entity_poly.")
    desired = [
        "entity_id",
        "type",
        "nstd_linkage",
        "nstd_monomer",
        "pdbx_seq_one_letter_code",
        "pdbx_seq_one_letter_code_can",
        "pdbx_strand_id",
        "pdbx_target_identifier",
    ]
    result = {k: d.get(k, ["?"] * n) for k in desired if k in d}
    # Wrap sequences: single-row uses semicolon block; multi-row uses embedded newlines
    for seq_key in ("pdbx_seq_one_letter_code", "pdbx_seq_one_letter_code_can"):
        if seq_key not in result:
            continue
        val = result[seq_key]
        if n == 1:
            # set_pair path: explicit semicolon block
            result[seq_key] = _seq_to_semicolon_block(val)
        else:
            # set_mmcif_category path: embedded newlines (gemmi adds semicolons)
            if isinstance(val, list):
                result[seq_key] = [_seq_wrap(v) if v not in ("?", ".") else v for v in val]
            else:
                result[seq_key] = _seq_wrap(str(val))
    cats["_entity_poly."] = result


# Residue masses derived from wwPDB CCD formula_weight values (chem_comp_fw - 18.015).
# Charged forms used for ARG, HIS, LYS (protonated at physiological pH).
_AA_RESIDUE_MW = {
    "ALA": 71.078,
    "ARG": 157.194,
    "ASN": 114.103,
    "ASP": 115.088,
    "CYS": 103.143,
    "GLN": 128.129,
    "GLU": 129.114,
    "GLY": 57.052,
    "HIS": 138.147,
    "ILE": 113.158,
    "LEU": 113.158,
    "LYS": 129.180,
    "MET": 131.196,
    "PHE": 147.174,
    "PRO": 97.115,
    "SER": 87.078,
    "THR": 101.104,
    "TRP": 186.210,
    "TYR": 163.174,
    "VAL": 99.131,
}


def _parse_compnd(pdb_lines: list[str]) -> dict[str, str]:
    """Return mol_id → MOLECULE description from COMPND records."""
    mol_id = "1"
    result: dict[str, str] = {}
    buf = ""
    for line in pdb_lines:
        if not line.startswith("COMPND"):
            continue
        text = line[10:].strip()
        buf += " " + text
    # Parse key:value; entries
    for item in buf.split(";"):
        item = item.strip()
        if ":" in item:
            key, _, val = item.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if key == "MOL_ID":
                mol_id = val
            elif key == "MOLECULE":
                result[mol_id] = val
    return result


def _calc_polymer_mw(entity: gemmi.Entity) -> str:
    """Calculate molecular weight of a polymer entity from its full sequence."""
    total = 18.015  # water for termini
    for mon in entity.full_sequence:
        # gemmi stores one-letter or three-letter codes
        code = str(mon)
        if len(code) == 1:
            # convert one-letter to three-letter
            one_to_three = {
                "A": "ALA",
                "R": "ARG",
                "N": "ASN",
                "D": "ASP",
                "C": "CYS",
                "Q": "GLN",
                "E": "GLU",
                "G": "GLY",
                "H": "HIS",
                "I": "ILE",
                "L": "LEU",
                "K": "LYS",
                "M": "MET",
                "F": "PHE",
                "P": "PRO",
                "S": "SER",
                "T": "THR",
                "W": "TRP",
                "Y": "TYR",
                "V": "VAL",
            }
            code = one_to_three.get(code, "")
        total += _AA_RESIDUE_MW.get(code, 0.0)
    return f"{total:.3f}"


def _parse_hetnam(pdb_lines: list[str]) -> dict[str, str]:
    """Parse HETNAM records → {res_code: description}."""
    hetnam: dict[str, str] = {}
    for line in pdb_lines:
        if not line.startswith("HETNAM"):
            continue
        code = line[11:14].strip()
        text = line[15:70].rstrip()
        if code:
            hetnam[code] = (hetnam.get(code, "") + text).strip()
    return hetnam


def _parse_formul(pdb_lines: list[str]) -> dict[str, tuple[int, str]]:
    """Parse FORMUL records → {res_code: (count, formula)}."""
    formul: dict[str, tuple[int, str]] = {}
    for line in pdb_lines:
        if not line.startswith("FORMUL"):
            continue
        # cols 8-9: component number, 12-14: residue, 19-70: formula
        code = line[12:15].strip()
        raw = line[19:70].rstrip()
        # Count prefix like "3(FE 3+)" → count=3, formula="FE 3+"
        m = re.match(r"[*]?(\d+)\((.+)\)", raw)
        if m:
            count = int(m.group(1))
            formula = m.group(2).strip()
        else:
            count = 1
            formula = raw.strip().lstrip("*")
        if code:
            formul[code] = (count, formula)
    return formul


# Known formula weights for common non-polymer ligands
_LIGAND_FORMULA_WEIGHT: dict[str, str] = {
    "FE": "55.845",
    "ZN": "65.38",
    "MG": "24.305",
    "CA": "40.078",
    "MN": "54.938",
    "CU": "63.546",
    "NI": "58.693",
    "CO": "58.933",
    "OGA": "147.086",
    "SO4": "96.063",
    "PO4": "94.971",
    "GOL": "92.094",
    "EDO": "62.068",
    "PEG": "420.491",
}


def _fix_entity_columns(cats: dict, st: gemmi.Structure, pdb_lines: list[str]) -> None:
    """Add missing columns to _entity category."""
    d = cats.get("_entity.", {})
    if not d:
        return
    n = _nrows(cats, "_entity.")
    ids = d.get("id", ["?"] * n)
    id_list = ids if isinstance(ids, list) else [ids]

    # Parse COMPND for molecule descriptions
    compnd = _parse_compnd(pdb_lines)
    # Parse HETNAM/FORMUL for non-polymer info
    hetnam = _parse_hetnam(pdb_lines)
    formul = _parse_formul(pdb_lines)

    # Build entity counts (number of subchains per entity)
    entity_map = {ent.name: ent for ent in st.entities}
    # Count water residues
    water_total = sum(
        1 for model in st for chain in model for res in chain if res.entity_type == gemmi.EntityType.Water
    )

    # Map entity_id → residue code for non-polymers using _atom_site data
    nonpoly_code: dict[str, str] = {}
    nonpoly_count: dict[str, int] = {}
    asite = cats.get("_atom_site.", {})
    a_eids = asite.get("label_entity_id", [])
    a_comps = asite.get("label_comp_id", [])
    for eid_val, comp_val in zip(a_eids, a_comps, strict=False):
        eid_s = str(eid_val)
        if eid_s not in nonpoly_code:
            nonpoly_code[eid_s] = str(comp_val)
        nonpoly_count[eid_s] = nonpoly_count.get(eid_s, 0) + 1

    src_methods = []
    descriptions = []
    formula_weights = []
    num_molecules = []

    for eid in id_list:
        ent = entity_map.get(str(eid))
        if ent and ent.entity_type == gemmi.EntityType.Polymer:
            src = "man"
            desc = compnd.get(str(eid), "?")
            fw = _calc_polymer_mw(ent)
            nmol = str(len(ent.subchains))
        elif ent and ent.entity_type == gemmi.EntityType.Water:
            src = "nat"
            desc = "water"
            fw = "18.015"
            nmol = str(water_total)
        else:
            src = "syn"
            rcode = nonpoly_code.get(str(eid), "")
            desc = hetnam.get(rcode, "?") if rcode else "?"
            fw = _LIGAND_FORMULA_WEIGHT.get(rcode, "?") if rcode else "?"
            # Count from FORMUL takes priority (handles stoichiometry like "3(FE 3+)")
            if rcode and rcode in formul:
                nmol = str(formul[rcode][0])
            else:
                nmol = str(nonpoly_count.get(str(eid), 1)) if str(eid) in nonpoly_count else "?"
        src_methods.append(src)
        descriptions.append(desc)
        formula_weights.append(fw)
        num_molecules.append(nmol)

    d["src_method"] = src_methods
    d["pdbx_description"] = descriptions
    d["formula_weight"] = formula_weights
    d["pdbx_number_of_molecules"] = num_molecules
    d.setdefault("pdbx_ec", ["?"] * n)
    d.setdefault("pdbx_mutation", ["?"] * n)
    d.setdefault("pdbx_fragment", ["?"] * n)
    d.setdefault("details", ["?"] * n)
    # Reorder to match maxit: id, type, src_method, pdbx_description, formula_weight,
    #   pdbx_number_of_molecules, pdbx_ec, pdbx_mutation, pdbx_fragment, details
    desired = [
        "id",
        "type",
        "src_method",
        "pdbx_description",
        "formula_weight",
        "pdbx_number_of_molecules",
        "pdbx_ec",
        "pdbx_mutation",
        "pdbx_fragment",
        "details",
    ]
    cats["_entity."] = {k: d.get(k, ["?"] * n) for k in desired if k in d}


def _fix_struct_ref_columns(cats: dict) -> None:
    """Fix struct_ref: maxit convention is a single row with entity_id='.'.
    Multiple struct_ref rows (one per entity) are collapsed to one."""
    d = cats.get("_struct_ref.", {})
    if not d:
        return

    def _first(v):
        """Get first element if list, otherwise as-is."""
        return v[0] if isinstance(v, list) and v else v

    # Take first row only; maxit outputs a single struct_ref per entry
    cats["_struct_ref."] = {
        "id": _first(d.get("id", "1")),
        "db_name": _first(d.get("db_name", "PDB")),
        "db_code": _first(d.get("db_code", "?")),
        "pdbx_db_accession": "?",
        "pdbx_db_isoform": "?",
        "entity_id": ".",
        "pdbx_seq_one_letter_code": "?",
        "pdbx_align_begin": "?",
    }


def _fix_audit_author(cats: dict) -> None:
    """Reorder audit_author columns and title-case names."""
    d = cats.get("_audit_author.", {})
    if not d:
        return
    # Determine which is which (ordinals are all digits)
    names_col = d.get("name", [])
    ordinals_col = d.get("pdbx_ordinal", [])

    # Title-case names
    def _titlecase_author(name: str) -> str:
        if not name or name == "?":
            return name
        # Format: "SURNAME, I." → "Surname, I."
        parts = name.split(",", 1)
        if len(parts) == 2:
            surname = parts[0].strip().title()
            initials = parts[1].strip()
            return f"{surname}, {initials}"
        return name.title()

    if isinstance(names_col, list):
        names_col = [_titlecase_author(n) for n in names_col]
    cats["_audit_author."] = {
        "name": names_col,
        "pdbx_ordinal": ordinals_col,
    }


def _fix_struct_conf(cats: dict, pdb_lines: list[str] | None = None) -> None:
    """Fix struct_conf: rename ids H1→HELX_P1, add pdbx_PDB_helix_id, fix column order."""
    d = cats.get("_struct_conf.", {})
    if not d:
        return
    n = _nrows(cats, "_struct_conf.")
    ids = d.get("id", ["?"] * n)

    # Parse PDB HELIX records for the original helixID field (cols 11-14)
    helix_id_map: dict[int, str] = {}  # serial number (1-indexed) → helixID
    if pdb_lines:
        for line in pdb_lines:
            if line.startswith("HELIX "):
                try:
                    serial = int(line[7:10].strip())
                    helix_id = line[11:14].strip()
                    helix_id_map[serial] = helix_id
                except (ValueError, IndexError):
                    pass

    # Rename H1 → HELX_P1, H2 → HELX_P2, etc.
    new_ids = []
    helix_nums = []
    for _, hid in enumerate(ids if isinstance(ids, list) else [ids]):
        m = re.match(r"H(\d+)$", str(hid))
        if m:
            num = int(m.group(1))
            new_ids.append(f"HELX_P{num}")
            # Use PDB helixID if available, otherwise use the serial number
            pdb_helix_id = helix_id_map.get(num, str(num))
            helix_nums.append(pdb_helix_id)
        else:
            new_ids.append(str(hid))
            helix_nums.append("?")

    # maxit column order for struct_conf
    desired = [
        "conf_type_id",
        "id",
        "pdbx_PDB_helix_id",
        "beg_label_comp_id",
        "beg_label_asym_id",
        "beg_label_seq_id",
        "pdbx_beg_PDB_ins_code",
        "end_label_comp_id",
        "end_label_asym_id",
        "end_label_seq_id",
        "pdbx_end_PDB_ins_code",
        "beg_auth_comp_id",
        "beg_auth_asym_id",
        "beg_auth_seq_id",
        "end_auth_comp_id",
        "end_auth_asym_id",
        "end_auth_seq_id",
        "pdbx_PDB_helix_class",
        "details",
        "pdbx_PDB_helix_length",
    ]
    new_d: dict[str, list] = {}
    new_d["conf_type_id"] = d.get("conf_type_id", ["HELX_P"] * n)
    new_d["id"] = new_ids
    new_d["pdbx_PDB_helix_id"] = helix_nums
    for k in desired[3:]:
        if k in d:
            new_d[k] = d[k]
        else:
            new_d[k] = ["?"] * n
    # Copy auth_comp_id from label_comp_id if missing
    if not any(v not in ("?", None) for v in new_d.get("beg_auth_comp_id", [])):
        new_d["beg_auth_comp_id"] = list(new_d.get("beg_label_comp_id", ["?"] * n))
    if not any(v not in ("?", None) for v in new_d.get("end_auth_comp_id", [])):
        new_d["end_auth_comp_id"] = list(new_d.get("end_label_comp_id", ["?"] * n))
    cats["_struct_conf."] = new_d


def _fix_struct_conf_type(cats: dict) -> None:
    """Add criteria and reference to struct_conf_type."""
    d = cats.get("_struct_conf_type.", {})
    if not d:
        return
    d.setdefault("criteria", "?")
    d.setdefault("reference", "?")


def _fix_struct_sheet_columns(cats: dict) -> None:
    """Add type and details columns to struct_sheet."""
    d = cats.get("_struct_sheet.", {})
    if not d:
        return
    n = _nrows(cats, "_struct_sheet.")
    d.setdefault("type", ["?"] * n)
    d.setdefault("details", ["?"] * n)
    # Reorder: id, type, number_strands, details
    new_d = {"id": d.get("id", ["?"] * n)}
    new_d["type"] = d.get("type", ["?"] * n)
    new_d["number_strands"] = d.get("number_strands", ["?"] * n)
    new_d["details"] = d.get("details", ["?"] * n)
    cats["_struct_sheet."] = new_d


def _fix_struct_sheet_order_columns(cats: dict) -> None:
    """Add offset column to struct_sheet_order."""
    d = cats.get("_struct_sheet_order.", {})
    if not d:
        return
    n = _nrows(cats, "_struct_sheet_order.")
    d.setdefault("offset", ["?"] * n)
    # Reorder: sheet_id, range_id_1, range_id_2, offset, sense
    desired = ["sheet_id", "range_id_1", "range_id_2", "offset", "sense"]
    cats["_struct_sheet_order."] = {k: d.get(k, ["?"] * n) for k in desired}


def _fix_struct_mon_prot_cis(cats: dict) -> None:
    """Fix struct_mon_prot_cis column order to match maxit."""
    d = cats.get("_struct_mon_prot_cis.", {})
    if not d:
        return
    n = _nrows(cats, "_struct_mon_prot_cis.")
    desired = [
        "pdbx_id",
        "label_comp_id",
        "label_seq_id",
        "label_asym_id",
        "label_alt_id",
        "pdbx_PDB_ins_code",
        "auth_comp_id",
        "auth_seq_id",
        "auth_asym_id",
        "pdbx_label_comp_id_2",
        "pdbx_label_seq_id_2",
        "pdbx_label_asym_id_2",
        "pdbx_PDB_ins_code_2",
        "pdbx_auth_comp_id_2",
        "pdbx_auth_seq_id_2",
        "pdbx_auth_asym_id_2",
        "pdbx_PDB_model_num",
        "pdbx_omega_angle",
    ]
    new_d: dict[str, list] = {}
    for k in desired:
        if k in d:
            new_d[k] = d[k]
        else:
            new_d[k] = ["?"] * n
    # auth_comp_id = label_comp_id if missing
    if not any(v != "?" for v in new_d.get("auth_comp_id", [])):
        new_d["auth_comp_id"] = list(new_d.get("label_comp_id", ["?"] * n))
    # pdbx_auth_comp_id_2 = pdbx_label_comp_id_2 if missing
    if not any(v not in ("?", None) for v in new_d.get("pdbx_auth_comp_id_2", [])):
        new_d["pdbx_auth_comp_id_2"] = list(new_d.get("pdbx_label_comp_id_2", ["?"] * n))
    # Format omega angle to 2 decimal places
    if "pdbx_omega_angle" in new_d:
        new_d["pdbx_omega_angle"] = [_fmt_float(v, 2) for v in new_d["pdbx_omega_angle"]]
    cats["_struct_mon_prot_cis."] = new_d


# ---------------------------------------------------------------------------
# Add missing categories
# ---------------------------------------------------------------------------


def _fix_struct_sheet_range(cats: dict) -> None:
    """Fix struct_sheet_range column order to match maxit."""
    d = cats.get("_struct_sheet_range.", {})
    if not d:
        return
    n = _nrows(cats, "_struct_sheet_range.")
    desired = [
        "sheet_id",
        "id",
        "beg_label_comp_id",
        "beg_label_asym_id",
        "beg_label_seq_id",
        "pdbx_beg_PDB_ins_code",
        "end_label_comp_id",
        "end_label_asym_id",
        "end_label_seq_id",
        "pdbx_end_PDB_ins_code",
        "beg_auth_comp_id",
        "beg_auth_asym_id",
        "beg_auth_seq_id",
        "end_auth_comp_id",
        "end_auth_asym_id",
        "end_auth_seq_id",
    ]
    new_d: dict[str, list] = {}
    for k in desired:
        if k in d:
            new_d[k] = d[k]
        else:
            new_d[k] = ["?"] * n
    # Copy auth fields from label if missing
    for prefix in ("beg", "end"):
        if not any(v not in ("?", None) for v in new_d.get(f"{prefix}_auth_comp_id", [])):
            new_d[f"{prefix}_auth_comp_id"] = list(new_d.get(f"{prefix}_label_comp_id", ["?"] * n))
    cats["_struct_sheet_range."] = new_d


def _fix_refine(cats: dict, pdb_id: str, pdb_lines: list[str]) -> None:
    """Fix _refine: correct pdbx_refine_id, parse REMARK 3, and add missing fields."""
    d = cats.get("_refine.", {})
    if not d:
        return

    # Parse REMARK 3 for values gemmi doesn't read
    remark3: dict[str, str] = {}
    for line in pdb_lines:
        if not line.startswith("REMARK   3"):
            continue
        text = line[11:].strip()
        m = re.search(r"MIN\(FOBS/SIGMA_FOBS\)\s*:\s*([\d.]+)", text)
        if m:
            remark3["pdbx_ls_sigma_F"] = m.group(1)
        m = re.search(r"FREE R VALUE TEST SET SIZE\s*\(%\)\s*:\s*([\d.]+)", text)
        if m:
            remark3["ls_percent_reflns_R_free"] = m.group(1)
        m = re.search(r"METHOD USED\s+:\s+(.+)", text)
        if m:
            remark3["solvent_model_details"] = m.group(1).strip()
        m = re.search(r"SOLVENT RADIUS\s*:\s*([\d.]+)", text)
        if m:
            remark3["pdbx_solvent_vdw_probe_radii"] = m.group(1)
        m = re.search(r"SHRINKAGE RADIUS\s*:\s*([\d.]+)", text)
        if m:
            remark3["pdbx_solvent_shrinkage_radii"] = m.group(1)
        m = re.search(r"REFINEMENT TARGET\s*:\s*(\S+)", text)
        if m:
            remark3["pdbx_stereochemistry_target_values"] = m.group(1)
        m = re.search(r"COORDINATE ERROR.*:\s*([\d.]+)", text)
        if m:
            remark3["overall_SU_ML"] = m.group(1)
        m = re.search(r"PHASE ERROR.*:\s*([\d.]+)", text)
        if m:
            remark3["pdbx_overall_phase_error"] = m.group(1)

    # Format resolution values to 2 decimal places
    for col in ("ls_d_res_low", "ls_d_res_high"):
        if col in d:
            d[col] = _fmt_float(d[col], 2)
    # Fix pdbx_refine_id: gemmi uses "1", maxit uses "X-RAY DIFFRACTION"
    d["pdbx_refine_id"] = "X-RAY DIFFRACTION"
    d.setdefault("entry_id", pdb_id)
    d.setdefault("pdbx_diffrn_id", "1")
    d.setdefault("pdbx_TLS_residual_ADP_flag", "?")
    d.setdefault("ls_number_reflns_all", "?")
    d.setdefault("pdbx_ls_sigma_I", "?")
    d["pdbx_ls_sigma_F"] = remark3.get("pdbx_ls_sigma_F", "?")
    d.setdefault("pdbx_data_cutoff_high_absF", "?")
    d.setdefault("pdbx_data_cutoff_low_absF", "?")
    d.setdefault("pdbx_data_cutoff_high_rms_absF", "?")
    d.setdefault("ls_R_factor_all", "?")
    d.setdefault("ls_R_factor_R_free_error", "?")
    d.setdefault("ls_R_factor_R_free_error_details", "?")
    d["ls_percent_reflns_R_free"] = remark3.get("ls_percent_reflns_R_free", "?")
    d.setdefault("ls_number_parameters", "?")
    d.setdefault("ls_number_restraints", "?")
    d.setdefault("occupancy_min", "?")
    d.setdefault("occupancy_max", "?")
    d.setdefault("correlation_coeff_Fo_to_Fc", "?")
    d.setdefault("correlation_coeff_Fo_to_Fc_free", "?")
    d.setdefault("B_iso_mean", "?")
    for b in (
        "aniso_B[1][1]",
        "aniso_B[2][2]",
        "aniso_B[3][3]",
        "aniso_B[1][2]",
        "aniso_B[1][3]",
        "aniso_B[2][3]",
    ):
        d.setdefault(b, "?")
    d["solvent_model_details"] = remark3.get("solvent_model_details", "?")
    d.setdefault("solvent_model_param_ksol", "?")
    d.setdefault("solvent_model_param_bsol", "?")
    d["pdbx_solvent_vdw_probe_radii"] = remark3.get("pdbx_solvent_vdw_probe_radii", "?")
    d.setdefault("pdbx_solvent_ion_probe_radii", "?")
    d["pdbx_solvent_shrinkage_radii"] = remark3.get("pdbx_solvent_shrinkage_radii", "?")
    d.setdefault("pdbx_ls_cross_valid_method", "?")
    d.setdefault("details", "?")
    d.setdefault("pdbx_starting_model", "?")
    d.setdefault("pdbx_isotropic_thermal_model", "?")
    d["pdbx_stereochemistry_target_values"] = remark3.get("pdbx_stereochemistry_target_values", "?")
    d.setdefault("pdbx_stereochem_target_val_spec_case", "?")
    d.setdefault("pdbx_R_Free_selection_details", "?")
    d.setdefault("pdbx_overall_ESU_R", "?")
    d.setdefault("pdbx_overall_ESU_R_Free", "?")
    d["overall_SU_ML"] = remark3.get("overall_SU_ML", "?")
    d["pdbx_overall_phase_error"] = remark3.get("pdbx_overall_phase_error", "?")
    d.setdefault("overall_SU_B", "?")
    d.setdefault("overall_SU_R_Cruickshank_DPI", "?")
    d.setdefault("pdbx_overall_SU_R_free_Cruickshank_DPI", "?")
    d.setdefault("pdbx_overall_SU_R_Blow_DPI", "?")
    d.setdefault("pdbx_overall_SU_R_free_Blow_DPI", "?")
    # Reorder to match maxit
    desired = [
        "pdbx_refine_id",
        "entry_id",
        "pdbx_diffrn_id",
        "pdbx_TLS_residual_ADP_flag",
        "ls_number_reflns_obs",
        "ls_number_reflns_all",
        "pdbx_ls_sigma_I",
        "pdbx_ls_sigma_F",
        "pdbx_data_cutoff_high_absF",
        "pdbx_data_cutoff_low_absF",
        "pdbx_data_cutoff_high_rms_absF",
        "ls_d_res_low",
        "ls_d_res_high",
        "ls_percent_reflns_obs",
        "ls_R_factor_obs",
        "ls_R_factor_all",
        "ls_R_factor_R_work",
        "ls_R_factor_R_free",
        "ls_R_factor_R_free_error",
        "ls_R_factor_R_free_error_details",
        "ls_percent_reflns_R_free",
        "ls_number_reflns_R_free",
        "ls_number_parameters",
        "ls_number_restraints",
        "occupancy_min",
        "occupancy_max",
        "correlation_coeff_Fo_to_Fc",
        "correlation_coeff_Fo_to_Fc_free",
        "B_iso_mean",
        "aniso_B[1][1]",
        "aniso_B[2][2]",
        "aniso_B[3][3]",
        "aniso_B[1][2]",
        "aniso_B[1][3]",
        "aniso_B[2][3]",
        "solvent_model_details",
        "solvent_model_param_ksol",
        "solvent_model_param_bsol",
        "pdbx_solvent_vdw_probe_radii",
        "pdbx_solvent_ion_probe_radii",
        "pdbx_solvent_shrinkage_radii",
        "pdbx_ls_cross_valid_method",
        "details",
        "pdbx_starting_model",
        "pdbx_method_to_determine_struct",
        "pdbx_isotropic_thermal_model",
        "pdbx_stereochemistry_target_values",
        "pdbx_stereochem_target_val_spec_case",
        "pdbx_R_Free_selection_details",
        "pdbx_overall_ESU_R",
        "pdbx_overall_ESU_R_Free",
        "overall_SU_ML",
        "pdbx_overall_phase_error",
        "overall_SU_B",
        "overall_SU_R_Cruickshank_DPI",
        "pdbx_overall_SU_R_free_Cruickshank_DPI",
        "pdbx_overall_SU_R_Blow_DPI",
        "pdbx_overall_SU_R_free_Blow_DPI",
    ]
    # Keys that maxit does NOT include
    _REFINE_EXCLUDE = {"ls_number_reflns_R_work"}
    cats["_refine."] = {k: d[k] for k in desired if k in d}
    # Add any remaining keys not in the exclude list
    for k, v in d.items():
        if k not in cats["_refine."] and k not in _REFINE_EXCLUDE:
            cats["_refine."][k] = v


def _fix_reflns(cats: dict, pdb_lines: list[str]) -> None:
    """Fix _reflns: add missing fields, fix field order, and format decimals."""
    d = cats.get("_reflns.", {})
    if not d:
        return
    # Parse REMARK 200/3 for various fields
    sigma_I = "?"
    b_wilson = "?"
    d_res_high_str = None  # use string from REMARK 200 to preserve precision
    redundancy_str = None
    for line in pdb_lines:
        if line.startswith("REMARK 200"):
            m = re.search(r"REJECTION CRITERIA\s*\(SIGMA\(I\)\)\s*:\s*([\d.]+)", line)
            if m:
                sigma_I = m.group(1)
            m = re.search(r"RESOLUTION RANGE HIGH\s*\(A\)\s*:\s*([\d.]+)", line)
            if m:
                d_res_high_str = m.group(1)
            m = re.search(r"DATA REDUNDANCY\s*:\s*([\d.]+)", line)
            if m and "SHELL" not in line:
                redundancy_str = m.group(1)
        elif line.startswith("REMARK   3"):
            m = re.search(r"FROM WILSON PLOT\s+\(A\*\*2\)\s*:\s*([\d.]+)", line)
            if m:
                b_wilson = m.group(1)
    d["pdbx_diffrn_id"] = "1"
    d["observed_criterion_sigma_I"] = sigma_I
    d["observed_criterion_sigma_F"] = "?"
    d.setdefault("number_all", "?")
    d["B_iso_Wilson_estimate"] = b_wilson
    # Override d_resolution_high with REMARK 200 string to preserve trailing zeros
    if d_res_high_str is not None:
        d["d_resolution_high"] = d_res_high_str
    # Override pdbx_redundancy with REMARK 200 string to preserve precision
    if redundancy_str is not None:
        d["pdbx_redundancy"] = redundancy_str
    # Format numeric fields
    if "d_resolution_low" in d:
        d["d_resolution_low"] = _fmt_float(d["d_resolution_low"], 3)
    if "pdbx_Rmerge_I_obs" in d:
        d["pdbx_Rmerge_I_obs"] = _fmt_float(d["pdbx_Rmerge_I_obs"], 5)
    if "pdbx_netI_over_sigmaI" in d:
        d["pdbx_netI_over_sigmaI"] = _fmt_float(d["pdbx_netI_over_sigmaI"], 4)
    if redundancy_str is None and "pdbx_redundancy" in d:
        d["pdbx_redundancy"] = _fmt_float(d["pdbx_redundancy"], 3)
    # Reorder: pdbx_diffrn_id and pdbx_ordinal before entry_id (maxit convention)
    desired = [
        "pdbx_diffrn_id",
        "pdbx_ordinal",
        "entry_id",
        "observed_criterion_sigma_I",
        "observed_criterion_sigma_F",
        "d_resolution_low",
        "d_resolution_high",
        "number_obs",
        "number_all",
        "percent_possible_obs",
        "pdbx_Rmerge_I_obs",
        "pdbx_Rsym_value",
        "pdbx_netI_over_sigmaI",
        "B_iso_Wilson_estimate",
        "pdbx_redundancy",
    ]
    new_d = {k: d[k] for k in desired if k in d}
    for k, v in d.items():
        if k not in new_d:
            new_d[k] = v
    cats["_reflns."] = new_d


def _fix_reflns_shell(cats: dict) -> None:
    """Fix _reflns_shell: field order and decimal formatting."""
    d = cats.get("_reflns_shell.", {})
    if not d:
        return
    d.setdefault("pdbx_diffrn_id", "1")
    d.setdefault("pdbx_Rsym_value", "?")
    # Format decimal fields
    if "d_res_high" in d:
        d["d_res_high"] = _fmt_float(d["d_res_high"], 2)
    if "d_res_low" in d:
        d["d_res_low"] = _fmt_float(d["d_res_low"], 2)
    if "Rmerge_I_obs" in d:
        d["Rmerge_I_obs"] = _fmt_float(d["Rmerge_I_obs"], 5)
    if "meanI_over_sigI_obs" in d:
        d["meanI_over_sigI_obs"] = _fmt_float(d["meanI_over_sigI_obs"], 3)
    if "pdbx_redundancy" in d:
        d["pdbx_redundancy"] = _fmt_float(d["pdbx_redundancy"], 2)
    desired = [
        "pdbx_diffrn_id",
        "pdbx_ordinal",
        "d_res_high",
        "d_res_low",
        "percent_possible_all",
        "Rmerge_I_obs",
        "pdbx_Rsym_value",
        "meanI_over_sigI_obs",
        "pdbx_redundancy",
    ]
    new_d = {k: d[k] for k in desired if k in d}
    for k, v in d.items():
        if k not in new_d:
            new_d[k] = v
    cats["_reflns_shell."] = new_d


def _fix_exptl_crystal_grow(cats: dict, pdb_lines: list[str]) -> None:
    """Add missing fields to _exptl_crystal_grow, parsing REMARK 280."""
    # Create category from scratch if missing (gemmi may not populate it)
    d = cats.setdefault("_exptl_crystal_grow.", {})
    # Parse REMARK 280 for crystallization details, preserving original line breaks
    remark280_lines: list[str] = []
    in_cond = False
    for line in pdb_lines:
        if not line.startswith("REMARK 280"):
            continue
        # Preserve original column content (cols 11 onwards, rstrip)
        text = line[10:].rstrip()  # col 10 = space before content
        m = re.search(r"CRYSTALLIZATION CONDITIONS:\s*(.+)", text)
        if m:
            in_cond = True
            remark280_lines.append(m.group(1).rstrip())
        elif in_cond:
            # Continuation line: PDB REMARK content starts at col 11 (0-indexed)
            content = line[11:].rstrip() if len(line) > 11 else ""
            if content and not re.match(r"(SOLVENT|MATTHEWS|CRYSTAL)", content.lstrip()):
                remark280_lines.append(content)
            else:
                in_cond = False
    if remark280_lines:
        pdbx_details = ";" + "\n".join(remark280_lines) + "\n;"
    else:
        pdbx_details = "?"

    d.setdefault("crystal_id", "1")
    d.setdefault("method", "?")
    d.setdefault("temp", "?")
    d.setdefault("temp_details", "?")
    d.setdefault("pH", "?")
    d["pdbx_details"] = pdbx_details
    # Fix pdbx_pH_range: copy from pH if available
    if "pH" in d:
        ph = d["pH"]
        d["pdbx_pH_range"] = ph if isinstance(ph, list) else ph
    else:
        d.setdefault("pdbx_pH_range", "?")
    # Reorder
    desired = [
        "crystal_id",
        "method",
        "temp",
        "temp_details",
        "pH",
        "pdbx_pH_range",
        "pdbx_details",
    ]
    n = _nrows(cats, "_exptl_crystal_grow.") or 1
    cats["_exptl_crystal_grow."] = {
        k: d.get(k, "?" if n == 1 else ["?"] * n)
        for k in desired
        if k in d or k in ("method", "temp", "temp_details", "pdbx_pH_range", "pdbx_details")
    }


def _fix_exptl_crystal(cats: dict, pdb_lines: list[str]) -> None:
    """Add missing density fields to _exptl_crystal, parsing REMARK 280."""
    d = cats.get("_exptl_crystal.", {})
    if not d:
        return
    # Parse REMARK 280 for density
    matthews = "?"
    percent_sol = "?"
    for line in pdb_lines:
        if not line.startswith("REMARK 280"):
            continue
        text = line[11:].strip()
        m = re.search(r"MATTHEWS COEFFICIENT.*:\s*([\d.]+)", text)
        if m:
            matthews = m.group(1)
        m = re.search(r"SOLVENT CONTENT.*:\s*([\d.]+)", text)
        if m:
            percent_sol = m.group(1)
    d.setdefault("density_meas", "?")
    d["density_Matthews"] = matthews
    d["density_percent_sol"] = percent_sol
    d.setdefault("description", "?")
    # Reorder: id, density_meas, density_Matthews, density_percent_sol, description
    desired = [
        "id",
        "density_meas",
        "density_Matthews",
        "density_percent_sol",
        "description",
    ]
    cats["_exptl_crystal."] = {k: d[k] for k in desired if k in d}


def _fix_pdbx_struct_assembly(cats: dict) -> None:
    """Fix oligomeric_details to uppercase and set count to ?."""
    d = cats.get("_pdbx_struct_assembly.", {})
    if not d:
        return
    # oligomeric_details should be uppercase (DIMERIC not dimeric)
    if "oligomeric_details" in d:
        v = d["oligomeric_details"]
        d["oligomeric_details"] = (
            v.upper() if isinstance(v, str) else [x.upper() if isinstance(x, str) else x for x in v]
        )
    # oligomeric_count should be ?
    if "oligomeric_count" in d:
        v = d["oligomeric_count"]
        d["oligomeric_count"] = "?" if not isinstance(v, list) else ["?"] * len(v)
    # Add details to pdbx_struct_assembly_prop; format MORE values to 1 dp
    pp = cats.get("_pdbx_struct_assembly_prop.", {})
    if pp:
        n = _nrows(cats, "_pdbx_struct_assembly_prop.")
        pp.setdefault("details", ["?"] * n if n > 1 else "?")
        types = pp.get("type", [])
        values = pp.get("value", [])
        if isinstance(types, list) and isinstance(values, list):
            for i, t in enumerate(types):
                if str(t).strip() == "MORE" and i < len(values):
                    v = values[i]
                    raw = v[0] if isinstance(v, list) else v
                    try:
                        values[i] = f"{float(raw):.1f}"
                    except (ValueError, TypeError):
                        pass


def _fix_pdbx_struct_oper_list(cats: dict) -> None:
    """Add name and symmetry_operation fields; format matrix values to 10 dp; fix field order."""
    d = cats.get("_pdbx_struct_oper_list.", {})
    if not d:
        return
    d.setdefault("name", "1_555")
    d.setdefault("symmetry_operation", "x,y,z")
    # Format matrix/vector values to 10 decimal places
    for k in list(d.keys()):
        if "matrix" in k or "vector" in k:
            v = d[k]
            raw = v[0] if isinstance(v, list) else v
            try:
                d[k] = f"{float(raw):.10f}"
            except (ValueError, TypeError):
                pass
    # Reorder: id, type, name, symmetry_operation, then matrix/vector in row order
    desired = [
        "id",
        "type",
        "name",
        "symmetry_operation",
        "matrix[1][1]",
        "matrix[1][2]",
        "matrix[1][3]",
        "vector[1]",
        "matrix[2][1]",
        "matrix[2][2]",
        "matrix[2][3]",
        "vector[2]",
        "matrix[3][1]",
        "matrix[3][2]",
        "matrix[3][3]",
        "vector[3]",
    ]
    new_d: dict = {}
    for k in desired:
        if k in d:
            new_d[k] = d[k]
    for k, v in d.items():
        if k not in new_d:
            new_d[k] = v
    cats["_pdbx_struct_oper_list."] = new_d


_SYNCHROTRON_NAMES = {
    "SPRING-8": "SPring-8",
    "APS": "APS",
    "ESRF": "ESRF",
    "ALS": "ALS",
    "NSLS": "NSLS",
    "NSLS-II": "NSLS-II",
    "DIAMOND": "Diamond",
    "DESY": "DESY",
    "SSRL": "SSRL",
    "SLS": "SLS",
    "SOLEIL": "Soleil",
    "ELETTRA": "Elettra",
    "PHOTON FACTORY": "Photon Factory",
    "KEK/PF": "Photon Factory",
}


def _fix_diffrn_radiation(cats: dict, pdb_lines: list[str]) -> None:
    """Fix _diffrn_radiation: add wavelength_id and pdbx_diffrn_protocol."""
    d = cats.get("_diffrn_radiation.", {})
    if not d:
        return
    # Parse REMARK 200 for diffraction protocol
    protocol = "?"
    for line in pdb_lines:
        if line.startswith("REMARK 200"):
            m = re.search(r"DIFFRACTION PROTOCOL\s*:\s*(.+)", line)
            if m:
                protocol = m.group(1).strip()
                break
    d["wavelength_id"] = "1"
    d["pdbx_diffrn_protocol"] = protocol
    desired = [
        "diffrn_id",
        "wavelength_id",
        "pdbx_monochromatic_or_laue_m_l",
        "monochromator",
        "pdbx_diffrn_protocol",
        "pdbx_scattering_type",
    ]
    new_d = {k: d[k] for k in desired if k in d}
    for k, v in d.items():
        if k not in new_d:
            new_d[k] = v
    cats["_diffrn_radiation."] = new_d


def _fix_diffrn_source(cats: dict) -> None:
    """Fix diffrn_source: correct synchrotron site name casing, add pdbx_wavelength."""
    d = cats.get("_diffrn_source.", {})
    if not d:
        return
    # Fix pdbx_synchrotron_site casing
    if "pdbx_synchrotron_site" in d:
        v = d["pdbx_synchrotron_site"]
        val = v[0] if isinstance(v, list) else v
        fixed = _SYNCHROTRON_NAMES.get(str(val).strip("'"), str(val).strip("'"))
        d["pdbx_synchrotron_site"] = [fixed] if isinstance(v, list) else fixed
    # Move wavelength to pdbx_wavelength if pdbx_wavelength_list has a single value
    # (single wavelength experiment → pdbx_wavelength; multi → pdbx_wavelength_list)
    wl_list = d.get("pdbx_wavelength_list", "?")
    # Strip surrounding quotes, brackets (gemmi sometimes wraps in ['1.1'])
    wl_list_str = str(wl_list).strip("'\"[]") if wl_list not in (None, "?", ".") else "?"
    if wl_list_str != "?" and "," not in wl_list_str:
        # Single wavelength: move to pdbx_wavelength, clear list
        d["pdbx_wavelength"] = wl_list_str
        d["pdbx_wavelength_list"] = "?"
    else:
        d.setdefault("pdbx_wavelength", "?")
    desired = [
        "diffrn_id",
        "source",
        "type",
        "pdbx_synchrotron_site",
        "pdbx_synchrotron_beamline",
        "pdbx_wavelength",
        "pdbx_wavelength_list",
    ]
    new_d = {k: d[k] for k in desired if k in d}
    for k, v in d.items():
        if k not in new_d:
            new_d[k] = v
    cats["_diffrn_source."] = new_d


def _fix_diffrn_detector(cats: dict) -> None:
    """Fix _diffrn_detector field order."""
    d = cats.get("_diffrn_detector.", {})
    if not d:
        return
    desired = ["diffrn_id", "detector", "type", "pdbx_collection_date", "details"]
    new_d = {k: d[k] for k in desired if k in d}
    for k, v in d.items():
        if k not in new_d:
            new_d[k] = v
    cats["_diffrn_detector."] = new_d


def _fix_diffrn(cats: dict) -> None:
    """Fix _diffrn field order; add ambient_temp_details."""
    d = cats.get("_diffrn.", {})
    if not d:
        return
    d.setdefault("ambient_temp_details", "?")
    # Reorder: id, ambient_temp, ambient_temp_details, crystal_id
    desired = ["id", "ambient_temp", "ambient_temp_details", "crystal_id"]
    new_d = {k: d[k] for k in desired if k in d}
    for k, v in d.items():
        if k not in new_d:
            new_d[k] = v
    cats["_diffrn."] = new_d


_SOFTWARE_CLASS_ORDER = [
    "data collection",
    "data reduction",
    "data scaling",
    "phasing",
    "model building",
    "refinement",
    "other",
]

# Canonical software name spellings (all-caps PDB → maxit convention)
_SOFTWARE_NAME_MAP = {
    "AIMLESS": "Aimless",
    "SCALA": "Scala",
    "MOSFLM": "MOSFLM",
    "DENZO": "DENZO",
    "SCALEPACK": "SCALEPACK",
    "TRUNCATE": "TRUNCATE",
    "REFMAC": "REFMAC",
    "COOT": "COOT",
    "PHENIX": "PHENIX",
    "BUSTER": "BUSTER",
    "SHELXL": "SHELXL",
    "CNS": "CNS",
}


def _fix_software(cats: dict, pdb_lines: list[str]) -> None:
    """Fix software: add citation_id, reorder columns and rows to match maxit."""
    d = cats.get("_software.", {})
    if not d:
        return
    n = _nrows(cats, "_software.")
    if n == 0:
        return

    # Normalize version: False/None → "."
    # For PHENIX, extract just version number from "PHENIX.REFINE: 1.8.1_1168"
    vers = d.get("version", ["."] * n)
    vers = vers if isinstance(vers, list) else [vers]
    new_vers = []
    for v in vers:
        if v is False or v is None or v == "":
            new_vers.append(".")
        else:
            s = str(v)
            # Extract version after last colon+space if present
            m = re.search(r":\s*(\S+)\s*$", s)
            if m:
                new_vers.append(m.group(1))
            else:
                new_vers.append(s)
    d["version"] = new_vers

    # Normalize software names (e.g. AIMLESS → Aimless)
    names = d.get("name", ["?"] * n)
    names = names if isinstance(names, list) else [names]
    d["name"] = [_SOFTWARE_NAME_MAP.get(str(nm), str(nm)) for nm in names]

    # Add citation_id column if missing
    d.setdefault("citation_id", ["?"] * n)

    # Sort rows by classification order
    cls_col = d.get("classification", ["?"] * n)
    cls_col = cls_col if isinstance(cls_col, list) else [cls_col]

    def _cls_key(cls: str) -> int:
        c = str(cls).lower() if cls else ""
        for i, ref in enumerate(_SOFTWARE_CLASS_ORDER):
            if ref in c:
                return i
        return len(_SOFTWARE_CLASS_ORDER)

    indices = sorted(range(n), key=lambda i: _cls_key(cls_col[i]))

    def _reorder(col):
        if isinstance(col, list):
            return [col[i] for i in indices]
        return col

    reordered: dict[str, list] = {}
    for k, v in d.items():
        reordered[k] = _reorder(v)

    # Renumber pdbx_ordinal after sorting
    reordered["pdbx_ordinal"] = [str(i + 1) for i in range(n)]

    # Column order: classification, name, version, citation_id, pdbx_ordinal
    desired = ["classification", "name", "version", "citation_id", "pdbx_ordinal"]
    new_d = {k: reordered[k] for k in desired if k in reordered}
    for k, v in reordered.items():
        if k not in new_d:
            new_d[k] = v
    cats["_software."] = new_d


def _fix_pdbx_struct_sheet_hbond(cats: dict) -> None:
    """Fix _pdbx_struct_sheet_hbond column order to match maxit."""
    d = cats.get("_pdbx_struct_sheet_hbond.", {})
    if not d:
        return
    n = _nrows(cats, "_pdbx_struct_sheet_hbond.")
    desired = [
        "sheet_id",
        "range_id_1",
        "range_id_2",
        "range_1_label_atom_id",
        "range_1_label_comp_id",
        "range_1_label_asym_id",
        "range_1_label_seq_id",
        "range_1_PDB_ins_code",
        "range_1_auth_atom_id",
        "range_1_auth_comp_id",
        "range_1_auth_asym_id",
        "range_1_auth_seq_id",
        "range_2_label_atom_id",
        "range_2_label_comp_id",
        "range_2_label_asym_id",
        "range_2_label_seq_id",
        "range_2_PDB_ins_code",
        "range_2_auth_atom_id",
        "range_2_auth_comp_id",
        "range_2_auth_asym_id",
        "range_2_auth_seq_id",
    ]
    new_d: dict[str, list] = {}
    for k in desired:
        if k in d:
            new_d[k] = d[k]
        else:
            new_d[k] = ["?"] * n
    # Fill auth_atom_id = label_atom_id, auth_comp_id = label_comp_id if missing
    for rng in ("range_1", "range_2"):
        if not any(v != "?" for v in new_d.get(f"{rng}_auth_atom_id", [])):
            new_d[f"{rng}_auth_atom_id"] = list(new_d.get(f"{rng}_label_atom_id", ["?"] * n))
        if not any(v != "?" for v in new_d.get(f"{rng}_auth_comp_id", [])):
            new_d[f"{rng}_auth_comp_id"] = list(new_d.get(f"{rng}_label_comp_id", ["?"] * n))
        if not any(v != "?" for v in new_d.get(f"{rng}_auth_asym_id", [])):
            new_d[f"{rng}_auth_asym_id"] = list(new_d.get(f"{rng}_label_asym_id", ["?"] * n))
    cats["_pdbx_struct_sheet_hbond."] = new_d


def _fix_refine_ls_restr_from_pdb(cats: dict, pdb_lines: list[str]) -> None:
    """Parse REMARK 3 restraint stats and build _refine_ls_restr."""
    # Only rebuild if the category is empty (gemmi doesn't generate it from PDB)
    existing = cats.get("_refine_ls_restr.", {})
    if existing and _nrows(cats, "_refine_ls_restr.") > 0:
        return

    # maxit output order for restraint types
    RESTR_ORDER = [
        "f_bond_d",
        "f_angle_d",
        "f_dihedral_angle_d",
        "f_chiral_restr",
        "f_plane_restr",
    ]
    PATTERNS = [
        (r"BOND\s*:\s+([\d.]+)\s+(\d+)", "f_bond_d"),
        (r"ANGLE\s*:\s+([\d.]+)\s+(\d+)", "f_angle_d"),
        (r"DIHEDRAL\s*:\s+([\d.]+)\s+(\d+)", "f_dihedral_angle_d"),
        (r"CHIRALITY\s*:\s+([\d.]+)\s+(\d+)", "f_chiral_restr"),
        (r"PLANARITY\s*:\s+([\d.]+)\s+(\d+)", "f_plane_restr"),
    ]
    found: dict[str, tuple[str, str]] = {}
    for line in pdb_lines:
        if not line.startswith("REMARK   3"):
            continue
        text = line[11:].strip()
        for pattern, restr_type in PATTERNS:
            m = re.search(pattern, text)
            if m and restr_type not in found:
                found[restr_type] = (m.group(1), m.group(2))

    if not found:
        return

    types, devs, numbers = [], [], []
    for rtype in RESTR_ORDER:
        if rtype in found:
            types.append(rtype)
            devs.append(found[rtype][0])
            numbers.append(found[rtype][1])

    n = len(types)
    cats["_refine_ls_restr."] = {
        "type": types,
        "dev_ideal": devs,
        "dev_ideal_target": ["?"] * n,
        "weight": ["?"] * n,
        "number": numbers,
        "pdbx_refine_id": ["X-RAY DIFFRACTION"] * n,
        "pdbx_restraint_function": ["?"] * n,
    }


def _add_database_2(cats: dict, pdb_id: str) -> None:
    cats["_database_2."] = {
        "database_id": "PDB",
        "database_code": pdb_id,
        "pdbx_database_accession": f"pdb_{pdb_id.lower():0>8s}",
        "pdbx_DOI": "?",
    }


def _add_revision_history(
    cats: dict,
    pdb_lines: list[str],
    prefix_date: datetime.datetime = datetime.datetime(1970, 1, 1),
) -> None:
    """add _pdbx_audit_revision_history loop
    If the input PDB file has:
    '''
    REVDAT   5   20-MAR-24 3WMD    1       SEQADV
    REVDAT   4   22-NOV-17 3WMD    1       REMARK
    REVDAT   3   05-MAR-14 3WMD    1       JRNL
    REVDAT   2   29-JAN-14 3WMD    1       REMARK
    REVDAT   1   15-JAN-14 3WMD    0
    '''
    then, returns
    '''
    _pdbx_audit_revision_history.ordinal
    _pdbx_audit_revision_history.data_content_type
    _pdbx_audit_revision_history.major_revision
    _pdbx_audit_revision_history.minor_revision
    _pdbx_audit_revision_history.revision_date
    1 'Structure model' 1 0 2014-01-15
    2 'Structure model' 1 1 2014-01-29
    3 'Structure model' 1 2 2014-03-05
    4 'Structure model' 1 3 2017-11-22
    5 'Structure model' 1 4 2024-03-20
    '''
    If there are no REVDAT records, adds a single revision with ordinal 1, major 1, minor 0, and the provided date (default 1970-01-01).
    '''
    _pdbx_audit_revision_history.ordinal
    _pdbx_audit_revision_history.data_content_type
    _pdbx_audit_revision_history.major_revision
    _pdbx_audit_revision_history.minor_revision
    _pdbx_audit_revision_history.revision_date
    1 'Structure model' 1 0 1970-01-01
    '''
    """
    revdat_lines = [line for line in pdb_lines if line.startswith("REVDAT")]
    if not revdat_lines:
        cats["_pdbx_audit_revision_history."] = {
            "ordinal": ["1"],
            "data_content_type": ["Structure model"],
            "major_revision": ["1"],
            "minor_revision": ["0"],
            "revision_date": [prefix_date.strftime("%Y-%m-%d")],
        }
        return

    revisions: list[tuple[int, str, str]] = []
    for line in revdat_lines:
        parts = line.split()
        # Minimal REVDAT format: REVDAT <ordinal> <date> <idcode> <mod>
        # e.g. "REVDAT   1   15-JAN-14 3WMD    0"
        if len(parts) < 3:
            continue
        ordinal_str = parts[1]
        date_str = parts[2]
        try:
            ordinal = int(ordinal_str)
        except ValueError:
            continue
        try:
            rev_date = datetime.datetime.strptime(date_str, "%d-%b-%y").strftime("%Y-%m-%d")
        except ValueError:
            rev_date = "?"
        revisions.append((ordinal, "Structure model", rev_date))

    if not revisions:
        cats["_pdbx_audit_revision_history."] = {
            "ordinal": ["1"],
            "data_content_type": ["Structure model"],
            "major_revision": ["1"],
            "minor_revision": ["0"],
            "revision_date": [prefix_date.strftime("%Y-%m-%d")],
        }
        return

    # Sort revisions by ordinal and assign minor revisions as 0,1,2,...
    revisions.sort(key=lambda x: x[0])

    cats["_pdbx_audit_revision_history."] = {
        "ordinal": [str(rev[0]) for rev in revisions],
        "data_content_type": [rev[1] for rev in revisions],
        "major_revision": ["1"] * len(revisions),
        "minor_revision": [str(i) for i in range(len(revisions))],
        "revision_date": [rev[2] for rev in revisions],
    }


def _add_citation_and_authors(cats: dict, pdb_lines: list[str]) -> None:
    jrnl: dict[str, str] = {}
    auth_parts: list[str] = []

    for line in pdb_lines:
        if not line.startswith("JRNL"):
            continue
        sub = line[12:16].strip()
        rest = line[19:].rstrip()
        if sub == "AUTH":
            auth_parts.append(rest.strip())
        elif sub == "TITL":
            jrnl["title"] = jrnl.get("title", "") + " " + rest.strip()
        elif sub == "REF":
            jrnl["ref"] = jrnl.get("ref", "") + " " + rest.strip()
        elif sub == "REFN":
            m = re.search(r"[EI]SSN\s+([\d\-X]+)", rest)
            if m:
                jrnl["issn"] = m.group(1)
        elif sub == "PMID":
            jrnl["pmid"] = rest.strip()
        elif sub == "DOI":
            jrnl["doi"] = rest.strip()

    if not jrnl:
        return

    title_raw = jrnl.get("title", "?").strip()
    # Title-case the citation title
    title = _title_case_words(title_raw)
    ref_text = jrnl.get("ref", "")
    journal, volume, page_first, year = "?", "?", "?", "?"
    m = re.match(r"(.+?)\s+V\.\s+(\S+)\s+(\S+)\s+(\d{4})", ref_text)
    if m:
        journal = m.group(1).strip()
        volume = m.group(2)
        page_first = m.group(3)
        year = m.group(4)

    # Title-case the journal abbreviation
    journal = _title_case_journal(journal)

    # Map PDB journal abbreviations → ASTM codes
    _JOURNAL_ASTM = {
        "J.AM.CHEM.SOC.": "JACSAT",
        "NATURE": "NATUAS",
        "SCIENCE": "SCIEAS",
        "PROC.NAT.ACAD.SCI.,U.S.A.": "PNASA6",
        "PROC.NATL.ACAD.SCI.USA": "PNASA6",
        "J.BIOL.CHEM.": "JBCHA3",
        "BIOCHEMISTRY": "BICHAW",
        "J.MOL.BIOL.": "JMOBAK",
        "NAT.STRUCT.MOL.BIOL.": "NSMBCU",
        "CELL": "CELLB5",
        "NAT.COMMUN.": "NCAOBW",
        "ELIFE": "ELIFA1",
        "STRUCTURE": "STRUE6",
        "ANGEW.CHEM.INT.ED.ENGL.": "ACIEF5",
        "ACTA CRYSTALLOGR.,SECT.D": "ABCRE6",
        "ACTA CRYSTALLOGR.D BIOL.CRYSTALLOGR.": "ABCRE6",
    }
    ref_jrnl_raw = jrnl.get("ref", "").split()[0] if jrnl.get("ref") else ""
    astm = _JOURNAL_ASTM.get(ref_jrnl_raw.upper(), "?")

    cats["_citation."] = {
        "id": "primary",
        "title": title,
        "journal_abbrev": journal,
        "journal_volume": volume,
        "page_first": page_first,
        "page_last": "?",
        "year": year,
        "journal_id_ASTM": astm,
        "country": "US",
        "journal_id_ISSN": jrnl.get("issn", "?"),
        "journal_id_CSD": "?",
        "book_publisher": "?",
        "pdbx_database_id_PubMed": jrnl.get("pmid", "?"),
        "pdbx_database_id_DOI": jrnl.get("doi", "?"),
    }

    auth_text = " ".join(auth_parts).rstrip(",")
    authors = _parse_pdb_author_list(auth_text)
    if authors:
        cats["_citation_author."] = {
            "citation_id": ["primary"] * len(authors),
            "name": authors,
            "identifier_ORCID": ["?"] * len(authors),
            "ordinal": [str(i) for i in range(1, len(authors) + 1)],
        }


def _add_entity_src_gen(cats: dict, pdb_lines: list[str]) -> None:
    sources = _parse_pdb_source(pdb_lines)
    if not sources:
        return

    # Build lists for all mol IDs (loop when multiple, single-row KV when one)
    def _row(mol_id: str, src: dict) -> dict:
        return {
            "entity_id": mol_id,
            "pdbx_src_id": "1",
            "pdbx_alt_source_flag": "?",
            "pdbx_seq_type": "?",
            "pdbx_beg_seq_num": "?",
            "pdbx_end_seq_num": "?",
            "gene_src_common_name": "?",
            "gene_src_genus": "?",
            "pdbx_gene_src_gene": src.get("GENE", "?"),
            "gene_src_species": "?",
            "gene_src_strain": "?",
            "gene_src_tissue": "?",
            "gene_src_tissue_fraction": "?",
            "gene_src_details": "?",
            "pdbx_gene_src_fragment": "?",
            "pdbx_gene_src_scientific_name": src.get("ORGANISM_SCIENTIFIC", "?"),
            "pdbx_gene_src_ncbi_taxonomy_id": src.get("ORGANISM_TAXID", "?"),
            "pdbx_gene_src_variant": "?",
            "pdbx_gene_src_cell_line": "?",
            "pdbx_gene_src_atcc": "?",
            "pdbx_gene_src_organ": "?",
            "pdbx_gene_src_organelle": "?",
            "pdbx_gene_src_cell": "?",
            "pdbx_gene_src_cellular_location": "?",
            "host_org_common_name": "?",
            "pdbx_host_org_scientific_name": src.get("EXPRESSION_SYSTEM", "?"),
            "pdbx_host_org_ncbi_taxonomy_id": src.get("EXPRESSION_SYSTEM_TAXID", "?"),
            "host_org_genus": "?",
            "pdbx_host_org_gene": "?",
            "pdbx_host_org_organ": "?",
            "host_org_species": "?",
            "pdbx_host_org_tissue": "?",
            "pdbx_host_org_tissue_fraction": "?",
            "pdbx_host_org_strain": src.get("EXPRESSION_SYSTEM_STRAIN", "?"),
            "pdbx_host_org_variant": "?",
            "pdbx_host_org_cell_line": "?",
            "pdbx_host_org_atcc": "?",
            "pdbx_host_org_culture_collection": "?",
            "pdbx_host_org_cell": "?",
            "pdbx_host_org_organelle": "?",
            "pdbx_host_org_cellular_location": "?",
            "pdbx_host_org_vector_type": src.get("EXPRESSION_SYSTEM_VECTOR_TYPE", "?"),
            "pdbx_host_org_vector": "?",
            "host_org_details": "?",
            "expression_system_id": "?",
            "plasmid_name": src.get("EXPRESSION_SYSTEM_PLASMID", "?"),
            "plasmid_details": "?",
            "pdbx_description": "?",
        }

    rows = [_row(str(mol_id), src) for mol_id, src in sources.items()]
    if not rows:
        return

    if len(rows) == 1:
        cats["_entity_src_gen."] = rows[0]
    else:
        # Convert to column-oriented dict (loop format)
        keys = list(rows[0].keys())
        cats["_entity_src_gen."] = {k: [r[k] for r in rows] for k in keys}


def _add_struct_ref_seq_dif(cats: dict, pdb_lines: list[str], pdb_id: str) -> None:
    records = []
    for line in pdb_lines:
        if not line.startswith("SEQADV"):
            continue
        res_name = line[12:15].strip()
        chain_id = line[16].strip()
        seq_num = line[18:22].strip()
        db_name = line[24:28].strip()
        db_acc = line[29:38].strip()
        details = line[49:70].strip()
        records.append(
            {
                "res_name": res_name,
                "chain_id": chain_id,
                "seq_num": seq_num,
                "db_name": db_name,
                "db_acc": db_acc,
                "details": details.lower(),
            }
        )

    if not records:
        return

    # Map chain -> align_id from struct_ref_seq
    chain_align: dict[str, str] = {}
    ref_seq = cats.get("_struct_ref_seq.", {})
    for i, strand in enumerate(ref_seq.get("pdbx_strand_id", [])):
        chain_align[strand] = ref_seq.get("align_id", [str(i + 1)])[i]

    align_ids, pdb_ids, mon_ids, strands = [], [], [], []
    seq_nums, ins_codes, db_names, db_accs = [], [], [], []
    db_mon_ids, db_seq_nums, details_list = [], [], []
    auth_seq_nums, ordinals = [], []

    for i, rec in enumerate(records, 1):
        align_id = chain_align.get(rec["chain_id"], "1")
        align_ids.append(align_id)
        pdb_ids.append(pdb_id)
        mon_ids.append(rec["res_name"])
        strands.append(rec["chain_id"])
        seq_nums.append(rec["seq_num"])
        ins_codes.append("?")
        db_names.append(rec["db_name"])
        db_accs.append(rec["db_acc"])
        db_mon_ids.append("?")
        db_seq_nums.append("?")
        details_list.append(rec["details"])
        auth_seq_nums.append("?")
        ordinals.append(str(i))

    cats["_struct_ref_seq_dif."] = {
        "align_id": align_ids,
        "pdbx_pdb_id_code": pdb_ids,
        "mon_id": mon_ids,
        "pdbx_pdb_strand_id": strands,
        "seq_num": seq_nums,
        "pdbx_pdb_ins_code": ins_codes,
        "pdbx_seq_db_name": db_names,
        "pdbx_seq_db_accession_code": db_accs,
        "db_mon_id": db_mon_ids,
        "pdbx_seq_db_seq_num": db_seq_nums,
        "details": details_list,
        "pdbx_auth_seq_num": auth_seq_nums,
        "pdbx_ordinal": ordinals,
    }


def _add_pdbx_poly_seq_scheme(cats: dict, st: gemmi.Structure) -> None:
    # Build full entity_poly_seq per entity: entity_id -> list of (seq_num, mon_id)
    ps = cats.get("_entity_poly_seq.", {})
    entity_full_seq: dict[str, list[tuple[int, str]]] = {}
    for i, eid in enumerate(ps.get("entity_id", [])):
        num = int(ps["num"][i])
        mon = ps["mon_id"][i]
        entity_full_seq.setdefault(eid, []).append((num, mon))

    asym_ids, entity_ids, seq_ids, mon_ids = [], [], [], []
    ndb_seq_nums, pdb_seq_nums, auth_seq_nums = [], [], []
    pdb_mon_ids, auth_mon_ids, strands = [], [], []
    ins_codes, heteros = [], []

    for model in st:
        for chain in model:
            polymer = chain.get_polymer()
            if not polymer:
                continue
            entity = st.get_entity_of(polymer)
            if not entity:
                continue
            entity_id = entity.name
            label_asym = polymer[0].subchain if polymer else chain.name

            # Build label_seq -> (auth_seq_num, res_name) map for observed residues
            obs_map: dict[int, tuple[int, str]] = {}
            for res in polymer:
                ls = res.label_seq
                auth_num = res.seqid.num
                if ls is not None and ls > 0 and auth_num is not None:
                    obs_map[ls] = (auth_num, res.name)

            # Compute auth←→label offset from first observed residue
            if obs_map:
                first_label = min(obs_map.keys())
                first_auth = obs_map[first_label][0]
                offset = first_auth - first_label  # auth = label + offset
            else:
                offset = 0

            full_seq = entity_full_seq.get(entity_id, [])
            for seq_num, mon_id in full_seq:
                ndb = str(seq_num)
                if seq_num in obs_map:
                    auth_num, _ = obs_map[seq_num]
                    pdb_seq = str(auth_num)
                    auth_seq = str(auth_num)
                    pdb_mon = mon_id
                    auth_mon = mon_id
                else:
                    # Unobserved residue: compute expected auth seq from offset
                    pdb_seq = str(seq_num + offset)
                    auth_seq = "?"
                    pdb_mon = "?"
                    auth_mon = "?"

                asym_ids.append(label_asym)
                entity_ids.append(entity_id)
                seq_ids.append(ndb)
                mon_ids.append(mon_id)
                ndb_seq_nums.append(ndb)
                pdb_seq_nums.append(pdb_seq)
                auth_seq_nums.append(auth_seq)
                pdb_mon_ids.append(pdb_mon)
                auth_mon_ids.append(auth_mon)
                strands.append(chain.name)
                ins_codes.append(".")
                heteros.append("n")
        break

    cats["_pdbx_poly_seq_scheme."] = {
        "asym_id": asym_ids,
        "entity_id": entity_ids,
        "seq_id": seq_ids,
        "mon_id": mon_ids,
        "ndb_seq_num": ndb_seq_nums,
        "pdb_seq_num": pdb_seq_nums,
        "auth_seq_num": auth_seq_nums,
        "pdb_mon_id": pdb_mon_ids,
        "auth_mon_id": auth_mon_ids,
        "pdb_strand_id": strands,
        "pdb_ins_code": ins_codes,
        "hetero": heteros,
    }


def _add_pdbx_nonpoly_scheme(cats: dict, st: gemmi.Structure) -> None:
    asym_ids, entity_ids, mon_ids = [], [], []
    ndb_seq_nums, pdb_seq_nums, auth_seq_nums = [], [], []
    pdb_mon_ids, auth_mon_ids, strands, ins_codes = [], [], [], []

    for model in st:
        for chain in model:
            seen_subchains: set[str] = set()
            for res in chain:
                sc = res.subchain
                if not sc or sc in seen_subchains:
                    continue
                subchain = chain.get_subchain(sc)
                if not subchain:
                    continue
                entity = st.get_entity_of(subchain)
                if not entity or entity.entity_type == gemmi.EntityType.Polymer:
                    continue
                seen_subchains.add(sc)

                for j, r in enumerate(subchain, 1):
                    auth_seq = str(r.seqid.num)
                    asym_ids.append(sc)
                    entity_ids.append(entity.name)
                    mon_ids.append(r.name)
                    ndb_seq_nums.append(str(j))
                    pdb_seq_nums.append(auth_seq)
                    auth_seq_nums.append(auth_seq)
                    pdb_mon_ids.append(r.name)
                    auth_mon_ids.append(r.name)
                    strands.append(chain.name)
                    ins_codes.append(".")
        break

    cats["_pdbx_nonpoly_scheme."] = {
        "asym_id": asym_ids,
        "entity_id": entity_ids,
        "mon_id": mon_ids,
        "ndb_seq_num": ndb_seq_nums,
        "pdb_seq_num": pdb_seq_nums,
        "auth_seq_num": auth_seq_nums,
        "pdb_mon_id": pdb_mon_ids,
        "auth_mon_id": auth_mon_ids,
        "pdb_strand_id": strands,
        "pdb_ins_code": ins_codes,
    }


def _add_pdbx_entity_nonpoly(cats: dict, st: gemmi.Structure) -> None:
    entity_ids, names, comp_ids = [], [], []
    for ent in st.entities:
        if ent.entity_type == gemmi.EntityType.Water:
            entity_ids.append(ent.name)
            names.append("water")
            comp_ids.append("HOH")
        elif ent.entity_type == gemmi.EntityType.NonPolymer:
            entity_ids.append(ent.name)
            names.append("?")
            comp_ids.append("?")

    if entity_ids:
        cats["_pdbx_entity_nonpoly."] = {
            "entity_id": entity_ids,
            "name": names,
            "comp_id": comp_ids,
        }


def _add_database_PDB_rev(cats: dict, pdb_lines: list[str], pdb_id: str) -> None:
    seen: dict[int, tuple[str, str]] = {}
    for line in pdb_lines:
        if not line.startswith("REVDAT"):
            continue
        num_str = line[7:10].strip()
        date_str = line[13:22].strip()
        mod_type = line[31:32].strip()
        try:
            num = int(num_str)
        except ValueError:
            continue
        if num not in seen:
            seen[num] = (date_str, mod_type)

    if not seen:
        return

    nums, dates, date_origs, statuses, replaces, mod_types = [], [], [], [], [], []
    # Parse deposition date from HEADER record
    depos_date = "?"
    for line in pdb_lines:
        if line.startswith("HEADER"):
            raw = line[50:59].strip()
            if raw:
                depos_date = _convert_pdb_date(raw)
            break

    for num in sorted(seen.keys()):
        date_str, mod_type = seen[num]
        nums.append(str(num))
        dates.append(_convert_pdb_date(date_str))
        date_origs.append(depos_date if num == 1 else "?")
        statuses.append("?")
        replaces.append(pdb_id)
        mod_types.append(mod_type)

    cats["_database_PDB_rev."] = {
        "num": nums,
        "date": dates,
        "date_original": date_origs,
        "status": statuses,
        "replaces": replaces,
        "mod_type": mod_types,
    }


def _add_atom_sites(cats: dict, pdb_lines: list[str], pdb_id: str) -> None:
    scale: dict[str, str] = {}
    for line in pdb_lines:
        if line.startswith("SCALE"):
            n = line[5]
            scale[f"m{n}1"] = line[10:20].strip()
            scale[f"m{n}2"] = line[20:30].strip()
            scale[f"m{n}3"] = line[30:40].strip()
            scale[f"v{n}"] = line[45:55].strip()

    if not scale:
        return

    d: dict[str, str] = {"entry_id": pdb_id}
    for i in "123":
        for j in "123":
            d[f"fract_transf_matrix[{i}][{j}]"] = scale.get(f"m{i}{j}", "0.000000")
    for i in "123":
        d[f"fract_transf_vector[{i}]"] = scale.get(f"v{i}", "0.00000")
    cats["_atom_sites."] = d


def _add_database_PDB_matrix(cats: dict, pdb_lines: list[str], pdb_id: str) -> None:
    origx: dict[str, str] = {}
    for line in pdb_lines:
        if line.startswith("ORIGX"):
            n = line[5]
            origx[f"m{n}1"] = line[10:20].strip()
            origx[f"m{n}2"] = line[20:30].strip()
            origx[f"m{n}3"] = line[30:40].strip()
            origx[f"t{n}"] = line[45:55].strip()

    if not origx:
        return

    d: dict[str, str] = {"entry_id": pdb_id}
    for i in "123":
        for j in "123":
            d[f"origx[{i}][{j}]"] = origx.get(f"m{i}{j}", "0.000000")
    for i in "123":
        d[f"origx_vector[{i}]"] = origx.get(f"t{i}", "0.00000")
    cats["_database_PDB_matrix."] = d


def _add_diffrn_radiation_wavelength(cats: dict, pdb_lines: list[str]) -> None:
    wavelengths: list[str] = []
    for line in pdb_lines:
        if "REMARK 200" in line and "WAVELENGTH OR RANGE" in line:
            vals = line.split(":")[-1].strip()
            if vals and vals.upper() != "NULL":
                wavelengths = [w.strip() for w in vals.split(",") if w.strip()]
            break

    if not wavelengths:
        return

    cats["_diffrn_radiation_wavelength."] = {
        "id": [str(i) for i in range(1, len(wavelengths) + 1)],
        "wavelength": wavelengths,
        "wt": ["1.0"] * len(wavelengths),
    }


def _add_refine_hist(cats: dict, pdb_lines: list[str], st: gemmi.Structure) -> None:
    n_protein = n_nucleic = n_ligand = n_solvent = 0

    for model in st:
        for chain in model:
            seen_sub: set[str] = set()
            for res in chain:
                sc = res.subchain
                if sc and sc not in seen_sub:
                    subchain = chain.get_subchain(sc)
                    if subchain:
                        entity = st.get_entity_of(subchain)
                        seen_sub.add(sc)
                        for r in subchain:
                            na = sum(1 for _ in r)
                            if entity and entity.entity_type == gemmi.EntityType.Polymer:
                                if entity.polymer_type in (
                                    gemmi.PolymerType.PeptideL,
                                    gemmi.PolymerType.PeptideD,
                                ):
                                    n_protein += na
                                else:
                                    n_nucleic += na
                            elif entity and entity.entity_type == gemmi.EntityType.Water:
                                n_solvent += na
                            else:
                                n_ligand += na
        break

    n_total = n_protein + n_nucleic + n_ligand + n_solvent
    refine_d = cats.get("_refine.", {})
    d_res_high = refine_d.get("ls_d_res_high", "?")
    d_res_low = refine_d.get("ls_d_res_low", "?")

    cats["_refine_hist."] = {
        "pdbx_refine_id": "X-RAY DIFFRACTION",
        "cycle_id": "LAST",
        "pdbx_number_atoms_protein": str(n_protein),
        "pdbx_number_atoms_nucleic_acid": str(n_nucleic),
        "pdbx_number_atoms_ligand": str(n_ligand),
        "number_atoms_solvent": str(n_solvent),
        "number_atoms_total": str(n_total),
        "d_res_high": d_res_high,
        "d_res_low": d_res_low,
    }


def _add_refine_ls_shell(cats: dict, pdb_lines: list[str]) -> None:
    shells: list[dict[str, str]] = []
    in_bins = False

    for line in pdb_lines:
        if not line.startswith("REMARK   3"):
            continue
        text = line[11:].strip()
        if "BIN  RESOLUTION RANGE" in text:
            in_bins = True
            continue
        if in_bins:
            m = re.match(
                r"\s*\d+\s+([\d.]+)\s*-\s*([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)",
                text,
            )
            if m:
                shells.append(
                    {
                        "d_res_low": m.group(1),
                        "d_res_high": m.group(2),
                        "compl": m.group(3),
                        "nwork": m.group(4),
                        "nfree": m.group(5),
                        "rwork": m.group(6),
                        "rfree": m.group(7),
                    }
                )
            elif text.strip() and not text[0].isspace():
                in_bins = False

    if not shells:
        return

    # maxit: highest resolution shell first
    shells = list(reversed(shells))

    cats["_refine_ls_shell."] = {
        "pdbx_refine_id": ["X-RAY DIFFRACTION"] * len(shells),
        "pdbx_total_number_of_bins_used": ["."] * len(shells),
        "d_res_high": [s["d_res_high"] for s in shells],
        "d_res_low": [s["d_res_low"] for s in shells],
        "number_reflns_R_work": [s["nwork"] for s in shells],
        "R_factor_R_work": [s["rwork"] for s in shells],
        "percent_reflns_obs": [f"{float(s['compl']) * 100:.2f}" for s in shells],
        "R_factor_R_free": [s["rfree"] for s in shells],
        "R_factor_R_free_error": ["."] * len(shells),
        "percent_reflns_R_free": ["."] * len(shells),
        "number_reflns_R_free": [s["nfree"] for s in shells],
        "number_reflns_all": ["."] * len(shells),
        "R_factor_all": ["."] * len(shells),
    }


def _add_pdbx_unobs_or_zero_occ_residues(cats: dict, pdb_lines: list[str], st: gemmi.Structure) -> None:
    missing: list[dict[str, str]] = []
    past_header = False
    in_r465 = False

    for line in pdb_lines:
        if line.startswith("REMARK 465"):
            in_r465 = True
            text = line[11:].strip()
            if re.match(r"M\s+RES\s+C\s+SSSEQI", text):
                past_header = True
                continue
            if past_header and text and not text.startswith("MISSING"):
                parts = text.split()
                # Can start with "M" (model num) or just RES C SSEQ
                if len(parts) >= 3 and parts[0].isalpha() and len(parts[0]) == 3:
                    res_name, chain_id, seq_num = parts[0], parts[1], parts[2]
                    missing.append({"res": res_name, "chain": chain_id, "seq": seq_num})
                elif len(parts) >= 4 and parts[1].isalpha() and len(parts[1]) == 3:
                    res_name, chain_id, seq_num = parts[1], parts[2], parts[3]
                    missing.append({"res": res_name, "chain": chain_id, "seq": seq_num})
        elif in_r465:
            in_r465 = False
            past_header = False

    if not missing:
        return

    # Build seq_id map from poly_seq_scheme
    poly_scheme = cats.get("_pdbx_poly_seq_scheme.", {})
    # (label_asym, auth_seq_num) -> seq_id
    seq_map: dict[tuple[str, str], str] = {}
    n_scheme = len(poly_scheme.get("asym_id", []))
    for i in range(n_scheme):
        asym = poly_scheme["asym_id"][i]
        auth = poly_scheme["pdb_seq_num"][i]
        sid = poly_scheme["seq_id"][i]
        seq_map[(asym, auth)] = sid

    ids, models, poly_flags, occ_flags = [], [], [], []
    auth_asyms, auth_comps, auth_seqs, pdb_ins_codes = [], [], [], []
    label_asyms, label_comps, label_seqs = [], [], []

    for i, rec in enumerate(missing, 1):
        # label_asym for polymer = same as auth chain (after our remapping)
        label_asym = rec["chain"]
        label_seq = seq_map.get((label_asym, rec["seq"]), "?")

        ids.append(str(i))
        models.append("1")
        poly_flags.append("Y")
        occ_flags.append("1")
        auth_asyms.append(rec["chain"])
        auth_comps.append(rec["res"])
        auth_seqs.append(rec["seq"])
        pdb_ins_codes.append("?")
        label_asyms.append(label_asym)
        label_comps.append(rec["res"])
        label_seqs.append(label_seq)

    cats["_pdbx_unobs_or_zero_occ_residues."] = {
        "id": ids,
        "PDB_model_num": models,
        "polymer_flag": poly_flags,
        "occupancy_flag": occ_flags,
        "auth_asym_id": auth_asyms,
        "auth_comp_id": auth_comps,
        "auth_seq_id": auth_seqs,
        "PDB_ins_code": pdb_ins_codes,
        "label_asym_id": label_asyms,
        "label_comp_id": label_comps,
        "label_seq_id": label_seqs,
    }


def _add_pdbx_unobs_or_zero_occ_atoms(cats: dict, pdb_lines: list[str], st: gemmi.Structure) -> None:
    """Parse REMARK 470 (missing atoms) and build _pdbx_unobs_or_zero_occ_atoms."""
    atom_records: list[dict] = []
    past_header = False

    for line in pdb_lines:
        if not line.startswith("REMARK 470"):
            continue
        text = line[11:].strip()
        if re.match(r"M\s+RES\s+C\s*SSEQI\s+ATOMS", text, re.I):
            past_header = True
            continue
        if past_header and text and not re.match(r"MISSING|THE\s|REMARK|^$", text, re.I):
            # Format: [M] RES C SSEQ  ATOMS...
            parts = text.split()
            if not parts:
                continue
            # Determine if first token is model number or residue name
            if parts[0].isdigit():
                if len(parts) >= 4:
                    res_name, chain, seq = parts[1], parts[2], parts[3]
                    atoms = parts[4:]
                else:
                    continue
            else:
                if len(parts) >= 3:
                    res_name, chain, seq = parts[0], parts[1], parts[2]
                    atoms = parts[3:]
                else:
                    continue
            ins = "?"
            for atom_name in atoms:
                atom_records.append(
                    {
                        "res": res_name,
                        "chain": chain,
                        "seq": seq,
                        "ins": ins,
                        "atom": atom_name,
                    }
                )

    if not atom_records:
        return

    # Build label_seq lookup from poly_seq_scheme
    poly_scheme = cats.get("_pdbx_poly_seq_scheme.", {})
    seq_map: dict[tuple[str, str], tuple[str, str]] = {}
    n_scheme = len(poly_scheme.get("asym_id", []))
    for i in range(n_scheme):
        asym = poly_scheme["asym_id"][i]
        auth_seq = poly_scheme["pdb_seq_num"][i]
        label_seq = poly_scheme["seq_id"][i]
        seq_map[(asym, auth_seq)] = (asym, label_seq)

    ids, models, poly_flags, occ_flags = [], [], [], []
    auth_asyms, auth_comps, auth_seqs, ins_codes = [], [], [], []
    auth_atoms, label_alts, label_asyms, label_comps, label_seqs, label_atoms = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i, rec in enumerate(atom_records, 1):
        label_asym, label_seq = seq_map.get((rec["chain"], rec["seq"]), (rec["chain"], "?"))
        ids.append(str(i))
        models.append("1")
        poly_flags.append("Y")
        occ_flags.append("1")
        auth_asyms.append(rec["chain"])
        auth_comps.append(rec["res"])
        auth_seqs.append(rec["seq"])
        ins_codes.append(rec["ins"])
        auth_atoms.append(rec["atom"])
        label_alts.append("?")
        label_asyms.append(label_asym)
        label_comps.append(rec["res"])
        label_seqs.append(label_seq)
        label_atoms.append(rec["atom"])

    cats["_pdbx_unobs_or_zero_occ_atoms."] = {
        "id": ids,
        "PDB_model_num": models,
        "polymer_flag": poly_flags,
        "occupancy_flag": occ_flags,
        "auth_asym_id": auth_asyms,
        "auth_comp_id": auth_comps,
        "auth_seq_id": auth_seqs,
        "PDB_ins_code": ins_codes,
        "auth_atom_id": auth_atoms,
        "label_alt_id": label_alts,
        "label_asym_id": label_asyms,
        "label_comp_id": label_comps,
        "label_seq_id": label_seqs,
        "label_atom_id": label_atoms,
    }


def _fix_struct_conn(cats: dict) -> None:
    """Reorder _struct_conn columns to match maxit and add missing columns."""
    d = cats.get("_struct_conn.", {})
    if not d:
        return
    n = _nrows(cats, "_struct_conn.")
    if n == 0:
        return

    # Build label_asym/seq lookup maps from poly_seq_scheme and nonpoly_scheme
    poly_scheme = cats.get("_pdbx_poly_seq_scheme.", {})
    # (auth_asym, auth_seq) -> (label_asym, label_seq)
    poly_map: dict[tuple[str, str], tuple[str, str]] = {}
    n_ps = len(poly_scheme.get("asym_id", []))
    for i in range(n_ps):
        asym = poly_scheme["asym_id"][i]
        auth_seq = poly_scheme["pdb_seq_num"][i]
        label_seq = poly_scheme["seq_id"][i]
        poly_map[(asym, auth_seq)] = (asym, label_seq)

    nonpoly = cats.get("_pdbx_nonpoly_scheme.", {})
    # (auth_asym, auth_seq, comp) -> label_asym
    nonpoly_map: dict[tuple[str, str, str], str] = {}
    n_np = len(nonpoly.get("asym_id", []))
    for i in range(n_np):
        la = nonpoly["asym_id"][i]
        aa = nonpoly["pdb_strand_id"][i]
        acomp = nonpoly["mon_id"][i]
        aseq = nonpoly["pdb_seq_num"][i]
        nonpoly_map[(aa, aseq, acomp)] = la

    def _get(key: str) -> list:
        v = d.get(key, ["?"] * n)
        return v if isinstance(v, list) else [v]

    # Existing columns
    ids = _get("id")
    conn_types = _get("conn_type_id")
    p1_auth_asym = _get("ptnr1_auth_asym_id")
    p1_auth_comp = _get("ptnr1_label_comp_id")  # gemmi uses label_comp = auth_comp
    p1_auth_seq = _get("ptnr1_auth_seq_id")
    p1_label_atom = _get("ptnr1_label_atom_id")
    p1_label_alt = _get("pdbx_ptnr1_label_alt_id")
    p1_sym = _get("ptnr1_symmetry")
    p2_auth_asym = _get("ptnr2_auth_asym_id")
    p2_auth_comp = _get("ptnr2_label_comp_id")
    p2_auth_seq = _get("ptnr2_auth_seq_id")
    p2_label_atom = _get("ptnr2_label_atom_id")
    p2_label_alt = _get("pdbx_ptnr2_label_alt_id")
    p2_sym = _get("ptnr2_symmetry")
    details = _get("details")
    dist_vals = _get("pdbx_dist_value")

    # Compute label columns
    p1_label_asym, p1_label_seq, p2_label_asym, p2_label_seq = [], [], [], []
    for i in range(n):
        aa1, ac1, as1 = p1_auth_asym[i], p1_auth_comp[i], p1_auth_seq[i]
        la1, ls1 = poly_map.get((aa1, as1), (None, None))
        if la1 is None:
            la1 = nonpoly_map.get((aa1, as1, ac1), aa1)
            ls1 = "."
        p1_label_asym.append(la1)
        p1_label_seq.append(ls1 if ls1 else ".")

        aa2, ac2, as2 = p2_auth_asym[i], p2_auth_comp[i], p2_auth_seq[i]
        la2, ls2 = poly_map.get((aa2, as2), (None, None))
        if la2 is None:
            la2 = nonpoly_map.get((aa2, as2, ac2), aa2)
            ls2 = "."
        p2_label_asym.append(la2)
        p2_label_seq.append(ls2 if ls2 else ".")

    # Format dist_value to 3dp
    fmt_dist = []
    for v in dist_vals:
        try:
            fmt_dist.append(f"{float(v):.3f}")
        except (ValueError, TypeError):
            fmt_dist.append(str(v))

    q = ["?"] * n
    cats["_struct_conn."] = {
        "id": ids,
        "conn_type_id": conn_types,
        "pdbx_leaving_atom_flag": q,
        "pdbx_PDB_id": q,
        "ptnr1_label_asym_id": p1_label_asym,
        "ptnr1_label_comp_id": p1_auth_comp,
        "ptnr1_label_seq_id": p1_label_seq,
        "ptnr1_label_atom_id": p1_label_atom,
        "pdbx_ptnr1_label_alt_id": p1_label_alt,
        "pdbx_ptnr1_PDB_ins_code": q,
        "pdbx_ptnr1_standard_comp_id": q,
        "ptnr1_symmetry": p1_sym,
        "ptnr2_label_asym_id": p2_label_asym,
        "ptnr2_label_comp_id": p2_auth_comp,
        "ptnr2_label_seq_id": p2_label_seq,
        "ptnr2_label_atom_id": p2_label_atom,
        "pdbx_ptnr2_label_alt_id": p2_label_alt,
        "pdbx_ptnr2_PDB_ins_code": q,
        "ptnr1_auth_asym_id": p1_auth_asym,
        "ptnr1_auth_comp_id": p1_auth_comp,
        "ptnr1_auth_seq_id": p1_auth_seq,
        "ptnr2_auth_asym_id": p2_auth_asym,
        "ptnr2_auth_comp_id": p2_auth_comp,
        "ptnr2_auth_seq_id": p2_auth_seq,
        "ptnr2_symmetry": p2_sym,
        "pdbx_ptnr3_label_atom_id": q,
        "pdbx_ptnr3_label_seq_id": q,
        "pdbx_ptnr3_label_comp_id": q,
        "pdbx_ptnr3_label_asym_id": q,
        "pdbx_ptnr3_label_alt_id": q,
        "pdbx_ptnr3_PDB_ins_code": q,
        "details": details,
        "pdbx_dist_value": fmt_dist,
        "pdbx_value_order": q,
        "pdbx_role": q,
    }


def _add_pdbx_validate_close_contact(cats: dict, pdb_lines: list[str]) -> None:
    contacts: list[dict] = []
    in_section = False

    for line in pdb_lines:
        if not line.startswith("REMARK 500"):
            if in_section:
                in_section = False
            continue
        text = line[11:].strip()
        if "ATM1  RES C  SSEQI" in text:
            in_section = True
            continue
        if in_section and text:
            parts = text.split()
            if len(parts) >= 9:
                try:
                    float(parts[-1])
                    contacts.append(
                        {
                            "a1": parts[0],
                            "r1": parts[1],
                            "c1": parts[2],
                            "s1": parts[3],
                            "a2": parts[4],
                            "r2": parts[5],
                            "c2": parts[6],
                            "s2": parts[7],
                            "dist": parts[8],
                        }
                    )
                except (ValueError, IndexError):
                    in_section = False

    if not contacts:
        return

    cats["_pdbx_validate_close_contact."] = {
        "id": [str(i) for i in range(1, len(contacts) + 1)],
        "PDB_model_num": ["1"] * len(contacts),
        "auth_atom_id_1": [c["a1"] for c in contacts],
        "auth_asym_id_1": [c["c1"] for c in contacts],
        "auth_comp_id_1": [c["r1"] for c in contacts],
        "auth_seq_id_1": [c["s1"] for c in contacts],
        "PDB_ins_code_1": ["?"] * len(contacts),
        "label_alt_id_1": ["?"] * len(contacts),
        "auth_atom_id_2": [c["a2"] for c in contacts],
        "auth_asym_id_2": [c["c2"] for c in contacts],
        "auth_comp_id_2": [c["r2"] for c in contacts],
        "auth_seq_id_2": [c["s2"] for c in contacts],
        "PDB_ins_code_2": ["?"] * len(contacts),
        "label_alt_id_2": ["?"] * len(contacts),
        "dist": [c["dist"] for c in contacts],
    }


def _add_pdbx_validate_torsion(cats: dict, pdb_lines: list[str]) -> None:
    torsions: list[dict] = []
    in_section = False

    for line in pdb_lines:
        if not line.startswith("REMARK 500"):
            if in_section:
                in_section = False
            continue
        text = line[11:].strip()
        if re.match(r"M\s+RES\s+CSSEQI.*PSI.*PHI", text):
            in_section = True
            continue
        if in_section and text:
            m = re.match(
                r"\s*(\w{3})\s+(\w)\s+(-?\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)",
                text,
            )
            if m:
                torsions.append(
                    {
                        "comp": m.group(1),
                        "chain": m.group(2),
                        "seq": m.group(3),
                        "psi": m.group(4),
                        "phi": m.group(5),
                    }
                )

    if not torsions:
        return

    cats["_pdbx_validate_torsion."] = {
        "id": [str(i) for i in range(1, len(torsions) + 1)],
        "PDB_model_num": ["1"] * len(torsions),
        "auth_comp_id": [t["comp"] for t in torsions],
        "auth_asym_id": [t["chain"] for t in torsions],
        "auth_seq_id": [t["seq"] for t in torsions],
        "PDB_ins_code": ["?"] * len(torsions),
        "label_alt_id": ["?"] * len(torsions),
        "phi": [t["phi"] for t in torsions],
        "psi": [t["psi"] for t in torsions],
    }


def _fix_atom_type(cats: dict, st: gemmi.Structure) -> None:
    elements: set[str] = set()
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    elements.add(atom.element.name.upper())
        break
    cats["_atom_type."] = {"symbol": sorted(elements)}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

# Category output order
_CATEGORY_ORDER = [
    "_pdbx_audit_revision_history.",
    "_audit_author.",
    "_struct.",
    "_database_2.",
    "_struct_keywords.",
    "_exptl.",
    "_pdbx_database_status.",
    "_citation.",
    "_citation_author.",
    "_entity_poly.",
    "_entity_poly_seq.",
    "_entity.",
    "_entity_src_gen.",
    "_struct_ref.",
    "_struct_ref_seq.",
    "_struct_ref_seq_dif.",
    "_pdbx_poly_seq_scheme.",
    "_pdbx_nonpoly_scheme.",
    "_chem_comp.",
    "_struct_conn.",
    "_struct_asym.",
    "_struct_conf_type.",
    "_struct_conf.",
    "_struct_mon_prot_cis.",
    "_struct_sheet.",
    "_struct_sheet_order.",
    "_struct_sheet_range.",
    "_pdbx_struct_assembly.",
    "_pdbx_struct_assembly_gen.",
    "_pdbx_struct_assembly_prop.",
    "_pdbx_struct_oper_list.",
    "_database_PDB_rev.",
    "_exptl_crystal_grow.",
    "_exptl_crystal.",
    "_cell.",
    "_symmetry.",
    "_atom_sites.",
    "_diffrn_source.",
    "_diffrn_detector.",
    "_diffrn.",
    "_diffrn_radiation.",
    "_diffrn_radiation_wavelength.",
    "_reflns.",
    "_reflns_shell.",
    "_refine.",
    "_refine_ls_restr.",
    "_refine_ls_shell.",
    "_refine_hist.",
    "_software.",
    "_pdbx_validate_close_contact.",
    "_pdbx_validate_torsion.",
    "_pdbx_unobs_or_zero_occ_atoms.",
    "_pdbx_unobs_or_zero_occ_residues.",
    "_atom_type.",
    "_database_PDB_matrix.",
    "_entry.",
    "_pdbx_entity_nonpoly.",
    "_pdbx_struct_sheet_hbond.",
    "_struct_conn_type.",
    "_refine_analyze.",
    "_atom_site.",
]


def _write_output(
    cats: Mapping[str, dict[str, list[object]]],
    pdb_id: str,
    output_path: str,
) -> None:
    doc = gemmi.cif.Document()
    block = doc.add_new_block(pdb_id)

    options = gemmi.cif.WriteOptions()
    options.misuse_hash = True
    options.align_loops = 40
    options.align_pairs = 45

    written: set[str] = set()
    for cat in _CATEGORY_ORDER:
        if cat in cats:
            _write_cat(block, cat, cats[cat])
            written.add(cat)

    for cat in sorted(cats.keys()):
        if cat not in written:
            _write_cat(block, cat, cats[cat])

    doc.write_file(output_path, options=options)
    _postprocess_quotes(output_path)


def _postprocess_quotes(path: str) -> None:
    """Fix CIF quoting and alignment to match maxit conventions."""
    _SQ_TOKEN = re.compile(r"(?<!\w)'([^']*)'")
    _NEEDS_Q = re.compile(r"[\s\(\)\[\]]")
    _KV_LINE = re.compile(r"^(_[a-zA-Z0-9_.\-]+(?:\[[^\]]*\])*)(\s+)(.*\S)\s*$")
    # Match bare (unquoted) tokens containing ( or ) that need double-quoting in loops
    _BARE_PAREN = re.compile(r'(?<!["\'])(\b\w[\w.()/\\-]*\([^)\s"\']*\))(?!\w*["\'])')

    def _replace_sq(line: str) -> str:
        def repl(m: re.Match) -> str:
            val = m.group(1)
            if _NEEDS_Q.search(val):
                return f'"{val}"'
            return val

        return _SQ_TOKEN.sub(repl, line)

    def _quote_bare_parens(line: str) -> str:
        """Add double quotes around bare (unquoted) tokens with parentheses.
        Skips content already inside double-quoted strings."""
        result = []
        i = 0
        while i < len(line):
            if line[i] == '"':
                # Find closing double-quote; keep quoted segment unchanged
                j = line.find('"', i + 1)
                if j == -1:
                    result.append(line[i:])
                    break
                result.append(line[i : j + 1])
                i = j + 1
            else:
                # Unquoted segment up to the next double-quote (or EOL)
                j = line.find('"', i)
                if j == -1:
                    segment = line[i:]
                    result.append(_BARE_PAREN.sub(lambda m: f'"{m.group(1)}"', segment))
                    break
                else:
                    segment = line[i:j]
                    result.append(_BARE_PAREN.sub(lambda m: f'"{m.group(1)}"', segment))
                    i = j
        return "".join(result)

    def _fix_alignment(block_lines: list[str]) -> list[str]:
        """Re-align a group of key-value lines to max_key_len + 3 spaces."""
        if len(block_lines) <= 1:
            return block_lines
        # Find max key length
        max_key = 0
        for bl in block_lines:
            m = _KV_LINE.match(bl.rstrip("\n\r"))
            if m:
                max_key = max(max_key, len(m.group(1)))
        if max_key == 0:
            return block_lines
        target = max_key + 3  # key padded to this width (3 spaces min), then value
        result = []
        for bl in block_lines:
            stripped = bl.rstrip("\n\r")
            m = _KV_LINE.match(stripped)
            if m:
                key = m.group(1)
                val = m.group(3)
                result.append(key.ljust(target) + val + "\n")
            else:
                result.append(bl)
        return result

    raw_lines = Path(path).read_text().splitlines(keepends=True)

    # Pass 1: join tag-on-separate-line with value-on-next-line
    # gemmi sometimes puts long values on the line after the tag
    # Only applies to single-row KV pairs (NOT loop column headers)
    joined: list[str] = []
    i = 0
    in_loop_pass1 = False
    in_semi_pass1 = False
    while i < len(raw_lines):
        line = raw_lines[i]
        s = line.rstrip("\n\r")
        if in_semi_pass1:
            if s == ";":
                in_semi_pass1 = False
            joined.append(line)
            i += 1
            continue
        if s.startswith(";"):
            in_semi_pass1 = True
            joined.append(line)
            i += 1
            continue
        if s == "loop_":
            in_loop_pass1 = True
            joined.append(line)
            i += 1
            continue
        if s == "#" or s == "":
            in_loop_pass1 = False
            joined.append(line)
            i += 1
            continue
        # A tag-only line: starts with _ but has no value on it
        if not in_loop_pass1 and s.startswith("_") and " " not in s and "\t" not in s:
            # Check if next line has the value (quoted string or bare word)
            if i + 1 < len(raw_lines):
                nxt = raw_lines[i + 1].rstrip("\n\r")
                if (
                    nxt
                    and not nxt.startswith("_")
                    and not nxt.startswith("#")
                    and not nxt.startswith(";")
                    and not nxt.startswith("loop_")
                ):
                    # Join: tag + " " + value
                    joined.append(s + " " + nxt + "\n")
                    i += 2
                    continue
        joined.append(line)
        i += 1

    # Pass 2: single-quoted → double-quoted / unquoted; also collect KV groups for alignment
    out = []
    in_block = False
    in_loop = False
    loop_before_block = False  # preserve loop state across semicolon blocks
    kv_group: list[str] = []  # current key-value group (same category, not a loop)
    kv_prefix: str = ""

    def flush_kv() -> None:
        fixed = _fix_alignment(kv_group)
        for fl in fixed:
            out.append(_replace_sq(fl))
        kv_group.clear()

    for line in joined:
        stripped = line.rstrip("\n\r")
        if in_block:
            if stripped == ";":
                in_block = False
                in_loop = loop_before_block  # restore loop state after block
            out.append(line)
            continue
        if stripped.startswith(";"):
            flush_kv()
            kv_prefix = ""
            loop_before_block = in_loop
            in_block = True
            in_loop = False
            out.append(line)
            continue
        if stripped.startswith("loop_"):
            flush_kv()
            kv_prefix = ""
            in_loop = True
            out.append(line)
            continue
        if stripped == "#" or stripped == "":
            flush_kv()
            kv_prefix = ""
            in_loop = False
            out.append(line)
            continue
        if in_loop:
            out.append(_quote_bare_parens(_replace_sq(line)))
            continue
        # Key-value line?
        m = _KV_LINE.match(stripped)
        if m:
            key = m.group(1)
            # Extract category prefix (up to and including dot)
            dot = key.find(".")
            prefix = key[: dot + 1] if dot >= 0 else key
            if prefix != kv_prefix:
                flush_kv()
                kv_prefix = prefix
            kv_group.append(line)
        else:
            flush_kv()
            kv_prefix = ""
            out.append(_replace_sq(line))

    flush_kv()
    Path(path).write_text("".join(out))


_CIF_NEEDS_QUOTE = re.compile(r"[\s\(\)\[\]]")
_CIF_RESERVED = {"data_", "loop_", "save_", "stop_", "global_"}


def _cif_quote(s: str) -> str:
    """Return CIF-formatted value: add double quotes when the value needs it."""
    if s in ("?", "."):
        return s
    if not s:
        return '""'
    # Already properly quoted (starts and ends with the same quote char)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s
    # Already a semicolon block
    if s.startswith(";"):
        return s
    needs = bool(_CIF_NEEDS_QUOTE.search(s)) or s.lower() in _CIF_RESERVED
    if needs:
        if '"' not in s:
            return f'"{s}"'
        elif "'" not in s:
            return f"'{s}'"
        else:
            # Fall back to semicolon block
            return f"\n;{s}\n;"
    return s


def _cif_val(v: object) -> object:
    """Convert string '?' -> None and '.' -> False for CIF special values."""
    if v == "?":
        return None
    if v == ".":
        return False
    return v


def _cif_vals(lst: list) -> list:
    return [_cif_val(v) for v in lst]


def _write_cat(block: gemmi.cif.Block, cat: str, data: dict) -> None:
    if not data:
        return

    # Normalise: ensure all values are lists with proper CIF special values
    norm: dict[str, list] = {}
    for k, v in data.items():
        raw = v if isinstance(v, list) else [v]
        norm[k] = _cif_vals(raw)

    first_val = next(iter(norm.values()))
    if not first_val and first_val != False and first_val != 0:  # noqa: E712
        if not any(True for _ in first_val):
            return

    n = len(norm[next(iter(norm))])
    if n == 0:
        return

    if n == 1:
        # Single-row → write as key-value pairs
        for k, vals in norm.items():
            v = vals[0]
            if v is None:
                block.set_pair(f"{cat}{k}", "?")
            elif v is False:
                block.set_pair(f"{cat}{k}", ".")
            else:
                block.set_pair(f"{cat}{k}", _cif_quote(str(v)))
    else:
        # Multi-row → write as loop
        block.set_mmcif_category(cat, norm)


def _generate_four_digit_ids(start: int = 0):
    """Yield sequential 4-digit IDs: 0000, 0001, 0002, ...
    If the counter exceeds 9999, this will raise an exception to avoid generating non-4-digit IDs.
    """
    i = start
    while i <= 9999:
        yield f"{i:04d}"
        i += 1
    raise ValueError("Counter exceeded 9999")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _parse_pdb_author_list(text: str) -> list[str]:
    """Convert PDB-style 'A.MINAMI,T.OSE' to CIF 'Minami, A.' list."""
    authors = []
    for auth in text.split(","):
        auth = auth.strip()
        if not auth:
            continue
        parts = auth.split(".")
        if len(parts) >= 2:
            surname = parts[-1].strip().title()
            initials = ".".join(p.upper() for p in parts[:-1] if p) + "."
            authors.append(f"{surname}, {initials}")
        else:
            authors.append(auth.title())
    return authors


def _parse_pdb_source(pdb_lines: list[str]) -> dict[str, dict[str, str]]:
    """Parse SOURCE records from PDB file."""
    text = ""
    for line in pdb_lines:
        if line.startswith("SOURCE"):
            text += line[10:].rstrip() + " "

    sources: dict[str, dict[str, str]] = {}
    current = "1"
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            k, v = item.split(":", 1)
            k, v = k.strip(), v.strip()
            if k == "MOL_ID":
                current = v
                sources.setdefault(current, {})
            else:
                sources.setdefault(current, {})[k] = v
    return sources


def _convert_pdb_date(date_str: str) -> str:
    """Convert DD-MMM-YY or DD-MMM-YYYY to YYYY-MM-DD."""
    MONTHS = {
        "JAN": "01",
        "FEB": "02",
        "MAR": "03",
        "APR": "04",
        "MAY": "05",
        "JUN": "06",
        "JUL": "07",
        "AUG": "08",
        "SEP": "09",
        "OCT": "10",
        "NOV": "11",
        "DEC": "12",
    }
    parts = date_str.split("-")
    if len(parts) != 3:
        return date_str
    day = parts[0].zfill(2)
    month = MONTHS.get(parts[1].upper(), "01")
    yr = int(parts[2])
    if yr >= 1000:
        year = yr
    else:
        year = yr + (2000 if yr < 50 else 1900)
    return f"{year}-{month}-{day}"


def _title_case_journal(journal: str) -> str:
    """Title-case a journal abbreviation like maxit does.

    Handles dot-separated segments like 'CHEM.BIOL.' → 'Chem.Biol.'
    """
    # Split on spaces first
    words = journal.split()
    result = []
    for w in words:
        if w.upper() in ("AND", "OF", "THE", "IN", "FOR"):
            result.append(w.capitalize())
        elif "." in w:
            # Handle abbreviations like 'CHEM.BIOL.' → 'Chem.Biol.'
            segs = w.split(".")
            result.append(".".join(s.capitalize() for s in segs))
        elif re.match(r"[A-Z]{2,}", w):
            result.append(w[0].upper() + w[1:].lower())
        else:
            result.append(w.capitalize())
    return " ".join(result)


def _title_case_word(word: str) -> str:
    """Capitalize a word, including each part of hyphenated compounds."""
    return "-".join(part.capitalize() for part in word.split("-"))


def _title_case_words(text: str) -> str:
    """Title-case a citation title (capitalize every word, like maxit)."""
    if not text or text == "?":
        return text
    return " ".join(_title_case_word(w) for w in text.split())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDB to mmCIF format.")
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="Input pdb file or input directory. If a directory is given, all .pdb files in it will be converted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output",
        required=True,
        help="Output mmCIF file or directory. If a directory is given, all .cif files in it will be created.",
    )
    parser.add_argument(
        "-p",
        "--pdb_id",
        dest="pdb_id",
        default="xxxx",
        help="PDB ID to use in mmCIF output. If input pdb file already has a PDB ID, this will be ignored.",
    )
    parser.add_argument(
        "--no-write_seqres",
        dest="write_seqres",
        help="Do not write pdb_seqres.txt file with sequences extracted from the CIF",
        action="store_false",
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

    input_path = Path(args.input)

    if input_path.is_file():
        logger.debug(f"Processing single file: {input_path}")
        pdb_to_cif(
            args.input,
            args.output,
            default_pdb_id=args.pdb_id,
            write_seqres=args.write_seqres,
        )
        return
    elif input_path.is_dir():
        logger.debug(f"Processing directory: {input_path}")
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        pdb_files = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix == ".pdb")
        logger.debug(f"Found {len(pdb_files)} pdb files in {input_path}")

        id_gen = _generate_four_digit_ids()
        for pdb_file in pdb_files:
            # If filename contains a 4 digits (e.g. 1abc.pdb), use that as the PDB ID; otherwise, generate sequential 4-digit IDs starting from 0000.
            match = re.search(r"^[A-Za-z0-9]{4}$", pdb_file.stem)
            if match:
                pdbid = match.group(0).lower()
                logger.debug(f"Extracted 4-digit ID {pdbid} from filename {pdb_file.name}")
            else:
                pdbid = next(id_gen)
                logger.info(f"The filename of {pdb_file.name} is not 4-character, using generated ID {pdbid}")
            output_path = output_dir / f"{pdbid}.cif"
            logger.debug(f"Converting {pdb_file} to {output_path} with a 4−digit ID {pdbid}")
            pdb_to_cif(
                str(pdb_file),
                str(output_path),
                default_pdb_id=pdbid,
                write_seqres=False,
            )
        if args.write_seqres:
            seqres_path = output_dir / "pdb_seqres.txt"
            cif_to_seqres(str(output_dir), str(seqres_path))
        return
    else:
        raise ValueError(f"Your input {args.input} does not exist.")


if __name__ == "__main__":
    main()
