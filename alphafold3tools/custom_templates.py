"""Build a template database from user-supplied structure files (``--custom_template``).

The template search normally runs hmmsearch against the wwPDB pdb_seqres.txt and
reads the hit structures from a directory of PDB mmCIF files. With
``--custom_template``, both are built from the PDB/mmCIF files in a directory
given by the user instead, so that unpublished structures or predicted models can
be used as templates:

1. PDB files (``X.pdb`` / ``X.pdb.gz``) are converted to mmCIF with pdbtocif.
2. Each mmCIF file is normalized into ``<template id>.cif`` in a work directory.
   The template search looks a hit up by the name in pdb_seqres.txt, which must
   match the file name and the data block name, so the template id is used for
   all of them (and for ``_entry.id``). Fields that predicted models lack (the
   release date and the canonical one-letter sequence) are filled in.
3. The mmCIF converted from ``X.pdb`` is also saved as ``X.cif`` next to it,
   without ever replacing an existing file.
4. A pdb_seqres.txt equivalent is written from the normalized files.

The work directory is then used as ``pdb_database_path`` / ``seqres_database_path``
of the regular template search.
"""

import dataclasses
import datetime
import os
import re
import tempfile
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Final

import gemmi
from loguru import logger

import alphafold3tools.structure_stores as structure_stores
from alphafold3tools.ciftoseqres import cif_to_seqres
from alphafold3tools.pdb_download import SEQRES_FILENAME
from alphafold3tools.pdbtocif import pdb_to_cif
from alphafold3tools.searchtemplates import PROTEIN_CHAIN, _parse_hit_metadata_with_gemmi
from alphafold3tools.structure.mmcif_parser import mmcifcontent

# In order of preference when several files have the same name.
STRUCTURE_SUFFIXES: Final[tuple[str, ...]] = (".cif", ".cif.gz", ".pdb", ".pdb.gz")
_PDB_SUFFIXES: Final[tuple[str, ...]] = (".pdb", ".pdb.gz")
# Same placeholder that pdbtocif writes for structures without a revision history.
PLACEHOLDER_RELEASE_DATE: Final[str] = "1970-01-01"
# Hit names are parsed as "<id>_<chain>" with <id> matching [a-z0-9]{4,} and
# <chain> matching \w+ (see _HIT_DESCRIPTION_REGEX in searchtemplates).
_MIN_TEMPLATE_ID_LENGTH: Final[int] = 4
_CHAIN_ID_REGEX: Final[re.Pattern[str]] = re.compile(r"\w+")
_REQUIRED_CATEGORIES: Final[tuple[str, ...]] = (
    "_entity.",
    "_entity_poly.",
    "_entity_poly_seq.",
    "_pdbx_poly_seq_scheme.",
    "_struct_asym.",
    "_atom_site.",
)


class CustomTemplateError(RuntimeError):
    """Raised when no template database can be built from a custom template directory."""


@dataclasses.dataclass(frozen=True)
class CustomTemplateDatabase:
    """A template database built from a custom template directory.

    Attributes:
      pdb_database_path: Directory of the normalized mmCIF files.
      seqres_database_path: The pdb_seqres.txt equivalent of the normalized files.
      sources: Template id -> the file in the custom template directory that the
        template was made from.
    """

    pdb_database_path: Path
    seqres_database_path: Path
    sources: Mapping[str, Path]


def _split_structure_suffix(path: Path) -> tuple[str, str] | None:
    """Returns ``(name, lowercase suffix)`` of a structure file, or None for other files."""
    lower_name = path.name.lower()
    for suffix in STRUCTURE_SUFFIXES:
        if lower_name.endswith(suffix) and len(lower_name) > len(suffix):
            return path.name[: -len(suffix)], suffix
    return None


def _structure_name(path: Path) -> str:
    split = _split_structure_suffix(path)
    if split is None:
        raise ValueError(f"{path} is not a PDB or mmCIF file.")
    return split[0]


def _is_pdb(path: Path) -> bool:
    return path.name.lower().endswith(_PDB_SUFFIXES)


def find_custom_template_files(directory: str | os.PathLike[str]) -> list[Path]:
    """Returns the structure files directly under ``directory``, one per name.

    PDB and mmCIF files, optionally gzip-compressed, are collected; suffixes are
    matched case-insensitively and subdirectories are not searched. Files whose
    names differ only in the suffix, in case or in Unicode normalization are the
    same structure (e.g. ``X.pdb`` and the ``X.cif`` converted from it), so only
    the first one in ``STRUCTURE_SUFFIXES`` order is used and the others are
    reported.
    """
    candidates: dict[str, list[tuple[int, Path]]] = {}
    for path in sorted(Path(directory).iterdir()):
        split = _split_structure_suffix(path)
        if split is None or not path.is_file():
            continue
        name, suffix = split
        # Case-insensitive file systems (e.g. macOS) also ignore these differences.
        key = unicodedata.normalize("NFC", name).casefold()
        candidates.setdefault(key, []).append((STRUCTURE_SUFFIXES.index(suffix), path))
    selected = []
    for _, paths in sorted(candidates.items()):
        paths.sort()
        chosen = paths[0][1]
        for _, ignored in paths[1:]:
            message = f"{ignored} is not used as a template because {chosen.name} has the same name."
            if _is_pdb(ignored) and not _is_pdb(chosen) and ignored.stat().st_mtime > chosen.stat().st_mtime:
                message += f" {ignored.name} is newer; delete {chosen.name} to convert {ignored.name} again."
            logger.warning(message)
        selected.append(chosen)
    return selected


def _template_ids(names: Iterable[str]) -> list[str]:
    """Returns a unique template id for each structure name.

    Names are lowercased and stripped of characters other than [a-z0-9], padded
    with "0" to the minimum length, and given a numeric suffix when duplicated.
    """
    used: set[str] = set()
    template_ids = []
    for name in names:
        base = re.sub(r"[^a-z0-9]", "", name.lower()).ljust(_MIN_TEMPLATE_ID_LENGTH, "0")
        template_id = base
        suffix = 1
        while template_id in used:
            template_id = f"{base}{suffix}"
            suffix += 1
        used.add(template_id)
        template_ids.append(template_id)
    return template_ids


def _create_new_file(path: Path, text: str) -> None:
    """Creates ``path`` with ``text``, raising FileExistsError instead of replacing anything.

    The text is written to a temporary file and hard-linked into place, so the file
    appears complete to other runs sharing the directory. Where hard links are not
    supported, the file is created with O_EXCL instead.
    """
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.chmod(tmp_path, 0o644)
        try:
            os.link(tmp_path, path)
        except FileExistsError:
            raise
        except OSError:
            with open(path, "x") as f:
                f.write(text)
    finally:
        tmp_path.unlink(missing_ok=True)


def _save_converted_mmcif(pdb_path: Path, converted: Path) -> None:
    """Saves the mmCIF converted from ``X.pdb`` / ``X.pdb.gz`` as ``X.cif`` next to it."""
    cif_path = pdb_path.with_name(f"{_structure_name(pdb_path)}.cif")
    try:
        _create_new_file(cif_path, converted.read_text())
    except FileExistsError:
        logger.warning(f"{cif_path} already exists, so the mmCIF file converted from {pdb_path.name} is not saved.")
        return
    logger.info(f"Converted {pdb_path} to {cif_path}.")


def _canonical_sequence(mon_ids: Sequence[str]) -> str:
    """Returns the one-letter sequence using parent residues (e.g. MSE -> M), X if unknown."""
    letters = []
    for mon_id in mon_ids:
        residue = gemmi.find_tabulated_residue(mon_id)
        letter = residue.one_letter_code.upper() if residue is not None else ""
        letters.append(letter if letter.strip() else "X")
    return "".join(letters)


def _fill_canonical_sequences(block: gemmi.cif.Block) -> None:
    """Fills missing ``_entity_poly.pdbx_seq_one_letter_code_can`` from ``_entity_poly_seq``."""
    entity_poly = block.get_mmcif_category("_entity_poly.")
    entity_ids = entity_poly["entity_id"]
    sequences = entity_poly.get("pdbx_seq_one_letter_code_can") or [None] * len(entity_ids)
    if all(isinstance(sequence, str) and sequence for sequence in sequences):
        return
    poly_seq = block.get_mmcif_category("_entity_poly_seq.")
    mon_ids: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for entity_id, num, mon_id in zip(poly_seq["entity_id"], poly_seq["num"], poly_seq["mon_id"], strict=True):
        # Microheterogeneity lists several residues at one position; keep the first.
        if (entity_id, num) not in seen:
            seen.add((entity_id, num))
            mon_ids.setdefault(entity_id, []).append(mon_id)
    entity_poly["pdbx_seq_one_letter_code_can"] = [
        sequence if isinstance(sequence, str) and sequence else _canonical_sequence(mon_ids.get(entity_id, []))
        for entity_id, sequence in zip(entity_ids, sequences, strict=True)
    ]
    block.set_mmcif_category("_entity_poly.", entity_poly)


def _iso_date(value: str | None) -> str | None:
    """Returns the raw mmCIF ``value`` as a YYYY-MM-DD date, or None if it is not a date."""
    if value is None or gemmi.cif.is_null(value):
        return None
    try:
        return datetime.date.fromisoformat(gemmi.cif.as_string(value)).isoformat()
    except ValueError:
        return None


def _fill_release_dates(block: gemmi.cif.Block, template_id: str) -> None:
    """Makes the release date readable by the template search and fills missing dates.

    The template search reads the first ``_pdbx_audit_revision_history.revision_date``
    as an unquoted ISO date; a missing or unreadable one is replaced by the placeholder.
    """
    revision_dates = block.find_values("_pdbx_audit_revision_history.revision_date")
    raw_date = revision_dates[0] if len(revision_dates) else None
    release_date = _iso_date(raw_date)
    if release_date is None:
        if raw_date is not None and not gemmi.cif.is_null(raw_date):
            logger.warning(
                f"The release date {raw_date} of custom template {template_id} is not a YYYY-MM-DD date; "
                f"{PLACEHOLDER_RELEASE_DATE} is used instead."
            )
        release_date = PLACEHOLDER_RELEASE_DATE
    if release_date != raw_date:
        block.set_mmcif_category(
            "_pdbx_audit_revision_history.",
            {"ordinal": ["1"], "revision_date": [release_date]},
        )
    if block.find_value("_pdbx_database_status.recvd_initial_deposition_date") is None:
        block.set_pair("_pdbx_database_status.recvd_initial_deposition_date", PLACEHOLDER_RELEASE_DATE)


def _normalize_mmcif(cif_path: Path, template_id: str) -> str:
    """Returns the mmCIF of ``cif_path`` (plain or gzip) rewritten as template ``template_id``.

    Raises:
      ValueError: if a category needed for template search is missing.
    """
    block = gemmi.cif.read(str(cif_path)).sole_block()
    category_names = set(block.get_mmcif_category_names())
    missing = [name.rstrip(".") for name in _REQUIRED_CATEGORIES if name not in category_names]
    if missing:
        raise ValueError(f"required mmCIF categories are missing: {', '.join(missing)}")
    if not block.find_values("_entity_poly.pdbx_strand_id"):
        raise ValueError("_entity_poly.pdbx_strand_id is missing")
    block.name = template_id
    block.set_pair("_entry.id", template_id)
    _fill_release_dates(block, template_id)
    _fill_canonical_sequences(block)
    return block.as_string()


def _polymer_chains(block: gemmi.cif.Block) -> list[tuple[str, str]]:
    """Returns ``(polymer type, author chain ID)`` of every polymer chain."""
    chains = []
    for row in block.find("_entity_poly.", ["type", "pdbx_strand_id"]):
        chains.extend((row.str(0), chain_id.strip()) for chain_id in row.str(1).split(",") if chain_id.strip())
    return chains


def _check_template(store: structure_stores.StructureStore, template_id: str) -> None:
    """Parses every protein chain of a normalized structure as the template search would.

    Checking here keeps a malformed file from failing every template search later.

    Raises:
      ValueError: if a chain ID cannot be parsed from a hit name, or the structure
        has no protein chain. Malformed structures raise whatever the
        template-search parsing raises.
    """
    block = gemmi.cif.read_string(store.get_mmcif_str(template_id)).sole_block()
    chains = _polymer_chains(block)
    invalid = [chain_id for _, chain_id in chains if not _CHAIN_ID_REGEX.fullmatch(chain_id)]
    if invalid:
        raise ValueError(f"chain IDs other than letters, digits and '_' are not supported: {', '.join(invalid)}")
    protein_chain_ids = [chain_id for poly_type, chain_id in chains if poly_type == PROTEIN_CHAIN]
    if not protein_chain_ids:
        raise ValueError("it has no protein chain")
    for chain_id in protein_chain_ids:
        _parse_hit_metadata_with_gemmi(store, template_id, chain_id)
        mmcifcontent(template_id, chain_id, store)


def _add_template(path: Path, template_id: str, store: structure_stores.StructureStore, work_dir: Path) -> None:
    """Writes the structure file ``path`` to ``work_dir`` as ``<template_id>.cif``.

    A PDB file is converted in ``work_dir``, and saved as ``X.cif`` next to it only
    once it has passed the checks, so a failed conversion leaves nothing behind.
    """
    if _is_pdb(path):
        converted = work_dir / "pdbtocif" / f"{template_id}.cif"
        converted.parent.mkdir(exist_ok=True)
        pdb_to_cif(str(path), str(converted), write_seqres=False)
        cif_path = converted
    else:
        cif_path = path
    target = work_dir / f"{template_id}.cif"
    try:
        target.write_text(_normalize_mmcif(cif_path, template_id))
        _check_template(store, template_id)
        if cif_path is not path:
            _save_converted_mmcif(path, cif_path)
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def prepare_custom_template_database(
    custom_template_dir: str | os.PathLike[str],
    work_dir: str | os.PathLike[str],
) -> CustomTemplateDatabase:
    """Builds a template database in ``work_dir`` from the structures in ``custom_template_dir``.

    PDB files are converted to mmCIF, which is also saved next to them (``X.pdb``
    -> ``X.cif``) unless that file exists. Files that cannot be used as templates
    are skipped with a warning.

    Args:
      custom_template_dir: Directory of PDB/mmCIF files, optionally gzip-compressed.
      work_dir: Empty directory (created if missing) to write the database into.

    Raises:
      NotADirectoryError: if ``custom_template_dir`` is not a directory.
      CustomTemplateError: if no structure can be used as a template, or if PDB
        files have to be converted but ``custom_template_dir`` is not writable.
    """
    directory = Path(custom_template_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"The --custom_template directory does not exist: {directory}")
    structure_files = find_custom_template_files(directory)
    if not structure_files:
        raise CustomTemplateError(
            f"No PDB or mmCIF files (*.pdb, *.cif, optionally gzip-compressed) were found in {directory}."
        )
    if any(_is_pdb(path) for path in structure_files) and not os.access(directory, os.W_OK):
        raise CustomTemplateError(f"{directory} is not writable, so the PDB files in it cannot be converted to mmCIF.")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    store = structure_stores.StructureStore(work)
    sources: dict[str, Path] = {}
    template_ids = _template_ids(_structure_name(path) for path in structure_files)
    for path, template_id in zip(structure_files, template_ids, strict=True):
        try:
            _add_template(path, template_id, store, work)
        except Exception as e:  # noqa: BLE001 - one malformed user file must not abort the run.
            logger.warning(f"{path} cannot be used as a template and is skipped: {type(e).__name__}: {e}")
            continue
        sources[template_id] = path
        logger.info(f"Using {path} as custom template {template_id}.")
    if not sources:
        raise CustomTemplateError(f"None of the structure files in {directory} can be used as a template.")
    seqres_path = work / SEQRES_FILENAME
    cif_to_seqres(str(work), str(seqres_path))
    logger.success(f"Prepared {len(sources)} custom template structure(s) from {directory}.")
    return CustomTemplateDatabase(pdb_database_path=work, seqres_database_path=seqres_path, sources=sources)
