#!/usr/bin/env python3
# %%
import dataclasses
import datetime
import functools
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Final

import gemmi
import numpy as np
import requests
from loguru import logger

import alphafold3tools.structure_stores as structure_stores
from alphafold3tools.msa_conversion import (
    align_sequence_to_gapless_query_cpp,
    convert_a3m_to_stockholm_cpp,
    fasta_string_iterator,
    parse_fasta_include_descriptions,
)

PROTEIN_CHAIN: Final[str] = "polypeptide(L)"


class AlignmentError(Exception):
    """Failed alignment between the hit sequence and the actual mmCIF sequence."""


def realign_hit_to_structure(
    *,
    hit_sequence: str,
    hit_start_index: int,
    hit_end_index: int,
    full_length: int,
    structure_sequence: str,
    query_to_hit_mapping: Mapping[int, int],
) -> Mapping[int, int]:
    """Realigns the hit sequence to the Structure sequence.

    For example, for the given input:
      query_sequence : ABCDEFGHIJKL
      hit_sequence   : ---DEFGHIJK-
      struc_sequence : XDEFGHKL
    the mapping is {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7}. However, the
    actual Structure sequence has an extra X at the start as well as no IJ. So the
    alignment from the query to the Structure sequence will be:
      hit_sequence   : ---DEFGHIJK-
      struc_aligned  : --XDEFGH--KL
    and the new mapping will therefore be: {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 10: 6}.

    Args:
      hit_sequence: The PDB seqres hit sequence obtained from Hmmsearch, but
        without any gaps. This is not the full PDB seqres template sequence but
        rather just its subsequence from hit_start_index to hit_end_index.
      hit_start_index: The start index of the hit sequence in the full PDB seqres
        template sequence (inclusive).
      hit_end_index: The end index of the hit sequence in the full PDB seqres
        template sequence (exclusive).
      full_length: The length of the full PDB seqres template sequence.
      structure_sequence: The actual sequence extracted from the Structure
        corresponding to this template. In vast majority of cases this is the same
        as the PDB seqres sequence, but this function handles the cases when not.
      query_to_hit_mapping: The mapping from the query sequence to the
        hit_sequence.

    Raises:
      AlignmentError: if the alignment between the sequence returned by Hmmsearch
        differs from the actual sequence found in the mmCIF and can't be aligned
        using the simple alignment algorithm.

    Returns:
      A mapping from the query sequence to the actual Structure sequence.
    """
    max_num_gaps = full_length - len(structure_sequence)
    if max_num_gaps < 0:
        raise AlignmentError(
            f"The Structure sequence ({len(structure_sequence)}) "
            f"must be shorter than the PDB seqres sequence ({full_length}):\n"
            f"Structure sequence : {structure_sequence}\n"
            f"PDB seqres sequence: {hit_sequence}"
        )

    if len(hit_sequence) != hit_end_index - hit_start_index:
        raise AlignmentError(
            f"The difference of {hit_end_index=} and {hit_start_index=} does not "
            f"equal to the length of the {hit_sequence}: {len(hit_sequence)}"
        )

    best_score = -1
    best_start = 0
    best_query_to_hit_mapping = query_to_hit_mapping
    max_num_gaps_before_subseq = min(hit_start_index, max_num_gaps)
    # It is possible the gaps needed to align the PDB seqres subsequence and
    # the Structure subsequence need to be inserted before the match region.
    # Try and pick the alignment with the best number of aligned residues.
    for num_gaps_before_subseq in range(0, max_num_gaps_before_subseq + 1):
        start = hit_start_index - num_gaps_before_subseq
        end = hit_end_index - num_gaps_before_subseq
        structure_subseq = structure_sequence[start:end]

        new_query_to_hit_mapping, score = _remap_to_struc_seq(
            hit_seq=hit_sequence,
            struc_seq=structure_subseq,
            max_num_gaps=max_num_gaps - num_gaps_before_subseq,
            mapping=query_to_hit_mapping,
        )
        if score >= best_score:
            # Use >= to prefer matches with larger number of gaps before.
            best_score = score
            best_start = start
            best_query_to_hit_mapping = new_query_to_hit_mapping

    return {q: h + best_start for q, h in best_query_to_hit_mapping.items()}


def _remap_to_struc_seq(
    *,
    hit_seq: str,
    struc_seq: str,
    max_num_gaps: int,
    mapping: Mapping[int, int],
) -> tuple[Mapping[int, int], int]:
    """Remaps the query -> hit mapping to match the actual Structure sequence.

    Args:
      hit_seq: The hit sequence - a subsequence of the PDB seqres sequence without
        any Hmmsearch modifications like inserted gaps or lowercased residues.
      struc_seq: The actual sequence obtained from the corresponding Structure.
      max_num_gaps: The maximum number of gaps that can be inserted in the
        Structure sequence. In practice, this is the length difference between the
        PDB seqres sequence and the actual Structure sequence.
      mapping: The mapping from the query residues to the hit residues. This will
        be remapped to point to the actual Structure sequence using a simple
        realignment algorithm.

    Returns:
      A tuple of (mapping, score):
        * Mapping from the query to the actual Structure sequence.
        * Score which is the number of matching aligned residues.

    Raises:
      ValueError if the structure sequence isn't shorter than the seqres sequence.
      ValueError if the alignment fails.
    """
    hit_seq_idx = 0
    struc_seq_idx = 0
    hit_to_struc_seq_mapping = {}
    score = 0

    # This while loop is guaranteed to terminate since we increase both
    # struc_seq_idx and hit_seq_idx by at least 1 in each iteration.
    remaining_num_gaps = max_num_gaps
    while hit_seq_idx < len(hit_seq) and struc_seq_idx < len(struc_seq):
        if hit_seq[hit_seq_idx] != struc_seq[struc_seq_idx]:
            # Explore which alignment aligns the next residue (if present).
            best_shift = 0
            for shift in range(0, remaining_num_gaps + 1):
                next_hit_res = hit_seq[hit_seq_idx + shift : hit_seq_idx + shift + 1]
                next_struc_res = struc_seq[struc_seq_idx : struc_seq_idx + 1]
                if next_hit_res == next_struc_res:
                    best_shift = shift
                    break
            hit_seq_idx += best_shift
            remaining_num_gaps -= best_shift

        hit_to_struc_seq_mapping[hit_seq_idx] = struc_seq_idx
        score += hit_seq[hit_seq_idx] == struc_seq[struc_seq_idx]
        hit_seq_idx += 1
        struc_seq_idx += 1

    fixed_mapping = {}
    for query_idx, original_hit_idx in mapping.items():
        fixed_hit_idx = hit_to_struc_seq_mapping.get(original_hit_idx)
        if fixed_hit_idx is not None:
            fixed_mapping[query_idx] = fixed_hit_idx

    return fixed_mapping, score


def parse_fasta(fasta_string: str) -> tuple[Sequence[str], Sequence[str]]:
    """Parses FASTA string and returns list of strings with amino-acid sequences.

    Arguments:
      fasta_string: The string contents of a FASTA file.

    Returns:
      A tuple of two lists:
      * A list of sequences.
      * A list of sequence descriptions taken from the comment lines. In the
        same order as the sequences.
    """
    return parse_fasta_include_descriptions(fasta_string)


def convert_stockholm_to_a3m(
    stockholm: IO[str],
    max_sequences: int | None = None,
    remove_first_row_gaps: bool = True,
    linewidth: int | None = None,
) -> str:
    """Converts MSA in Stockholm format to the A3M format."""
    descriptions = {}
    sequences = {}
    reached_max_sequences = False

    if linewidth is not None and linewidth <= 0:
        raise ValueError("linewidth must be > 0 or None")

    for line in stockholm:
        reached_max_sequences = max_sequences and len(sequences) >= max_sequences
        line = line.strip()
        # Ignore blank lines, markup and end symbols - remainder are alignment
        # sequence parts.
        if not line or line.startswith(("#", "//")):
            continue
        seqname, aligned_seq = line.split(maxsplit=1)
        if seqname not in sequences:
            if reached_max_sequences:
                continue
            sequences[seqname] = ""
        sequences[seqname] += aligned_seq

    if not sequences:
        return ""

    stockholm.seek(0)
    for line in stockholm:
        line = line.strip()
        if line[:4] == "#=GS":
            # Description row - example format is:
            # #=GS UniRef90_Q9H5Z4/4-78            DE [subseq from] cDNA: FLJ22755 ...
            columns = line.split(maxsplit=3)
            seqname, feature = columns[1:3]
            value = columns[3] if len(columns) == 4 else ""
            if feature != "DE":
                continue
            if reached_max_sequences and seqname not in sequences:
                continue
            descriptions[seqname] = value
            if len(descriptions) == len(sequences):
                break

    assert len(descriptions) <= len(sequences)

    # Convert sto format to a3m line by line
    a3m_sequences = {}
    # query_sequence is assumed to be the first sequence
    query_sequence = next(iter(sequences.values()))
    for seqname, sto_sequence in sequences.items():
        if remove_first_row_gaps:
            a3m_sequences[seqname] = align_sequence_to_gapless_query_cpp(
                sequence=sto_sequence, query_sequence=query_sequence
            ).replace(".", "")
        else:
            a3m_sequences[seqname] = sto_sequence.replace(".", "")

    fasta_chunks = []

    for seqname, seq in a3m_sequences.items():
        fasta_chunks.append(f'>{seqname} {descriptions.get(seqname, "")}')

        if linewidth:
            fasta_chunks.extend(
                seq[i : linewidth + i] for i in range(0, len(seq), linewidth)
            )
        else:
            fasta_chunks.append(seq)

    return "\n".join(fasta_chunks) + "\n"  # Include terminating newline.


def convert_a3m_to_stockholm(a3m: str, max_seqs: int | None = None) -> str:
    """Converts MSA in the A3M format to the Stockholm format."""
    sequences, descriptions = parse_fasta(a3m)
    if max_seqs is not None:
        sequences = sequences[:max_seqs]
        descriptions = descriptions[:max_seqs]

    stockholm = ["# STOCKHOLM 1.0", ""]

    # Add the Stockholm header with the sequence metadata.
    names = []
    for i, description in enumerate(descriptions):
        name, _, rest = description.replace("\t", " ").partition(" ")
        # Ensure that the names are unique - stockholm format requires that
        # the sequence names are unique.
        name = f"{name}_{i}"
        names.append(name)
        # Avoid zero-length description due to historic hmmbuild parsing bug.
        desc = rest.strip() or "<EMPTY>"
        stockholm.append(f"#=GS {name.strip()} DE {desc}")
    stockholm.append("")

    # Convert insertions in a sequence into gaps in all other sequences that don't
    # have an insertion in that column as well.
    sequences = convert_a3m_to_stockholm_cpp(sequences)

    # Add the MSA data.
    max_name_width = max(len(name) for name in names)
    for name, sequence in zip(names, sequences, strict=True):
        # Align the names to the left and pad with spaces to the maximum length.
        stockholm.append(f"{name:<{max_name_width}s} {sequence}")

    # Add the reference annotation for the query (the first sequence).
    ref_annotation = "".join("." if c == "-" else "x" for c in sequences[0])
    stockholm.append(f'{"#=GC RF":<{max_name_width}s} {ref_annotation}')
    stockholm.append("//")

    return "\n".join(stockholm)


def _parse_hit_description(description: str) -> tuple[str, str, int, int, int]:
    """Parses the hmmsearch A3M sequence description line."""
    # Example lines (protein, nucleic, no description):
    # >4pqx_A/2-217 [subseq from] mol:protein length:217  Free text
    # >4pqx_A/2-217 [subseq from] mol:na length:217  Free text
    # >5g3r_A/1-55 [subseq from] mol:protein length:352
    if match := re.fullmatch(_HIT_DESCRIPTION_REGEX, description):
        return (
            match["pdb_id"],
            match["chain_id"],
            int(match["start"]),
            int(match["end"]),
            int(match["length"]),
        )
    else:
        raise ValueError(f'Could not parse description "{description}"')


_DAYS_BEFORE_QUERY_DATE: Final[int] = 60
_HIT_DESCRIPTION_REGEX = re.compile(
    r"(?P<pdb_id>[a-z0-9]{4,})_(?P<chain_id>\w+)/(?P<start>\d+)-(?P<end>\d+) "
    r".* length:(?P<length>\d+)\b.*"
)

_STANDARDIZED_AA = {"B": "D", "J": "X", "O": "X", "U": "C", "Z": "E"}


class Error(Exception):
    """Base class for exceptions."""


class HitDateError(Error):
    """An error indicating that invalid release date was detected."""


class InvalidTemplateError(Error):
    """An error indicating that template is invalid."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class Hit:
    """Template hit metrics derived from the MSA for filtering and featurising.

    Attributes:
      pdb_id: The PDB ID of the hit.
      auth_chain_id: The author chain ID of the hit.
      hmmsearch_sequence: Hit sequence as given in hmmsearch a3m output.
      structure_sequence: Hit sequence as given in PDB structure.
      unresolved_res_indices: Indices of unresolved residues in the structure
        sequence. 0-based.
      query_sequence: The query nucleotide/amino acid sequence.
      start_index: The start index of the sequence relative to the full PDB seqres
        sequence. Inclusive and uses 0-based indexing.
      end_index: The end index of the sequence relative to the full PDB seqres
        sequence. Exclusive and uses 0-based indexing.
      full_length: Length of the full PDB seqres sequence. This can be different
        from the length from the actual sequence we get from the mmCIF and we use
        this to detect whether we need to realign or not.
      release_date: The release date of the PDB corresponding to this hit.
      chain_poly_type: The polymer type of the selected hit structure.
    """

    pdb_id: str
    auth_chain_id: str
    hmmsearch_sequence: str
    structure_sequence: str
    unresolved_res_indices: Sequence[int] | None
    query_sequence: str
    start_index: int
    end_index: int
    full_length: int
    release_date: datetime.date
    chain_poly_type: str

    @functools.cached_property
    def query_to_hit_mapping(self) -> Mapping[int, int]:
        """0-based query index to hit index mapping."""
        query_to_hit_mapping = {}
        hit_index = 0
        query_index = 0
        for residue in self.hmmsearch_sequence:
            # Gap inserted in the template
            if residue == "-":
                query_index += 1
            # Deleted residue in the template (would be a gap in the query).
            elif residue.islower():
                hit_index += 1
            # Normal aligned residue, in both query and template. Add to mapping.
            elif residue.isupper():
                query_to_hit_mapping[query_index] = hit_index
                query_index += 1
                hit_index += 1

        structure_subseq = self.structure_sequence[self.start_index : self.end_index]
        if self.matching_sequence != structure_subseq:
            # The seqres sequence doesn't match the structure sequence. Two cases:
            # 1. The sequences have the same length. The sequences are different
            #    because our 3->1 residue code mapping is different from the one PDB
            #    uses. We don't do anything in this case as both sequences have the
            #    same length, so the original query to hit mapping stays valid.
            # 2. The sequences don't have the same length, the one in structure is
            #    shorter. In this case we change the mapping to match the actual
            #    structure sequence using a simple realignment algorithm.
            # This procedure was validated on all PDB seqres (2023_01_12) sequences
            # and handles all cases that can happen.
            if self.full_length != len(self.structure_sequence):
                return realign_hit_to_structure(
                    hit_sequence=self.matching_sequence,
                    hit_start_index=self.start_index,
                    hit_end_index=self.end_index,
                    full_length=self.full_length,
                    structure_sequence=self.structure_sequence,
                    query_to_hit_mapping=query_to_hit_mapping,
                )

        # Hmmsearch returns a subsequence and so far indices have been relative to
        # the subsequence. Add an offset to index relative to the full structure
        # sequence.
        return {q: h + self.start_index for q, h in query_to_hit_mapping.items()}

    @property
    def matching_sequence(self) -> str:
        """Returns the matching hit sequence including insertions.

        Make deleted residues uppercase and remove gaps ("-").
        """
        return self.hmmsearch_sequence.upper().replace("-", "")

    @functools.cached_property
    def output_templates_sequence(self) -> str:
        """Returns the final template sequence."""
        result_seq = ["-"] * len(self.query_sequence)
        for query_index, template_index in self.query_to_hit_mapping.items():
            result_seq[query_index] = self.structure_sequence[template_index]
        return "".join(result_seq)

    @property
    def length_ratio(self) -> float:
        """Ratio of the length of the hit sequence to the query."""
        return len(self.matching_sequence) / len(self.query_sequence)

    @property
    def align_ratio(self) -> float:
        """Ratio of the number of aligned residues to the query length."""
        return len(self.query_to_hit_mapping) / len(self.query_sequence)

    @functools.cached_property
    def is_valid(self) -> bool:
        """Whether hit can be used as a template."""
        if self.unresolved_res_indices is None:
            return False

        return bool(
            set(self.query_to_hit_mapping.values()) - set(self.unresolved_res_indices)
        )

    @property
    def full_name(self) -> str:
        """A full name of the hit."""
        return f"{self.pdb_id}_{self.auth_chain_id}"

    def __post_init__(self):
        if not self.pdb_id.islower() and not self.pdb_id.isdigit():
            raise ValueError(f"pdb_id must be lowercase {self.pdb_id}")

        if not (0 <= self.start_index <= self.end_index):
            raise ValueError(
                "Start must be non-negative and less than or equal to end index. "
                f"Range: {self.start_index}-{self.end_index}"
            )

        if len(self.matching_sequence) != (self.end_index - self.start_index):
            raise ValueError(
                "Sequence length must be equal to end_index - start_index. "
                f"{len(self.matching_sequence)} != {self.end_index} - "
                f"{self.start_index}"
            )

        if self.full_length < 0:
            raise ValueError(f"Full length must be non-negative: {self.full_length}")

    def keep(
        self,
        *,
        release_date_cutoff: datetime.date | None,
        max_subsequence_ratio: float | None,
        min_hit_length: int | None,
        min_align_ratio: float | None,
    ) -> bool:
        """Returns whether the hit should be kept.

        In addition to filtering on all of the provided parameters, this method also
        excludes hits with unresolved residues.

        Args:
          release_date_cutoff: Maximum release date of the template.
          max_subsequence_ratio: If set, excludes hits which are an exact
            subsequence of the query sequence, and longer than this ratio. Useful to
            avoid ground truth leakage.
          min_hit_length: If set, excludes hits which have fewer residues than this.
          min_align_ratio: If set, excludes hits where the number of residues
            aligned to the query is less than this proportion of the template
            length.
        """
        # Exclude hits which are too recent.
        if release_date_cutoff is not None and self.release_date > release_date_cutoff:
            return False

        # Exclude hits which are large duplicates of the query_sequence.
        if (
            max_subsequence_ratio is not None
            and self.length_ratio > max_subsequence_ratio
        ):
            if self.matching_sequence in self.query_sequence:
                return False

        # Exclude hits which are too short.
        if min_hit_length is not None and len(self.matching_sequence) < min_hit_length:
            return False

        # Exclude hits with unresolved residues.
        if not self.is_valid:
            return False

        # Exclude hits with too few alignments.
        try:
            if min_align_ratio is not None and self.align_ratio <= min_align_ratio:
                return False
        except AlignmentError as e:
            logger.warning("Failed to align %s: %s", self, str(e))
            return False

        return True


def _filter_hits(
    hits: Iterable[Hit],
    release_date_cutoff: datetime.date,
    max_subsequence_ratio: float | None,
    min_align_ratio: float | None,
    min_hit_length: int | None,
    deduplicate_sequences: bool,
    max_hits: int | None,
) -> Sequence[Hit]:
    """Filters hits based on the filter config."""
    filtered_hits = []
    seen_before = set()
    for hit in hits:
        if not hit.keep(
            max_subsequence_ratio=max_subsequence_ratio,
            min_align_ratio=min_align_ratio,
            min_hit_length=min_hit_length,
            release_date_cutoff=release_date_cutoff,
        ):
            continue

        # Remove duplicate templates, keeping the first.
        if deduplicate_sequences:
            if hit.output_templates_sequence in seen_before:
                continue
            seen_before.add(hit.output_templates_sequence)

        filtered_hits.append(hit)
        if max_hits and len(filtered_hits) == max_hits:
            break

    return filtered_hits


def _polymer_auth_asym_id_to_label_asym_id_with_gemmi(
    model: gemmi.Model,
) -> Mapping[str, str]:
    """
    Docstring for polymer_auth_asym_id_to_label_asym_id_with_gemmi

    :param model: Description
    :type model: gemmi.Model
    :return: Description
    :rtype: dict[str, str]
    """
    auth_asym_id_to_label_asym_id = {}
    for chain in model:
        polymer = chain.get_polymer()
        if polymer is None:
            continue
        auth_asym_id = chain.name
        label_asym_id = polymer.subchain_id()

        if auth_asym_id in auth_asym_id_to_label_asym_id:
            raise ValueError(
                f'Author chain ID "{auth_asym_id}" does not have a unique mapping '
                f'to internal chain ID "{label_asym_id}", it is already mapped to '
                f'"{auth_asym_id_to_label_asym_id[auth_asym_id]}".'
            )
        auth_asym_id_to_label_asym_id[auth_asym_id] = label_asym_id
    return auth_asym_id_to_label_asym_id


def _parse_hit_metadata_with_gemmi(
    structure_store: structure_stores.StructureStore,
    pdb_id: str,
    auth_chain_id: str,
) -> tuple[Any, str | None, Sequence[int] | None]:
    """Parse hit metadata by parsing mmCIF from structure store."""
    try:
        doc = gemmi.cif.read_string(structure_store.get_mmcif_str(pdb_id))
    except structure_stores.NotFoundError:
        logger.warning(
            "Failed to get mmCIF for %s (author chain %s).", pdb_id, auth_chain_id
        )
        return None, None, None
    # get release date from the 1st "_pdbx_audit_revision_history.revision_date"
    block = doc.sole_block()
    model = gemmi.read_structure_string(structure_store.get_mmcif_str(pdb_id))[0]
    release_date = block.find_values("_pdbx_audit_revision_history.revision_date")[0]
    # get sequence
    entity_polys = block.find(
        "_entity_poly.", ["type", "pdbx_seq_one_letter_code_can", "pdbx_strand_id"]
    )
    for entity_poly in entity_polys:
        if entity_poly["type"] != "'polypeptide(L)'":
            raise ValueError(f"Unexpected polymer type: {entity_poly['type']}")
        if auth_chain_id in entity_poly["pdbx_strand_id"].split(","):
            sequence = (
                entity_poly["pdbx_seq_one_letter_code_can"]
                .replace("'", "")
                .replace(";", "")
                .replace("\n", "")
            )
    # missing residues
    label_asym_residues = model[auth_chain_id].get_polymer()
    all_res_ids = np.arange(1, len(sequence) + 1)
    resolved_res_ids = np.array([res.label_seq for res in label_asym_residues])
    unresolved_res_ids = (
        np.isin(all_res_ids, resolved_res_ids, invert=True).nonzero()[0] + 1
    ).tolist()
    return release_date, sequence, unresolved_res_ids


def _download_mmcif_file_for_pdbid(
    pdb_id: str,
    mmcif_dir: str,
    url: str = "https://files.rcsb.org/download",
) -> None:
    """Download mmCIF file from RCSB PDB if not already present.
    Args:
        pdbid: PDB ID of the structure to download.
        mmcif_dir: Directory to save the mmCIF file.
        url: Base URL for downloading mmCIF files.
        use_tempdir: Whether to use a temporary directory for downloading.
    """

    pdb_id = pdb_id.lower()
    mmcif_path = os.path.join(mmcif_dir, f"{pdb_id}.cif")
    if os.path.exists(mmcif_path):
        logger.info("mmCIF file for %s already exists.", pdb_id)
        return
    else:
        response = requests.get(url)
        if response.status_code == 200:
            with open(Path(mmcif_path), "wb") as f:
                f.write(response.content)
            logger.info("Downloaded mmCIF file for %s.", pdb_id)
        else:
            logger.error(
                "Failed to download mmCIF file for %s. Status code: %d",
                pdb_id,
                response.status_code,
            )


@dataclasses.dataclass(init=False)
class Templates:
    """A container for templates that were found for the given query sequence.

    The structure_store is constructed from the config by default. Callers can
    optionally supply a structure_store to the constructor to avoid the cost of
    construction and metadata loading.
    """

    def __init__(
        self,
        *,
        query_sequence: str,
        hits: Sequence[Hit],
        max_template_date: datetime.date,
        structure_store: structure_stores.StructureStore,
        query_release_date: datetime.date | None = None,
    ):
        self._query_sequence = query_sequence
        self._hits = tuple(hits)
        self._max_template_date = max_template_date
        self._query_release_date = query_release_date
        self._hit_structures = {}
        self._structure_store = structure_store

        if any(h.query_sequence != self._query_sequence for h in self.hits):
            raise ValueError("All hits must match the query sequence.")

        if self._hits:
            chain_poly_type = self._hits[0].chain_poly_type
            if any(h.chain_poly_type != chain_poly_type for h in self.hits):
                raise ValueError("All hits must have the same chain_poly_type.")

    @classmethod
    def from_seq_and_a3m(
        cls,
        *,
        query_sequence: str,
        msa_a3m: str,
        max_template_date: datetime.date,
        database_path: os.PathLike[str] | str,
        max_a3m_query_sequences: int | None,
        structure_store: structure_stores.StructureStore,
        query_release_date: datetime.date | None = None,
        chain_poly_type: str = PROTEIN_CHAIN,
    ) -> "Templates":
        """Creates templates from a run of hmmsearch tool against a custom a3m.

        Args:
        query_sequence: The polymer sequence of the target query.
        msa_a3m: An a3m of related polymers aligned to the query sequence, this is
            used to create an HMM for the hmmsearch run.
        max_template_date: This is used to filter templates for training, ensuring
            that they do not leak ground truth information used in testing sets.
        database_path: A path to the sequence database to search for templates.
        hmmsearch_config: Config with Hmmsearch settings.
        max_a3m_query_sequences: The maximum number of input MSA sequences to use
            to construct the profile which is then used to search for templates.
        structure_store: Structure store to fetch template structures from.
        filter_config: Optional config that controls which and how many hits to
            keep. More performant than constructing and then filtering. If not
            provided, no filtering is done.
        query_release_date: The release_date of the template query, this is used
            to filter templates for training, ensuring that they do not leak
            structure information from the future.
        chain_poly_type: The polymer type of the templates.

        Returns:
        Templates object containing a list of Hits initialised from the
        structure_store metadata and a3m alignments.
        """
        hmmsearch_a3m = run_hmmsearch_with_a3m(
            hmmbuild_path=shutil.which("hmmbuild"),
            hmmsearch_path=shutil.which("hmmsearch"),
            database_path=database_path,
            max_a3m_query_sequences=max_a3m_query_sequences,
            a3m=msa_a3m,
        )
        return cls.from_hmmsearch_a3m(
            query_sequence=query_sequence,
            a3m=hmmsearch_a3m,
            max_template_date=max_template_date,
            query_release_date=query_release_date,
            chain_poly_type=chain_poly_type,
            structure_store=structure_store,
        )

    @classmethod
    def from_hmmsearch_a3m(
        cls,
        *,
        query_sequence: str,
        a3m: str,
        max_template_date: datetime.date,
        structure_store: structure_stores.StructureStore,
        query_release_date: datetime.date | None = None,
        chain_poly_type: str = PROTEIN_CHAIN,
    ) -> "Templates":
        """Creates Templates from a Hmmsearch A3M.

        Args:
        query_sequence: The polymer sequence of the target query.
        a3m: Results of Hmmsearch in A3M format. This provides a list of potential
            template alignments and pdb codes.
        max_template_date: This is used to filter templates for training, ensuring
            that they do not leak ground truth information used in testing sets.
        structure_store: Structure store to fetch template structures from.
        filter_config: Optional config that controls which and how many hits to
            keep. More performant than constructing and then filtering. If not
            provided, no filtering is done.
        query_release_date: The release_date of the template query, this is used
            to filter templates for training, ensuring that they do not leak
            structure information from the future.
        chain_poly_type: The polymer type of the templates.

        Returns:
        Templates object containing a list of Hits initialised from the
        structure_store metadata and a3m alignments.
        """

        def hit_generator(a3m: str):
            if not a3m:
                return  # Hmmsearch could return an empty string if there are no hits.

            for hit_seq, hit_desc in fasta_string_iterator(a3m):
                pdb_id, auth_chain_id, start, end, full_length = _parse_hit_description(
                    hit_desc
                )

                release_date, sequence, unresolved_res_ids = (
                    _parse_hit_metadata_with_gemmi(
                        structure_store, pdb_id, auth_chain_id
                    )
                )
                if unresolved_res_ids is None:
                    continue
                if sequence is None:
                    ValueError(
                        f"Failed to get sequence for {pdb_id} (author chain "
                        f"{auth_chain_id})."
                    )
                else:
                    # seq_unresolved_res_num are 1-based, setting to 0-based indices.
                    unresolved_indices = [i - 1 for i in unresolved_res_ids]

                    yield Hit(
                        pdb_id=pdb_id,
                        auth_chain_id=auth_chain_id,
                        hmmsearch_sequence=hit_seq,
                        structure_sequence=sequence,
                        query_sequence=query_sequence,
                        unresolved_res_indices=unresolved_indices,
                        start_index=start - 1,  # Raw value is resid, not index.
                        end_index=end,
                        full_length=full_length,
                        release_date=datetime.date.fromisoformat(release_date),
                        chain_poly_type=chain_poly_type,
                    )

        hits = _filter_hits(
            hit_generator(a3m),
            release_date_cutoff=max_template_date,
            max_subsequence_ratio=0.95,
            min_align_ratio=0.1,
            min_hit_length=10,
            deduplicate_sequences=True,
            max_hits=4,
        )

        return Templates(
            query_sequence=query_sequence,
            query_release_date=query_release_date,
            hits=hits,
            max_template_date=max_template_date,
            structure_store=structure_store,
        )

    @property
    def query_sequence(self) -> str:
        return self._query_sequence

    @property
    def hits(self) -> tuple[Hit, ...]:
        return self._hits

    @property
    def query_release_date(self) -> datetime.date | None:
        return self._query_release_date

    @property
    def num_hits(self) -> int:
        return len(self._hits)

    @functools.cached_property
    def release_date_cutoff(self) -> datetime.date:
        if self.query_release_date is None:
            return self._max_template_date
        return min(
            self._max_template_date,
            self.query_release_date - datetime.timedelta(days=_DAYS_BEFORE_QUERY_DATE),
        )

    def __repr__(self) -> str:
        return f"Templates({self.num_hits} hits)"

    def filter(
        self,
        *,
        max_subsequence_ratio: float | None,
        min_align_ratio: float | None,
        min_hit_length: int | None,
        deduplicate_sequences: bool,
        max_hits: int | None,
    ) -> "Templates":
        """Returns a new Templates object with only the hits that pass all filters.

        This also filters on query_release_date and max_template_date.

        Args:
          max_subsequence_ratio: If set, excludes hits which are an exact
            subsequence of the query sequence, and longer than this ratio. Useful to
            avoid ground truth leakage.
          min_align_ratio: If set, excludes hits where the number of residues
            aligned to the query is less than this proportion of the template
            length.
          min_hit_length: If set, excludes hits which have fewer residues than this.
          deduplicate_sequences: Whether to exclude duplicate template sequences,
            keeping only the first. This can be useful in increasing the diversity
            of hits especially in the case of homomer hits.
          max_hits: If set, excludes any hits which exceed this count.
        """
        filtered_hits = _filter_hits(
            hits=self._hits,
            release_date_cutoff=self.release_date_cutoff,
            max_subsequence_ratio=max_subsequence_ratio,
            min_align_ratio=min_align_ratio,
            min_hit_length=min_hit_length,
            deduplicate_sequences=deduplicate_sequences,
            max_hits=max_hits,
        )
        return Templates(
            query_sequence=self.query_sequence,
            query_release_date=self.query_release_date,
            hits=filtered_hits,
            max_template_date=self._max_template_date,
            structure_store=self._structure_store,
        )


def run_hmmsearch_with_a3m(
    *,
    hmmbuild_path: str | None,
    hmmsearch_path: str | None,
    database_path: os.PathLike[str] | str,
    max_a3m_query_sequences: int | None,
    a3m: str,
    ncpus: int = 8,
) -> str:
    """Runs Hmmsearch to get a3m string of hits."""
    # STO enables us to annotate query non-gap columns as reference columns.
    sto = convert_a3m_to_stockholm(a3m, max_a3m_query_sequences)
    # Run hmmbuild
    with tempfile.TemporaryDirectory() as tmpdir:
        hmmbuild_output_path = os.path.join(tmpdir, "query.hmm")
        with open(os.path.join(tmpdir, "query.sto"), "w") as sto_file:
            sto_file.write(sto)
        # hmmbuild --informat stockholm --hand --amino out.hmm {sto}
        if hmmbuild_path is None:
            raise ValueError(
                "hmmbuild command is not found. "
                "Please install HMMER and ensure "
                "hmmbuild is in your PATH."
            )
        subprocess.run(
            [
                hmmbuild_path,
                "--informat",
                "stockholm",
                "--hand",
                "--amino",
                hmmbuild_output_path,
                os.path.join(tmpdir, "query.sto"),
            ],
            check=True,
            capture_output=True,
        )
        # Run hmmsearch
        # hmmsearch --noali --cpu 8 --F1 0.1 --F2 0.1 --F3 0.1 -E 100 --incE 100 --domE 100 --incdomE 100 -A orf5_templates.sto output.hmm pdb_seqres.txt
        if hmmsearch_path is None:
            raise ValueError(
                "hmmsearch command is not found. "
                "Please install HMMER and ensure "
                "hmmsearch is in your PATH."
            )
        hmmsearch_a3m_path = os.path.join(tmpdir, "hmmsearch.a3m")
        hmmsearch_cmd = [
            hmmsearch_path,
            "--noali",
            "--cpu",
            str(ncpus),
            "--F1",
            "0.1",
            "--F2",
            "0.1",
            "--F3",
            "0.1",
            "-E",
            "100",
            "--incE",
            "100",
            "--domE",
            "100",
            "--incdomE",
            "100",
            "-A",
            hmmsearch_a3m_path,
            hmmbuild_output_path,
            database_path,
        ]
        subprocess.run(hmmsearch_cmd, check=True, capture_output=True)
        # convert to a3m
        with open(hmmsearch_a3m_path, "r") as f:
            hmmsearch_a3m = convert_stockholm_to_a3m(
                stockholm=f, remove_first_row_gaps=False, linewidth=60
            )

    return hmmsearch_a3m


# %%
a3mfile = "/Users/YoshitakaM/Desktop/orf5.a3m"
outstofile = "/Users/YoshitakaM/Desktop/orf5.sto"
mmcif_dir = "/Users/YoshitakaM/Desktop/mmcif_files"


# query sequence is the first sequence in the a3m file
with open(a3mfile, "r") as f:
    query_sequence = parse_fasta(f.read())[0][0]

hmmbuild_path = shutil.which("hmmbuild")
hmmsearch_path = shutil.which("hmmsearch")
if hmmbuild_path is None or hmmsearch_path is None:
    raise ValueError("hmmbuild or hmmsearch not found in PATH")

protein_templates = Templates.from_seq_and_a3m(
    query_sequence=query_sequence,
    msa_a3m=open(a3mfile, "r").read(),
    max_template_date=datetime.date(2099, 12, 31),
    database_path="/Users/YoshitakaM/Desktop/pdb_seqres.txt",
    max_a3m_query_sequences=None,
    structure_store=structure_stores.StructureStore(mmcif_dir),
    chain_poly_type=PROTEIN_CHAIN,
)
print(protein_templates)
# %%
for hit in protein_templates.hits:
    print(f"Hit: {hit.full_name}")
    print(f" Release date: {hit.release_date}")
    print(f" Length ratio: {hit.length_ratio:.3f}")
    print(f" Align ratio: {hit.align_ratio:.3f}")
    print(f" Is valid: {hit.is_valid}")
    print(f" HMMsearch sequence: {hit.hmmsearch_sequence}")
    print(f" Structure sequence: {hit.structure_sequence}")
    print(f" Output template sequence: {hit.output_templates_sequence}")
    print(f" Query to hit mapping: {hit.query_to_hit_mapping}")

# %%
