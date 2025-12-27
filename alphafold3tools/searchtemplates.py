import dataclasses
import datetime
import functools
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Final, TypeAlias

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
from alphafold3tools.structure.mmcif_parser import mmcifcontent

PROTEIN_CHAIN: Final[str] = "polypeptide(L)"
TemplateFeatures: TypeAlias = Mapping[
    str, np.ndarray | bytes | Mapping[str, np.ndarray | bytes]
]
NestedInts: TypeAlias = int | Sequence["NestedInts"]


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
        fasta_chunks.append(f">{seqname} {descriptions.get(seqname, '')}")

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
    stockholm.append(f"{'#=GC RF':<{max_name_width}s} {ref_annotation}")
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
            logger.warning(f"Failed to align {self}: {str(e)}")
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


def polymer_auth_asym_id_to_label_asym_id_with_gemmi(
    block: gemmi.cif.Block,
) -> Mapping[str, str]:
    """
    Mapping from author chain ID to internal chain ID, polymers only.

    Args:
      block: A gemmi.cif.Block object.

    Returns:
      A mapping from author chain ID to the internal (label) chain ID for the
      given polymer types in the Structure, ligands/water are ignored.

    Raises:
      ValueError: If the mapping from internal chain IDs to author chain IDs is
        not a bijection for polymer chains.
    """
    auth_asym_id_to_label_asym_id = {}
    struc = gemmi.make_structure_from_block(block)
    for chain in struc[0]:
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
) -> tuple[Any, str | None, list[int] | None]:
    """Parse hit metadata by parsing mmCIF from structure store."""
    try:
        doc = gemmi.cif.read_string(structure_store.get_mmcif_str(pdb_id))
    except structure_stores.NotFoundError:
        logger.warning(
            f"Failed to get mmCIF for {pdb_id} (author chain {auth_chain_id})."
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
    return (release_date, sequence, unresolved_res_ids)


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
        logger.info(f"mmCIF file for {pdb_id} already exists.")
        return
    else:
        full_url = f"{url}/{pdb_id}.cif"
        response = requests.get(full_url)
        if response.status_code == 200:
            with open(Path(mmcif_path), "wb") as f:
                f.write(response.content)
            logger.info("Downloaded mmCIF file for %s.", pdb_id)
        else:
            logger.error(
                f"Failed to download mmCIF file for {pdb_id}. Status code: {response.status_code}"
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
        seqres_database_path: os.PathLike[str] | str,
        max_a3m_query_sequences: int | None,
        structure_store: structure_stores.StructureStore,
        hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
        hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
        query_release_date: datetime.date | None = None,
        chain_poly_type: str = PROTEIN_CHAIN,
        savehmmsto: bool = False,
    ) -> "Templates":
        """Creates templates from a run of hmmsearch tool against a custom a3m.

        Args:
        query_sequence: The polymer sequence of the target query.
        msa_a3m: An a3m of related polymers aligned to the query sequence, this is
            used to create an HMM for the hmmsearch run.
        max_template_date: This is used to filter templates for training, ensuring
            that they do not leak ground truth information used in testing sets.
        pdb_database_path: A path to the sequence database to search for templates.
        seqres_database_path: A path to the seqres database to build the HMM from.
        max_a3m_query_sequences: The maximum number of input MSA sequences to use
            to construct the profile which is then used to search for templates.
        structure_store: Structure store to fetch template structures from.
        hmmbuild_binary_path: Path to the hmmbuild binary. If None, uses "hmmbuild" from PATH.
        hmmsearch_binary_path: Path to the hmmsearch binary. If None, uses "hmmsearch" from PATH.
        query_release_date: The release_date of the template query, this is used
            to filter templates for training, ensuring that they do not leak
            structure information from the future.
        chain_poly_type: The polymer type of the templates.
        savehmmsto: Whether to save the HMM file generated by hmmbuild.

        Returns:
        Templates object containing a list of Hits initialised from the
        structure_store metadata and a3m alignments.
        """
        hmmsearch_a3m = run_hmmsearch_with_a3m(
            hmmbuild_binary_path=hmmbuild_binary_path,
            hmmsearch_binary_path=hmmsearch_binary_path,
            seqres_database_path=seqres_database_path,
            max_a3m_query_sequences=max_a3m_query_sequences,
            a3m=msa_a3m,
            savehmmsto=savehmmsto,
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

    def get_hits_with_structures(
        self,
    ) -> Sequence[tuple[Hit, gemmi.cif.Block]]:
        """Returns hits + Structures, Structures filtered to the hit's chain."""
        results = []
        structures = {struc.name.lower(): struc for struc in self.structures}
        for hit in self.hits:
            if not hit.is_valid:
                raise InvalidTemplateError(
                    "Hits must be filtered before calling get_hits_with_structures."
                )
            block = structures[hit.pdb_id]
            results.append((hit, block))
        return results

    @property
    def structures(self) -> Iterator[gemmi.cif.Block]:
        """Yields template structures for each unique PDB ID among hits.

        If there are multiple hits in the same Structure, the Structure will be
        included only once by this method.

        Yields:
        A Structure object for each unique PDB ID among hits.

        Raises:
        HitDateError: If template's release date exceeds max cutoff date.
        """

        for hit in self.hits:
            if hit.release_date > self.release_date_cutoff:  # pylint: disable=comparison-with-callable
                raise HitDateError(
                    f"Invalid release date for hit {hit.pdb_id=}, when release date "
                    f"cutoff is {self.release_date_cutoff}."
                )

        # Get the set of pdbs to load. In particular, remove duplicate PDB IDs.
        targets_to_load = tuple({(hit.pdb_id, hit.auth_chain_id) for hit in self.hits})

        for target_name, target_chain_id in targets_to_load:
            yield mmcifcontent(
                target_name,
                target_chain_id,
                self._structure_store,
                fix_mse_residues=True,
                fix_arginines=True,
                include_water=False,
                include_bonds=False,
                include_other=True,
            )


class Template:
    """Structural template input."""

    __slots__ = ("_mmcif", "_query_to_template")

    def __init__(self, *, mmcif: str, query_to_template_map: Mapping[int, int]):
        """Initializes the template.

        Args:
          mmcif: The structural template in mmCIF format. The mmCIF should have only
            one protein chain.
          query_to_template_map: A mapping from query residue index to template
            residue index.
        """
        self._mmcif = mmcif
        # Needed to make the Template class hashable.
        self._query_to_template = tuple(query_to_template_map.items())

    @property
    def query_to_template_map(self) -> Mapping[int, int]:
        return dict(self._query_to_template)

    @property
    def mmcif(self) -> str:
        return self._mmcif

    def __hash__(self) -> int:
        return hash((self._mmcif, tuple(sorted(self._query_to_template))))

    def __eq__(self, other) -> bool:
        mmcifs_equal = self._mmcif == other._mmcif
        maps_equal = sorted(self._query_to_template) == sorted(other._query_to_template)
        return mmcifs_equal and maps_equal


class ProteinChain:
    """Protein chain input."""

    __slots__ = (
        "_id",
        "_sequence",
        "_ptms",
        "_description",
        "_paired_msa",
        "_unpaired_msa",
        "_templates",
    )

    def __init__(
        self,
        *,
        id: str,  # pylint: disable=redefined-builtin
        sequence: str,
        ptms: Sequence[tuple[str, int]],
        description: str | None = None,
        paired_msa: str | None = None,
        unpaired_msa: str | None = None,
        templates: Sequence[Template] | None = None,
    ):
        """Initializes a single protein chain input.

        Args:
          id: Unique protein chain identifier.
          sequence: The amino acid sequence of the chain.
          ptms: A list of tuples containing the post-translational modification type
            and the (1-based) residue index where the modification is applied.
          description: An optional textual description of the protein chain.
          paired_msa: Paired A3M-formatted MSA for this chain. This MSA is not
            deduplicated and will be used to compute paired features. If None, this
            field is unset and must be filled in by the data pipeline before
            featurisation. If set to an empty string, it will be treated as a custom
            MSA with no sequences.
          unpaired_msa: Unpaired A3M-formatted MSA for this chain. This will be
            deduplicated and used to compute unpaired features. If None, this field
            is unset and must be filled in by the data pipeline before
            featurisation. If set to an empty string, it will be treated as a custom
            MSA with no sequences.
          templates: A list of structural templates for this chain. If None, this
            field is unset and must be filled in by the data pipeline before
            featurisation. The list can be empty or contain up to 20 templates.
        """
        if not all(res.isalpha() for res in sequence):
            raise ValueError(f'Protein must contain only letters, got "{sequence}"')
        if any(not 0 < mod[1] <= len(sequence) for mod in ptms):
            raise ValueError(f"Invalid protein modification index: {ptms}")
        if any(mod[0].startswith("CCD_") for mod in ptms):
            raise ValueError(
                f'Protein ptms must not contain the "CCD_" prefix, got {ptms}'
            )
        # Use hashable containers for ptms and templates.
        self._id = id
        self._sequence = sequence
        self._ptms = tuple(ptms)
        self._description = description
        self._paired_msa = paired_msa
        self._unpaired_msa = unpaired_msa
        self._templates = tuple(templates) if templates is not None else None

    @property
    def id(self) -> str:
        return self._id

    @property
    def templates(self) -> Sequence[Template] | None:
        return self._templates

    def __len__(self) -> int:
        return len(self._sequence)

    def __eq__(self, other) -> bool:
        return (
            self._id == other._id
            and self._sequence == other._sequence
            and self._ptms == other._ptms
            and self._description == other._description
            and self._paired_msa == other._paired_msa
            and self._unpaired_msa == other._unpaired_msa
            and self._templates == other._templates
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._id,
                self._sequence,
                self._ptms,
                self._description,
                self._paired_msa,
                self._unpaired_msa,
                self._templates,
            )
        )

    def hash_without_id(self) -> int:
        """Returns a hash ignoring the ID - useful for deduplication."""
        return hash(
            (
                self._sequence,
                self._ptms,
                self._description,
                self._paired_msa,
                self._unpaired_msa,
                self._templates,
            )
        )


def run_hmmsearch_with_a3m(
    *,
    hmmbuild_binary_path: str | None,
    hmmsearch_binary_path: str | None,
    seqres_database_path: os.PathLike[str] | str,
    max_a3m_query_sequences: int | None,
    a3m: str,
    savehmmsto: bool = False,
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
        if hmmbuild_binary_path is None:
            raise ValueError(
                "hmmbuild command is not found. "
                "Please install HMMER and ensure "
                "hmmbuild is in your PATH."
            )
        hmmbuild_cmd = [
            hmmbuild_binary_path,
            "--informat",
            "stockholm",
            "--hand",
            "--amino",
            hmmbuild_output_path,
            os.path.join(tmpdir, "query.sto"),
        ]
        logger.info(f"Running hmmbuild with command: {' '.join(hmmbuild_cmd)}")
        subprocess.run(
            hmmbuild_cmd,
            check=True,
            capture_output=True,
        )
        # Run hmmsearch
        # hmmsearch --noali --cpu 8 --F1 0.1 --F2 0.1 --F3 0.1 -E 100 --incE 100
        # --domE 100 --incdomE 100 -A orf5_templates.sto output.hmm pdb_seqres.txt
        if hmmsearch_binary_path is None:
            raise ValueError(
                "hmmsearch command is not found. "
                "Please install HMMER and ensure "
                "hmmsearch is in your PATH."
            )
        if savehmmsto:
            currentworkingdir = os.getcwd()
            hmmsearch_sto_path = os.path.join(currentworkingdir, "hmmsearch.a3m")
        else:
            hmmsearch_sto_path = os.path.join(tmpdir, "hmmsearch.a3m")
        hmmsearch_cmd = [
            hmmsearch_binary_path,
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
            hmmsearch_sto_path,
            hmmbuild_output_path,
            seqres_database_path,
        ]
        subprocess.run(hmmsearch_cmd, check=True, capture_output=True)
        # convert to a3m
        with open(hmmsearch_sto_path, "r") as f:
            hmmsearch_a3m = convert_stockholm_to_a3m(
                stockholm=f, remove_first_row_gaps=False, linewidth=60
            )

    return hmmsearch_a3m


def _make_templates_list(templates: Sequence[Template]) -> list[dict[str, Any]]:
    """Writes templates to a JSON file."""
    templates_list = []
    for template in templates:
        templates_list.append(
            {
                "mmcif": template.mmcif,
                "queryIndices": list(template.query_to_template_map.keys()),
                "templateIndices": list(template.query_to_template_map.values()),
            }
        )
    return templates_list


def search_templates(
    msa_a3m_string: str,
    pdb_database_path: str | os.PathLike[str] | None = None,
    seqres_database_path: str | os.PathLike[str] | None = None,
    savehmmsto: bool = False,
    max_template_date: datetime.date = datetime.date(2099, 12, 31),
    hmmbuild_binary_path: str | None = shutil.which("hmmbuild"),
    hmmsearch_binary_path: str | None = shutil.which("hmmsearch"),
) -> list[dict[str, Any]]:
    """Searches for templates using hmmsearch given an a3m MSA.
    The query sequence is extracted from the first line of the a3m.
    Args:
        msa_a3m_string: An a3m string containing the MSA for the query sequence.
        pdb_database_path: Path to the pdb_seqres.txt database file.
        seqres_database_path: Path to the seqres database file.
        savehmmsto: Whether to save the HMM file generated by hmmbuild.
        max_template_date: Maximum release date for templates.
        hmmbuild_binary_path: Path to the hmmbuild binary. If None, uses "hmmbuild" from PATH.
        hmmsearch_binary_path: Path to the hmmsearch binary. If None, uses "hmmsearch" from PATH.
    Returns:
        A list of templates in dictionary format.
    Raises:
        ValueError: If required binaries or database paths are not provided.
    """
    if hmmbuild_binary_path is None or hmmsearch_binary_path is None:
        raise ValueError("hmmbuild or hmmsearch not found in PATH")
    if pdb_database_path is None:
        raise ValueError("pdb_database_path must be provided.")
    structure_store = structure_stores.StructureStore(pdb_database_path)
    if seqres_database_path is None:
        raise ValueError("seqres_database_path must be provided.")

    query_sequence = parse_fasta(msa_a3m_string)[0][0]
    template_hits = Templates.from_seq_and_a3m(
        query_sequence=query_sequence,
        msa_a3m=msa_a3m_string,
        max_template_date=max_template_date,
        seqres_database_path=seqres_database_path,
        max_a3m_query_sequences=None,
        structure_store=structure_store,
        hmmbuild_binary_path=hmmbuild_binary_path,
        hmmsearch_binary_path=hmmsearch_binary_path,
        chain_poly_type=PROTEIN_CHAIN,
        savehmmsto=savehmmsto,
    )

    options = gemmi.cif.WriteOptions()
    options.misuse_hash = True
    options.align_loops = 20
    options.prefer_pairs = True
    templates = [
        Template(
            mmcif=block.as_string(options=options),
            query_to_template_map=hit.query_to_hit_mapping,
        )
        for hit, block in template_hits.get_hits_with_structures()
    ]
    return _make_templates_list(templates)
