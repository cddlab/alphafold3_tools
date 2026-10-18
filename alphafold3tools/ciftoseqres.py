import argparse
from pathlib import Path

import gemmi
from loguru import logger

from alphafold3tools.log import log_setup
from alphafold3tools.utils import add_version_option

_MOL_TYPE: dict[str, str] = {
    "polypeptide(L)": "protein",
    "polypeptide(D)": "protein",
    "polyribonucleotide": "na",
    "polydeoxyribonucleotide": "na",
    "polydeoxyribonucleotide/polyribonucleotide hybrid": "na",
    "other": "other",
}


def _parse_cif_for_seqres(cif_path: Path) -> list[tuple[str, str, str, str, str]]:
    """Parse a single CIF file and return records for pdb_seqres.txt.

    Returns:
        List of (pdb_id_lower, auth_asym_id, mol_type, sequence, title) tuples.
    """
    doc = gemmi.cif.read(str(cif_path))
    block = doc.sole_block()

    pdb_id = block.name.lower()

    keywords = block.find_value("_struct_keywords.text")
    if keywords and keywords not in ("?", "."):
        title = gemmi.cif.as_string(keywords)
    else:
        struct_title = block.find_value("_struct.title")
        if struct_title and struct_title not in ("?", "."):
            title = gemmi.cif.as_string(struct_title)
        else:
            title = "no_title"
    # A multi-line title would break the one-line FASTA header.
    title = " ".join(title.split()) or "no_title"

    poly_table = block.find(
        "_entity_poly.",
        ["entity_id", "type", "pdbx_seq_one_letter_code_can", "pdbx_strand_id"],
    )

    records: list[tuple[str, str, str, str, str]] = []
    for row in poly_table:
        poly_type = gemmi.cif.as_string(row[1])
        seq_raw = gemmi.cif.as_string(row[2])
        strand_ids_raw = gemmi.cif.as_string(row[3])

        sequence = seq_raw.replace("\n", "").replace(" ", "").replace("\r", "")

        mol_type = _MOL_TYPE.get(poly_type, "other")

        strand_ids = [s.strip() for s in strand_ids_raw.split(",") if s.strip()]

        for chain_id in strand_ids:
            records.append((pdb_id, chain_id, mol_type, sequence, title))

    return records


def cif_to_seqres(input_path: str, output_path: str) -> None:
    """Convert CIF file(s) to pdb_seqres.txt format.

    Args:
        input_path: Path to a CIF file or directory containing CIF files.
        output_path: Path to the output pdb_seqres.txt file.
    """
    p = Path(input_path)
    if p.is_dir():
        cif_files = sorted(p.glob("*.cif"))
        if not cif_files:
            cif_files = sorted(p.glob("**/*.cif"))
        logger.info(f"Found {len(cif_files)} CIF file(s) in {p}")
    else:
        cif_files = [p]

    lines: list[str] = []
    for cif_file in cif_files:
        logger.info(f"Processing {cif_file}")
        records = _parse_cif_for_seqres(cif_file)
        for pdb_id, chain_id, mol_type, sequence, title in records:
            length = len(sequence)
            header = f">{pdb_id}_{chain_id} mol:{mol_type} length:{length}  {title}"
            lines.append(header)
            lines.append(sequence)

    Path(output_path).write_text("\n".join(lines) + "\n")
    logger.info(f"Written to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert mmCIF file(s) to pdb_seqres.txt format.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input",
        required=True,
        help="Input CIF file or directory containing CIF files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="pdb_seqres.txt",
        help="Output file path (default: pdb_seqres.txt).",
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="log_level",
        action="store_const",
        const="DEBUG",
        default="SUCCESS",
        help="Enable debug logging.",
    )
    add_version_option(parser)
    args = parser.parse_args()

    log_setup(args.log_level)
    cif_to_seqres(args.input, args.output)


if __name__ == "__main__":
    main()
