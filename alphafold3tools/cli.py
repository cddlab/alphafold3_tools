"""Single-entrypoint dispatcher for the af3tools command suite.

Lazily imports only the selected subcommand's module so that heavy/
side-effectful imports (matplotlib.use("Agg") in paeplot, np.set_printoptions
in metrics, rdkit in sdftoccd, gemmi in pdbtocif/superpose_ciffiles) do not run
on every invocation.
"""

import importlib
import sys

from alphafold3tools import __version__

PROG = "af3tools"

# Subcommand name -> module path. Insertion order = help-listing order
# (mirrors README section order).
SUBCOMMANDS: dict[str, str] = {
    "msatojson": "alphafold3tools.msatojson",
    "fastatojson": "alphafold3tools.fastatojson",
    "modjson": "alphafold3tools.modjson",
    "paeplot": "alphafold3tools.paeplot",
    "superpose_ciffiles": "alphafold3tools.superpose_ciffiles",
    "sdftoccd": "alphafold3tools.sdftoccd",
    "jsontomsa": "alphafold3tools.jsontomsa",
    "pdbtocif": "alphafold3tools.pdbtocif",
    "metrics": "alphafold3tools.metrics",
}

_DESCRIPTIONS: dict[str, str] = {
    "msatojson": "Convert an a3m MSA file to AlphaFold3 input JSON.",
    "fastatojson": "Convert a FASTA file to AlphaFold3 input JSON.",
    "modjson": "Add/remove ligand entities in an AlphaFold3 input JSON.",
    "paeplot": "Plot predicted aligned error (PAE) for AF3 outputs.",
    "superpose_ciffiles": "Superpose AF3 model.cif files into a multi-model CIF.",
    "sdftoccd": "Convert an SDF file to user-provided CCD (mmCIF).",
    "jsontomsa": "Extract a3m MSA from an AlphaFold3 input JSON.",
    "pdbtocif": "Convert a PDB file to mmCIF format.",
    "metrics": "Calculate ipSAE / ipTM / pDockQ / LIS interaction metrics.",
}


def _usage() -> str:
    width = max(len(name) for name in SUBCOMMANDS)
    lines = [
        f"usage: {PROG} <subcommand> [options]",
        "",
        f"AlphaFold3 input/output toolkit (v{__version__}).",
        "",
        "subcommands:",
    ]
    for name in SUBCOMMANDS:
        lines.append(f"  {name.ljust(width)}  {_DESCRIPTIONS[name]}")
    lines += [
        "",
        f"Run '{PROG} <subcommand> -h' for subcommand-specific help.",
        f"Run '{PROG} --version' to print the version.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        print(_usage())
        return 0
    if args[0] in ("-V", "--version"):
        print(f"{PROG} {__version__}")
        return 0

    cmd, rest = args[0], args[1:]
    if cmd not in SUBCOMMANDS:
        print(f"{PROG}: error: unknown subcommand '{cmd}'", file=sys.stderr)
        print(f"Run '{PROG} --help' for the list of subcommands.", file=sys.stderr)
        return 2

    # Rewrite argv so the delegated main() sees prog == "af3tools <cmd>"
    # and parses only the remaining args. Each module's parse_args() reads
    # sys.argv implicitly, so this keeps them working unchanged and makes
    # help/usage/version display "af3tools <cmd>".
    sys.argv = [f"{PROG} {cmd}", *rest]

    module = importlib.import_module(SUBCOMMANDS[cmd])
    result = module.main()
    return result if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
