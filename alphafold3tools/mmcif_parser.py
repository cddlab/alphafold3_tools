import dataclasses
import datetime
import functools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Final, Self, TypeAlias

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

ALA = sys.intern("ALA")
ARG = sys.intern("ARG")
ASN = sys.intern("ASN")
ASP = sys.intern("ASP")
CYS = sys.intern("CYS")
GLN = sys.intern("GLN")
GLU = sys.intern("GLU")
GLY = sys.intern("GLY")
HIS = sys.intern("HIS")
ILE = sys.intern("ILE")
LEU = sys.intern("LEU")
LYS = sys.intern("LYS")
MET = sys.intern("MET")
PHE = sys.intern("PHE")
PRO = sys.intern("PRO")
SER = sys.intern("SER")
THR = sys.intern("THR")
TRP = sys.intern("TRP")
TYR = sys.intern("TYR")
VAL = sys.intern("VAL")
UNK = sys.intern("UNK")
GAP = sys.intern("-")

# Unknown ligand.
UNL = sys.intern("UNL")

# Non-standard version of MET (with Se instead of S), but often appears in PDB.
MSE = sys.intern("MSE")

PROTEIN_COMMON_ONE_TO_THREE: Mapping[str, str] = {
    "A": ALA,
    "R": ARG,
    "N": ASN,
    "D": ASP,
    "C": CYS,
    "Q": GLN,
    "E": GLU,
    "G": GLY,
    "H": HIS,
    "I": ILE,
    "L": LEU,
    "K": LYS,
    "M": MET,
    "F": PHE,
    "P": PRO,
    "S": SER,
    "T": THR,
    "W": TRP,
    "Y": TYR,
    "V": VAL,
}

PROTEIN_COMMON_THREE_TO_ONE: Mapping[str, str] = {
    v: k for k, v in PROTEIN_COMMON_ONE_TO_THREE.items()
}

canonical_sequence = "GMNGMLLSRIKKKAMELAEDLKLVDFSFGLPYTWVLVEGIEGRALGVAMTLPEEVQRYTNSIEEPSLLEFIDKADSLNIIERTLGVAAINAVSQYYIDLREAKWIDVTELIQQDEIKRIAIIGNMPPVVRTLKEKYEVYVFERNMKLWDRDTYSDTLEYHILPEVDGIIASASCIVNGTLDMILDRAKKAKLIVITGPTGQLLPEFLKGTKVTHLASMKVTNIEKALVKLKLGSFKGFESESIKYVIEV"


def write_pdbx_poly_seq_scheme(canonical_sequence: str) -> str:
    """Writes pdbx_poly_seq_scheme for the canonical sequence."""
    pdbx_lines = []
    for i, aa in enumerate(canonical_sequence, start=1):
        pdbx_lines.append(
            f"ATOM  {i:>5}  CA  {aa:>3} A{i:>4}    "
            f"{0.000:>8.3f}{0.000:>8.3f}{0.000:>8.3f}  1.00  0.00           C"
        )
    return "\n".join(pdbx_lines)
