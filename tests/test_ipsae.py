"""Tests for alphafold3tools.ipsae module.

Compares run_ipsae output against reference .orig files.
The last column (model path) is excluded from comparison because it depends
on how the path is specified at call time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphafold3tools.ipsae import run_ipsae

TESTDATA = Path(__file__).parent.parent / "testfiles" / "ipsae"

AF2_PAE = TESTDATA / "RAF1_KSR1_MEK1_9f755_scores_af2mv3_model_1_s_0.json"
AF2_PDB = TESTDATA / "RAF1_KSR1_MEK1_9f755_unrelaxed_af2mv3_model_1_s_0.pdb"
AF2_ORIG_TXT = TESTDATA / "RAF1_KSR1_MEK1_9f755_unrelaxed_af2mv3_model_1_s_0_15_15.txt.orig"
AF2_ORIG_BYRES = TESTDATA / "RAF1_KSR1_MEK1_9f755_unrelaxed_af2mv3_model_1_s_0_15_15_byres.txt.orig"
AF2_ORIG_PML = TESTDATA / "RAF1_KSR1_MEK1_9f755_unrelaxed_af2mv3_model_1_s_0_15_15.pml.orig"

AF3_PAE = TESTDATA / "tlxi_tlxi_confidences.json"
AF3_CIF = TESTDATA / "tlxi_tlxi_model.cif"
AF3_ORIG_TXT = TESTDATA / "tlxi_tlxi_model_10_10.txt.orig"
AF3_ORIG_BYRES = TESTDATA / "tlxi_tlxi_model_10_10_byres.txt.orig"
AF3_ORIG_PML = TESTDATA / "tlxi_tlxi_model_10_10.pml.orig"

AF2_ORIG_JSON = TESTDATA / "RAF1_KSR1_MEK1_9f755_unrelaxed_af2mv3_model_1_s_0_15_15.json.orig"
AF3_ORIG_JSON = TESTDATA / "tlxi_tlxi_model_10_10.json.orig"


def _strip_model_col(line: str) -> str:
    """Remove last whitespace-delimited token (model path) from a data line."""
    parts = line.rstrip("\n").rsplit(None, 1)
    return parts[0] if len(parts) == 2 else line.rstrip("\n")


def _data_lines(path: Path) -> list[str]:
    """Return non-empty lines with model-path column stripped."""
    result = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("alias"):
            result.append(stripped)
        else:
            result.append(_strip_model_col(line))
    return result


@pytest.fixture(scope="module")
def af2_paths():
    return run_ipsae(AF2_PAE, AF2_PDB, pae_cutoff=15, dist_cutoff=15)


@pytest.fixture(scope="module")
def af3_paths():
    return run_ipsae(AF3_PAE, AF3_CIF, pae_cutoff=10, dist_cutoff=10)


@pytest.fixture(scope="module")
def af2_json_paths():
    return run_ipsae(AF2_PAE, AF2_PDB, pae_cutoff=15, dist_cutoff=15, output_json=True)


@pytest.fixture(scope="module")
def af3_json_paths():
    return run_ipsae(AF3_PAE, AF3_CIF, pae_cutoff=10, dist_cutoff=10, output_json=True)


class TestAF2:
    def test_output_files_exist(self, af2_paths):
        for key in ("txt", "byres", "pml"):
            assert af2_paths[key].exists()

    def test_txt_matches_orig(self, af2_paths):
        result = _data_lines(af2_paths["txt"])
        expected = _data_lines(AF2_ORIG_TXT)
        assert result == expected

    def test_byres_header(self, af2_paths):
        first_line = af2_paths["byres"].read_text().splitlines()[0]
        orig_first = AF2_ORIG_BYRES.read_text().splitlines()[0]
        assert first_line == orig_first

    def test_byres_numeric_values(self, af2_paths):
        result_lines = [
            l for l in af2_paths["byres"].read_text().splitlines()
            if l.strip() and not l.startswith("i ")
        ]
        orig_lines = [
            l for l in AF2_ORIG_BYRES.read_text().splitlines()
            if l.strip() and not l.startswith("i ")
        ]
        assert len(result_lines) == len(orig_lines)
        for r, o in zip(result_lines, orig_lines):
            assert r == o, f"Byres mismatch:\n  got:      {r}\n  expected: {o}"

    def test_pml_alias_lines(self, af2_paths):
        result = [l for l in af2_paths["pml"].read_text().splitlines()
                  if l.startswith("alias")]
        expected = [l for l in AF2_ORIG_PML.read_text().splitlines()
                    if l.startswith("alias")]
        assert result == expected


class TestAF3:
    def test_output_files_exist(self, af3_paths):
        for key in ("txt", "byres", "pml"):
            assert af3_paths[key].exists()

    def test_txt_matches_orig(self, af3_paths):
        result = _data_lines(af3_paths["txt"])
        expected = _data_lines(AF3_ORIG_TXT)
        assert result == expected

    def test_byres_numeric_values(self, af3_paths):
        result_lines = [
            l for l in af3_paths["byres"].read_text().splitlines()
            if l.strip() and not l.startswith("i ")
        ]
        orig_lines = [
            l for l in AF3_ORIG_BYRES.read_text().splitlines()
            if l.strip() and not l.startswith("i ")
        ]
        assert len(result_lines) == len(orig_lines)
        for r, o in zip(result_lines, orig_lines):
            assert r == o, f"Byres mismatch:\n  got:      {r}\n  expected: {o}"

    def test_pml_alias_lines(self, af3_paths):
        result = [l for l in af3_paths["pml"].read_text().splitlines()
                  if l.startswith("alias")]
        expected = [l for l in AF3_ORIG_PML.read_text().splitlines()
                    if l.startswith("alias")]
        assert result == expected


import json as _json


class TestAF2Json:
    def test_json_file_exists(self, af2_json_paths):
        assert af2_json_paths["json"].exists()

    def test_json_matches_orig(self, af2_json_paths):
        result = _json.loads(af2_json_paths["json"].read_text())
        expected = _json.loads(AF2_ORIG_JSON.read_text())
        assert result == expected


class TestAF3Json:
    def test_json_file_exists(self, af3_json_paths):
        assert af3_json_paths["json"].exists()

    def test_json_matches_orig(self, af3_json_paths):
        result = _json.loads(af3_json_paths["json"].read_text())
        expected = _json.loads(AF3_ORIG_JSON.read_text())
        assert result == expected
