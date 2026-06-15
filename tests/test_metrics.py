"""Tests for alphafold3tools.metrics module.

Compares run_ipsae output against reference .orig files.
The last column (model path) is excluded from comparison because it depends
on how the path is specified at call time.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from alphafold3tools.metrics import (
    compute_lis_metrics,
    find_colabfold_inputs,
    find_inputs,
    run_ipsae,
)

TESTDATA = Path(__file__).parent.parent / "testfiles" / "ipsae"
AF2_DIR = TESTDATA / "af2"
AF3_DIR = TESTDATA / "af3"
CF_DIR = TESTDATA / "cf"

AF2_PAE = AF2_DIR / "af2test_scores_rank_001_alphafold2_multimer_v3_model_1_seed_000.json"
AF2_PDB = AF2_DIR / "af2test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000.pdb"
AF2_ORIG_TXT = (
    AF2_DIR / "af2test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000_15_15.txt.orig"
)
AF2_ORIG_BYRES = (
    AF2_DIR
    / "af2test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000_15_15_byres.txt.orig"
)
AF2_ORIG_PML = (
    AF2_DIR / "af2test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000_15_15.pml.orig"
)
AF2_ORIG_JSON = (
    AF2_DIR
    / "af2test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000_15_15.json.orig"
)

AF3_PAE = AF3_DIR / "tlxi_tlxi_confidences.json"
AF3_CIF = AF3_DIR / "tlxi_tlxi_model.cif"
AF3_ORIG_TXT = AF3_DIR / "tlxi_tlxi_model_10_10.txt.orig"
AF3_ORIG_BYRES = AF3_DIR / "tlxi_tlxi_model_10_10_byres.txt.orig"
AF3_ORIG_PML = AF3_DIR / "tlxi_tlxi_model_10_10.pml.orig"
AF3_ORIG_JSON = AF3_DIR / "tlxi_tlxi_model_10_10.json.orig"


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
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="module")
def af2_struct(tmp_dir):
    dst = tmp_dir / AF2_PDB.name
    shutil.copy(AF2_PDB, dst)
    return dst


@pytest.fixture(scope="module")
def af3_struct(tmp_dir):
    dst = tmp_dir / AF3_CIF.name
    shutil.copy(AF3_CIF, dst)
    return dst


@pytest.fixture(scope="module")
def af2_paths(af2_struct):
    return run_ipsae(AF2_PAE, af2_struct, pae_cutoff=15, dist_cutoff=15)


@pytest.fixture(scope="module")
def af3_paths(af3_struct):
    return run_ipsae(AF3_PAE, af3_struct, pae_cutoff=10, dist_cutoff=10)


@pytest.fixture(scope="module")
def af2_json_paths(af2_struct):
    return run_ipsae(
        AF2_PAE, af2_struct, pae_cutoff=15, dist_cutoff=15, output_json=True
    )


@pytest.fixture(scope="module")
def af3_json_paths(af3_struct):
    return run_ipsae(
        AF3_PAE, af3_struct, pae_cutoff=10, dist_cutoff=10, output_json=True
    )


@pytest.fixture(scope="module")
def af2_dir_paths(tmp_dir):
    pae, struct = find_inputs(AF2_DIR)
    subdir = tmp_dir / "af2_dir"
    subdir.mkdir(exist_ok=True)
    dst = subdir / struct.name
    shutil.copy(struct, dst)
    return run_ipsae(pae, dst, pae_cutoff=15, dist_cutoff=15)


@pytest.fixture(scope="module")
def af3_dir_paths(tmp_dir):
    pae, struct = find_inputs(AF3_DIR)
    subdir = tmp_dir / "af3_dir"
    subdir.mkdir(exist_ok=True)
    dst = subdir / struct.name
    shutil.copy(struct, dst)
    return run_ipsae(pae, dst, pae_cutoff=10, dist_cutoff=10)


@pytest.fixture(scope="module")
def af2_dir_json_paths(tmp_dir):
    pae, struct = find_inputs(AF2_DIR)
    subdir = tmp_dir / "af2_dir_json"
    subdir.mkdir(exist_ok=True)
    dst = subdir / struct.name
    shutil.copy(struct, dst)
    return run_ipsae(
        pae, dst, pae_cutoff=15, dist_cutoff=15, output_json=True, model_name=AF2_DIR.name
    )


@pytest.fixture(scope="module")
def af3_dir_json_paths(tmp_dir):
    pae, struct = find_inputs(AF3_DIR)
    subdir = tmp_dir / "af3_dir_json"
    subdir.mkdir(exist_ok=True)
    dst = subdir / struct.name
    shutil.copy(struct, dst)
    return run_ipsae(
        pae, dst, pae_cutoff=10, dist_cutoff=10, output_json=True, model_name=AF3_DIR.name
    )


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
            line
            for line in af2_paths["byres"].read_text().splitlines()
            if line.strip() and not line.startswith("i ")
        ]
        orig_lines = [
            line
            for line in AF2_ORIG_BYRES.read_text().splitlines()
            if line.strip() and not line.startswith("i ")
        ]
        assert len(result_lines) == len(orig_lines)
        for r, o in zip(result_lines, orig_lines, strict=False):
            assert r == o, f"Byres mismatch:\n  got:      {r}\n  expected: {o}"

    def test_pml_alias_lines(self, af2_paths):
        result = [
            line
            for line in af2_paths["pml"].read_text().splitlines()
            if line.startswith("alias")
        ]
        expected = [
            line
            for line in AF2_ORIG_PML.read_text().splitlines()
            if line.startswith("alias")
        ]
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
            line
            for line in af3_paths["byres"].read_text().splitlines()
            if line.strip() and not line.startswith("i ")
        ]
        orig_lines = [
            line
            for line in AF3_ORIG_BYRES.read_text().splitlines()
            if line.strip() and not line.startswith("i ")
        ]
        assert len(result_lines) == len(orig_lines)
        for r, o in zip(result_lines, orig_lines, strict=False):
            assert r == o, f"Byres mismatch:\n  got:      {r}\n  expected: {o}"

    def test_pml_alias_lines(self, af3_paths):
        result = [
            line
            for line in af3_paths["pml"].read_text().splitlines()
            if line.startswith("alias")
        ]
        expected = [
            line
            for line in AF3_ORIG_PML.read_text().splitlines()
            if line.startswith("alias")
        ]
        assert result == expected


class TestAF2Json:
    def test_json_file_exists(self, af2_json_paths):
        assert af2_json_paths["json"].exists()

    def test_json_matches_orig(self, af2_json_paths):
        result = json.loads(af2_json_paths["json"].read_text())
        expected = json.loads(AF2_ORIG_JSON.read_text())
        assert result == expected


class TestAF3Json:
    def test_json_file_exists(self, af3_json_paths):
        assert af3_json_paths["json"].exists()

    def test_json_matches_orig(self, af3_json_paths):
        result = json.loads(af3_json_paths["json"].read_text())
        expected = json.loads(AF3_ORIG_JSON.read_text())
        assert result == expected


class TestFindInputs:
    def test_af2_detects_pae(self):
        pae, _ = find_inputs(AF2_DIR)
        assert "_scores_rank_001_alphafold2_multimer_v3_model_" in pae.name
        assert pae.suffix == ".json"

    def test_af2_detects_struct(self):
        _, struct = find_inputs(AF2_DIR)
        assert struct.suffix == ".pdb"
        assert struct.exists()

    def test_af3_detects_pae(self):
        pae, _ = find_inputs(AF3_DIR)
        assert pae.name.endswith("_confidences.json")
        assert not pae.name.endswith("_summary_confidences.json")

    def test_af3_detects_struct(self):
        _, struct = find_inputs(AF3_DIR)
        assert struct.suffix == ".cif"
        assert struct.exists()

    def test_not_a_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            find_inputs(tmp_path / "nonexistent")

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(ValueError):
            find_inputs(tmp_path)


class TestAF2Dir:
    def test_output_files_exist(self, af2_dir_paths):
        for key in ("txt", "byres", "pml"):
            assert af2_dir_paths[key].exists()

    def test_txt_matches_orig(self, af2_dir_paths):
        result = _data_lines(af2_dir_paths["txt"])
        expected = _data_lines(AF2_ORIG_TXT)
        assert result == expected


class TestAF3Dir:
    def test_output_files_exist(self, af3_dir_paths):
        for key in ("txt", "byres", "pml"):
            assert af3_dir_paths[key].exists()

    def test_txt_matches_orig(self, af3_dir_paths):
        result = _data_lines(af3_dir_paths["txt"])
        expected = _data_lines(AF3_ORIG_TXT)
        assert result == expected


class TestAF2DirJson:
    def test_json_key_is_dir_name(self, af2_dir_json_paths):
        data = json.loads(af2_dir_json_paths["json"].read_text())
        assert AF2_DIR.name in data

    def test_json_key_is_not_struct_stem(self, af2_dir_json_paths):
        data = json.loads(af2_dir_json_paths["json"].read_text())
        struct_stem = find_inputs(AF2_DIR)[1].stem
        assert struct_stem not in data


class TestAF3DirJson:
    def test_json_key_is_dir_name(self, af3_dir_json_paths):
        data = json.loads(af3_dir_json_paths["json"].read_text())
        assert AF3_DIR.name in data

    def test_json_key_is_not_struct_stem(self, af3_dir_json_paths):
        data = json.loads(af3_dir_json_paths["json"].read_text())
        struct_stem = find_inputs(AF3_DIR)[1].stem
        assert struct_stem not in data


# ── ColabFold batch fixtures ───────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cf_batch_paths(tmp_dir):
    """Run batch ipSAE for all complete predictions in CF_DIR."""
    batch = find_colabfold_inputs(CF_DIR)
    results = {}
    cf_subdir = tmp_dir / "cf_batch"
    cf_subdir.mkdir(exist_ok=True)
    for pae_file, struct_file, prefix in batch:
        dst = cf_subdir / struct_file.name
        shutil.copy(struct_file, dst)
        paths = run_ipsae(
            pae_file, dst, pae_cutoff=15, dist_cutoff=15, model_name=prefix
        )
        results[prefix] = paths
    return results


@pytest.fixture(scope="module")
def cf_batch_json_paths(tmp_dir):
    """Run batch ipSAE with --json for all complete predictions in CF_DIR."""
    batch = find_colabfold_inputs(CF_DIR)
    results = {}
    cf_subdir = tmp_dir / "cf_batch_json"
    cf_subdir.mkdir(exist_ok=True)
    for pae_file, struct_file, prefix in batch:
        dst = cf_subdir / struct_file.name
        shutil.copy(struct_file, dst)
        paths = run_ipsae(
            pae_file,
            dst,
            pae_cutoff=15,
            dist_cutoff=15,
            output_json=True,
            model_name=prefix,
        )
        results[prefix] = paths
    return results


class TestFindColabfoldInputs:
    def test_finds_complete_prefixes(self):
        batch = find_colabfold_inputs(CF_DIR)
        prefixes = [p for _, _, p in batch]
        assert "cf_test_a" in prefixes
        assert "cf_test_b" in prefixes

    def test_skips_incomplete_prefix(self):
        batch = find_colabfold_inputs(CF_DIR)
        prefixes = [p for _, _, p in batch]
        assert "cf_test_incomplete" not in prefixes

    def test_returns_existing_pae_files(self):
        for pae_file, _, _ in find_colabfold_inputs(CF_DIR):
            assert pae_file.exists()
            assert pae_file.suffix == ".json"

    def test_returns_existing_struct_files(self):
        for _, struct_file, _ in find_colabfold_inputs(CF_DIR):
            assert struct_file.exists()
            assert struct_file.suffix == ".pdb"

    def test_not_a_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            find_colabfold_inputs(tmp_path / "nonexistent")

    def test_empty_directory_returns_empty(self, tmp_path):
        assert find_colabfold_inputs(tmp_path) == []


class TestBatchCF:
    def test_all_prefixes_have_output_files(self, cf_batch_paths):
        assert set(cf_batch_paths.keys()) == {"cf_test_a", "cf_test_b"}
        for paths in cf_batch_paths.values():
            for key in ("txt", "byres", "pml"):
                assert paths[key].exists()

    def test_txt_data_matches_af2_orig(self, cf_batch_paths):
        orig = _data_lines(AF2_DIR / "af2test_unrelaxed_rank_001_alphafold2_multimer_v3_model_1_seed_000_15_15.txt.orig")
        for paths in cf_batch_paths.values():
            assert _data_lines(paths["txt"]) == orig


class TestBatchCFJson:
    def test_all_prefixes_have_json(self, cf_batch_json_paths):
        assert set(cf_batch_json_paths.keys()) == {"cf_test_a", "cf_test_b"}
        for paths in cf_batch_json_paths.values():
            assert paths["json"].exists()

    def test_json_key_is_prefix(self, cf_batch_json_paths):
        for prefix, paths in cf_batch_json_paths.items():
            data = json.loads(paths["json"].read_text())
            assert prefix in data


# ── LIS-family metrics (AFM-LIS) ──────────────────────────────────────────────

# Expected values produced by AFM-LIS lis.py on the same testfiles, with the
# default pae_cutoff_lis=12 and cb_cutoff_lis=8.
AF2_LIS_EXPECTED: dict[str, dict[str, float]] = {
    "A-B": {
        "LIS": 0.3544, "cLIS": 0.7520, "LIA": 104873, "cLIA": 89,
        "iLIS": 0.5162, "iLIA": 3055.1, "iLISA": 1577.2, "actifpTM": 0.9772,
    },
    "A-C": {
        "LIS": 0.3146, "cLIS": 0.5817, "LIA": 128811, "cLIA": 202,
        "iLIS": 0.4278, "iLIA": 5101.0, "iLISA": 2182.1, "actifpTM": 0.9722,
    },
    "B-C": {
        "LIS": 0.1008, "cLIS": 0.0, "LIA": 15876, "cLIA": 0,
        "iLIS": 0.0, "iLIA": 0.0, "iLISA": 0.0, "actifpTM": 0.1924,
    },
}

AF3_LIS_EXPECTED: dict[str, dict[str, float]] = {
    "A-B": {
        "LIS": 0.5054, "cLIS": 0.7257, "LIA": 134389, "cLIA": 242,
        "iLIS": 0.6056, "iLIA": 5702.8, "iLISA": 3453.6, "actifpTM": 0.9798,
    },
}


def _lis_metrics_from_json(json_path: Path) -> dict[str, dict]:
    """Return ``{"A-B": {...}, ...}`` from the lis_metrics blocks in a JSON file."""
    data = json.loads(json_path.read_text())
    model = next(iter(data.values()))
    return {k: v["lis_metrics"] for k, v in model.items() if isinstance(v, dict)}


def _assert_lis_close(actual: dict, expected: dict, rtol: float = 1e-3) -> None:
    """Compare LIS metrics within tolerance (counts must be exact)."""
    for key, exp in expected.items():
        got = actual[key]
        assert got["LIA"] == exp["LIA"], f"LIA mismatch for {key}"
        assert got["cLIA"] == exp["cLIA"], f"cLIA mismatch for {key}"
        for metric in ("LIS", "cLIS", "iLIS", "iLIA", "iLISA", "actifpTM"):
            assert abs(got[metric] - exp[metric]) <= max(rtol * abs(exp[metric]), 0.0011), (
                f"{metric} mismatch for {key}: got {got[metric]}, expected {exp[metric]}"
            )


class TestLISMetricsAF2:
    def test_matches_afm_lis_baseline(self, af2_json_paths):
        actual = _lis_metrics_from_json(af2_json_paths["json"])
        _assert_lis_close(actual, AF2_LIS_EXPECTED)


class TestLISMetricsAF3:
    def test_matches_afm_lis_baseline(self, af3_json_paths):
        actual = _lis_metrics_from_json(af3_json_paths["json"])
        _assert_lis_close(actual, AF3_LIS_EXPECTED)


class TestLISMetricsHeaderInJson:
    def test_default_cutoffs_in_header(self, af2_json_paths):
        data = json.loads(af2_json_paths["json"].read_text())
        model = next(iter(data.values()))
        assert model["lis_pae_cutoff"] == 12
        assert model["lis_cb_cutoff"] == 8


class TestLISMetricsCustomCutoffs:
    """Verify --lis_pae_cutoff/--lis_cb_cutoff actually change the metric values."""

    def test_pae_cutoff_8_gives_different_lis(self, tmp_path, af2_struct):
        dst = tmp_path / af2_struct.name
        shutil.copy(af2_struct, dst)
        paths = run_ipsae(
            AF2_PAE,
            dst,
            pae_cutoff=15,
            dist_cutoff=15,
            output_json=True,
            lis_pae_cutoff=8.0,
        )
        data = json.loads(paths["json"].read_text())
        model = next(iter(data.values()))
        assert model["lis_pae_cutoff"] == 8
        # Tightening the PAE cutoff should never increase LIA counts
        ab = model["A-B"]["lis_metrics"]
        assert ab["LIA"] < AF2_LIS_EXPECTED["A-B"]["LIA"]

    def test_cb_cutoff_change_propagates(self, tmp_path, af2_struct):
        dst = tmp_path / af2_struct.name
        shutil.copy(af2_struct, dst)
        paths = run_ipsae(
            AF2_PAE,
            dst,
            pae_cutoff=15,
            dist_cutoff=15,
            output_json=True,
            lis_cb_cutoff=4.0,
        )
        data = json.loads(paths["json"].read_text())
        model = next(iter(data.values()))
        assert model["lis_cb_cutoff"] == 4
        ab = model["A-B"]["lis_metrics"]
        # Smaller contact cutoff should reduce cLIA
        assert ab["cLIA"] <= AF2_LIS_EXPECTED["A-B"]["cLIA"]


class TestComputeLISMetricsUnit:
    def test_internal_relations_hold(self):
        import numpy as np

        rng = np.random.default_rng(42)
        chains = np.array(["A"] * 6 + ["B"] * 6)
        unique_chains = np.array(["A", "B"])
        residues = [
            {"res": "ALA", "chainid": c, "resnum": i + 1,
             "residue": f"ALA {c} {i + 1}"}
            for i, c in enumerate(chains)
        ]
        pae = rng.uniform(2, 18, (12, 12)).astype(np.float64)
        dist = np.full((12, 12), 30.0)
        np.fill_diagonal(dist, 0.0)
        dist[0:3, 6:9] = 5.0
        dist[6:9, 0:3] = 5.0

        m = compute_lis_metrics(unique_chains, chains, residues, pae, dist)
        pair = m["A-B"]
        assert pair["iLIS"] == pytest.approx(
            (pair["LIS"] * pair["cLIS"]) ** 0.5, abs=5e-4
        )
        assert pair["iLIA"] == pytest.approx(
            (pair["LIA"] * pair["cLIA"]) ** 0.5, abs=0.05
        )
        assert pair["iLISA"] == pytest.approx(
            pair["iLIS"] * pair["iLIA"], abs=0.1
        )
        assert 0.0 <= pair["actifpTM"] <= 1.0

    def test_no_contacts_yields_zero_clis(self):
        import numpy as np

        chains = np.array(["A"] * 4 + ["B"] * 4)
        unique_chains = np.array(["A", "B"])
        residues = [
            {"res": "ALA", "chainid": c, "resnum": i + 1,
             "residue": f"ALA {c} {i + 1}"}
            for i, c in enumerate(chains)
        ]
        pae = np.full((8, 8), 5.0)
        dist = np.full((8, 8), 100.0)
        np.fill_diagonal(dist, 0.0)

        m = compute_lis_metrics(unique_chains, chains, residues, pae, dist)
        pair = m["A-B"]
        assert pair["cLIS"] == 0.0
        assert pair["cLIA"] == 0
        assert pair["iLIS"] == 0.0
        assert pair["iLIA"] == 0.0
        assert pair["iLISA"] == 0.0
