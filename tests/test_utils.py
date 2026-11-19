from pathlib import Path

from alphafold3tools.utils import get_seed_sample_numbers, get_seednumbers


def test_get_seed_sample_numbers_beyond_five_samples(tmp_path: Path):
    for seed in (2, 10):
        for sample in range(12):
            (tmp_path / f"seed-{seed}_sample-{sample}").mkdir()
    # entries that are not AlphaFold3 sample directories are ignored
    (tmp_path / "seed-3_sample-0_backup").mkdir()
    (tmp_path / "seed-4_sample-0").write_text("not a directory")
    assert get_seed_sample_numbers(tmp_path) == {2: list(range(12)), 10: list(range(12))}
    assert get_seednumbers(tmp_path) == [2, 10]


def test_get_seed_sample_numbers_empty(tmp_path: Path):
    assert get_seed_sample_numbers(tmp_path) == {}
    assert get_seednumbers(tmp_path) == []
