from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROLE_DIRECTORIES = [
    "data_protocol",
    "baselines",
    "padim",
    "patchcore",
    "evaluation",
    "deployment",
]


def test_six_role_directories_are_present():
    assert all((ROOT / directory / "README.md").is_file() for directory in ROLE_DIRECTORIES)


def test_dataset_is_not_committed():
    for directory in (ROOT / "data", ROOT / "datasets"):
        assert not directory.exists() or not any(directory.rglob("*"))
