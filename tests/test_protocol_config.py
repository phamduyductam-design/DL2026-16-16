from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_is_normal_only_and_reproducible():
    config = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    protocol = config["protocol"]

    assert protocol["seed"] == 42
    assert 0.15 <= protocol["validation_ratio"] <= 0.20
    assert protocol["train_source"] == "official_train_good_only"
    assert protocol["ground_truth_usage"] == "evaluation_only"
    assert config["threshold"]["source"] == "validation_normal_only"


def test_required_categories_are_locked():
    config = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    assert config["dataset"]["categories"]["primary"] == [
        "wood",
        "metal_nut",
        "capsule",
    ]

