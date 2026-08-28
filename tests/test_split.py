from pathlib import Path

import pytest

from data_protocol.split import assert_disjoint, deterministic_split


def test_split_is_deterministic_and_disjoint():
    paths = [Path(f"image_{index:03d}.png") for index in range(20)]
    train_a, validation_a = deterministic_split(paths, seed=42)
    train_b, validation_b = deterministic_split(paths, seed=42)
    assert train_a == train_b
    assert validation_a == validation_b
    assert len(validation_a) == 4
    assert_disjoint(train_a, validation_a)


def test_overlap_is_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint([Path("same.png")], [Path("same.png")])

