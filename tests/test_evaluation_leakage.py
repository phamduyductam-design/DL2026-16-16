import inspect

import pytest

from evaluation_research.thresholds import (
    fit_normal_threshold,
    fit_thresholds_from_normal_validation,
)


_FORBIDDEN_INPUT_NAMES = {
    "labels",
    "masks",
    "ground_truth",
    "test_labels",
    "test_masks",
    "test_ground_truth",
}


def test_threshold_functions_do_not_expose_label_or_mask_inputs():
    normal_signature = inspect.signature(fit_normal_threshold)
    combined_signature = inspect.signature(
        fit_thresholds_from_normal_validation
    )

    normal_parameters = set(normal_signature.parameters)
    combined_parameters = set(combined_signature.parameters)

    assert _FORBIDDEN_INPUT_NAMES.isdisjoint(normal_parameters)
    assert _FORBIDDEN_INPUT_NAMES.isdisjoint(combined_parameters)


@pytest.mark.parametrize(
    "forbidden_name",
    sorted(_FORBIDDEN_INPUT_NAMES),
)
def test_threshold_fit_rejects_test_label_and_mask_keywords(
    forbidden_name,
):
    kwargs = {
        "normal_image_scores": [0.1, 0.2, 0.3],
        "normal_pixel_scores": [[[0.01, 0.02], [0.03, 0.04]]],
        forbidden_name: [0, 1],
    }

    with pytest.raises(TypeError):
        fit_thresholds_from_normal_validation(**kwargs)
