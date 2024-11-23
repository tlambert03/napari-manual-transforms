"""Tests for `TransformationModel`."""

from collections.abc import Sequence

import numpy as np
import pytest

from napari_manual_transforms._model import TransformationModel

TRANSLATION_INPUTS = [
    (1, 2, 3),
    (-1, -2, -3),
    (0, 0, 0),
    (-123.5, 321.1, 0.01),
    (1e-12, -1e-13, 1e20),
    [1, 2, 3],
    [-1, -2, -3],
    [0, 0, 0],
    [-123.5, 321.1, 0.01],
    [1e-12, -1e-13, 1e20],
]
SCALE_INPUTS = [1000.0, 1e20, 1e-6, 0.01, 10.1]


@pytest.mark.parametrize("translation", TRANSLATION_INPUTS)
@pytest.mark.parametrize("scale", SCALE_INPUTS)
def test_transformation_model_instantiation(
    scale: float,
    translation: Sequence[float],
) -> None:
    """Verify translation validation on model instantiation."""
    model = TransformationModel(translation=translation, scale=scale)

    np.testing.assert_equal(translation, model.translation)
    assert model.scale == scale


def test_transformation_model_default_transformation_values() -> None:
    """Verify translation validation on model instantiation."""
    model = TransformationModel()

    np.testing.assert_equal([0.0, 0.0, 0.0], model.translation)
    assert model.scale == 1.0


@pytest.mark.parametrize("translation", TRANSLATION_INPUTS)
def test_transformation_model_translation_update(translation: Sequence[float]) -> None:
    """Verify translation updates on transformation model."""
    model = TransformationModel()
    model.translation = translation

    np.testing.assert_equal(translation, model.translation)


@pytest.mark.parametrize("scale", SCALE_INPUTS)
def test_transformation_model_scale_update(scale: float) -> None:
    """Verify scale updates on transformation model."""
    model = TransformationModel()
    model.scale = scale

    np.testing.assert_equal(scale, model.scale)
