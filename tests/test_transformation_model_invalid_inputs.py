"""Input validation tests for `TransformationModel`."""

from collections.abc import Sequence

import pytest

from napari_manual_transforms._model import TransformationModel

INVALID_TRANSLATION_INPUTS = [(1, 2, 3, 4), (1, 2), (1,), [1, 2, 3, 4], [1, 2], [1]]
INVALID_SCALE_INPUTS = [0, 0.0, 9e-7, 1e-7, 1e-10]


@pytest.mark.parametrize("translation", INVALID_TRANSLATION_INPUTS)
def test_transformation_model_translation_invalid_input_on_init(
    translation: Sequence[float],
) -> None:
    """Verify failing translation validation on model instantiation.

    The model instance expects a 3D translation vector that must be of
    length 3. Otherwise, a `ValueError` is expected to be raised.
    """
    with pytest.raises(ValueError, match="Input must be 1D and have a length of 3"):
        TransformationModel(translation=translation)


@pytest.mark.parametrize("translation", INVALID_TRANSLATION_INPUTS)
def test_transformation_model_translation_invalid_input_on_update(
    translation: Sequence[float],
) -> None:
    """Verify failing translation validation on model update.

    The model instance expects a 3D translation vector that must be of
    length 3. Otherwise, a `ValueError` is expected to be raised.
    """
    model = TransformationModel()
    with pytest.raises(ValueError, match="Input must be 1D and have a length of 3"):
        model.translation = translation


def test_transformation_model_scale_minimum_value_set() -> None:
    """Verify minimum scale to be set.

    When using the transformation model and its generated transformation
    matrix within napari, the rotation matrix must not be scaled too,
    otherwise the application throws an error. This check validates that
    the transformation model has its minimum scaling value set for
    scaling input validations.
    """
    assert TransformationModel.MIN_SCALE == 1e-6


@pytest.mark.parametrize("scale", INVALID_SCALE_INPUTS)
def test_transformation_model_scale_invalid_input_on_init(scale: float) -> None:
    """Verify scale validation on model instantiation.

    The model instance must warn the user of a scale below a minimum
    threshold, but automtically set it to its minimum value implicitly.
    """
    with pytest.warns(
        UserWarning,
        match=f"Scale value {scale} is too low. Automatically adjusting to 1e-06.",
    ):
        model = TransformationModel(scale=scale)

    assert model.scale == TransformationModel.MIN_SCALE


@pytest.mark.parametrize("scale", INVALID_SCALE_INPUTS)
def test_transformation_model_scale_invalid_input_on_update(scale: float) -> None:
    """Verify scale validation on model update.

    The model instance must warn the user of a scale below a minimum
    threshold, but automtically set it to its minimum value implicitly.
    """
    model = TransformationModel()

    with pytest.warns(
        UserWarning,
        match=f"Scale value {scale} is too low. Automatically adjusting to 1e-06.",
    ):
        model.scale = scale

    assert model.scale == TransformationModel.MIN_SCALE
