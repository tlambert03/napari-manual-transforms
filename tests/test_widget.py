import numpy as np
from pytransform3d.rotations import matrix_from_angle

from napari_manual_transforms import RotationWidget


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(qtbot, make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    layer = viewer.add_image(np.random.random((100, 100, 100)))

    # create our widget, passing in the viewer
    wdg = RotationWidget(viewer)
    assert np.allclose(layer.affine.affine_matrix, np.eye(4))
    viewer.window.add_dock_widget(wdg)

    assert wdg._active is layer
    wdg._center_origin()

    assert tuple(wdg._model.quaternion) == (1, 0, 0, 0)

    T = np.eye(4)
    T[:3, :3] = matrix_from_angle(0, np.pi)

    with qtbot.waitSignal(wdg._model.valueChanged):
        layer.affine = T

    assert np.allclose(wdg._model.matrix, T[:3, :3])
