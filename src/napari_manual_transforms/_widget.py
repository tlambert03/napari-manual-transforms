from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, List, Optional

import numpy as np
from pytransform3d import rotations, transformations
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDial, QGridLayout, QLabel, QPushButton, QWidget
from superqt import QLabeledSlider
from superqt.utils import signals_blocked
from vispy.util.keys import ALT

from ._util import _Quaternion

if TYPE_CHECKING:
    import napari.layers
    import napari.viewer
    from napari.utils.events import Event


class LayerFollower(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        self._viewer: Optional[napari.Viewer] = None
        self._active: Optional[napari.layers.Layer] = None
        super().__init__()
        self._connect_viewer(viewer)

    # viewer connections

    def _connect_viewer(self, viewer: napari.Viewer):
        self._viewer = viewer
        sel_events = viewer.layers.selection.events
        sel_events.active.connect(self._on_active_layer_change)
        if active := viewer.layers.selection.active:
            self._connect_layer(active)

    def _disconnect_viewer(self):
        if self._viewer is not None:
            sel_events = self._viewer.layers.selection.events
            sel_events.changed.disconnect(self._on_active_layer_change)
        self._viewer = None

    def _on_active_layer_change(self, event: Event):
        if self._active is not None:
            self._disconnect_layer()
        if event.value is not None:
            self._connect_layer(event.value)

    # layer connections

    def _connect_layer(self, layer: napari.layers.Layer):
        self._active = layer
        layer.events.connect(self._on_layer_event)

    def _disconnect_layer(self):
        if self._active is not None:
            self._active.events.disconnect(self._on_layer_event)
        self._active = None

    def _on_layer_event(self, event):
        # layer = event.source
        # attr = event.type
        # value = getattr(layer, attr, None)
        ...

    # cleanup

    def __del__(self):
        with contextlib.suppress(Exception):
            self._disconnect_viewer()
            self._disconnect_layer()


class RotationWidget(LayerFollower):
    def __init__(self, viewer: napari.viewer.Viewer, use_dials=False) -> None:
        super().__init__(viewer)
        self.setLayout(QGridLayout())
        self._dials: List[Optional[QDial]] = [None, None, None]
        self._orig: List[Optional[QLabeledSlider]] = [None, None, None]

        _txt = QLabel("(hold alt while dragging canvas to edit)")
        _txt.setStyleSheet("font-size: 9pt; color: #AAA")
        self.layout().addWidget(_txt, 0, 0, 2, 2)
        for x, l in enumerate("ZYX"):
            self._dials[x] = self._make_dial(use_dials)
            if x == 1:
                self._dials[x].setRange(-90, 90)
            r = self.layout().rowCount()
            self.layout().addWidget(QLabel(f"↻ {l}"), r, 0)
            self.layout().addWidget(self._dials[x], r, 1)

        self._reset_rot = QPushButton("reset rotation")
        self._reset_rot.clicked.connect(self._reset_rotation)
        self.layout().addWidget(
            self._reset_rot, self.layout().rowCount(), 0, 2, 2
        )

        for i, l in enumerate("ZYX"):
            wdg = QLabeledSlider(Qt.Horizontal)
            wdg.setRange(-1000, 1000)
            if self._active is not None:
                wdg.setValue(self._active.data.shape[i] // 2)
            wdg.valueChanged.connect(self._set_euler)
            r = self.layout().rowCount()
            self.layout().addWidget(QLabel(f"{l} orig"), r, 0)
            self.layout().addWidget(wdg, r, 1)
            self._orig[i] = wdg

        self._reset_orig = QPushButton("center origin")
        self._reset_orig.clicked.connect(self._reset_origin)
        self.layout().addWidget(
            self._reset_orig, self.layout().rowCount(), 0, 2, 2
        )
        self.layout().setRowStretch(self.layout().rowCount(), 1)

    def _connect_viewer(self, viewer: napari.Viewer):
        super()._connect_viewer(viewer)
        viewer.mouse_drag_callbacks.append(self._on_mouse_drag)

    def _on_mouse_drag(self, viewer, event):
        """update layer affine when alt-dragging."""
        if self._active is None or ALT not in event.modifiers:
            return

        dq = transformations.dual_quaternion_from_transform(
            self._active.affine.affine_matrix
        )

        q = _Quaternion(*dq[:4])
        p2 = None
        wh = event.source.size
        M = np.eye(4)
        T = np.eye(4)
        T[:3, -1] = np.array([d.value() for d in self._orig])

        yield

        while event.type == "mouse_move":
            p1, p2 = p2, event.pos
            if p1 is None:
                p1 = p2

            qp2 = _Quaternion.from_arcball(p1, wh)
            qp1 = _Quaternion.from_arcball(p2, wh)
            q = qp2 * qp1 * q

            M[:3, :3] = rotations.matrix_from_quaternion((q.w, q.x, q.y, q.z))
            self._active.affine = T @ M @ np.linalg.inv(T)
            yield

    def _make_dial(self, use_dial: bool = True):
        if use_dial:
            dial = QDial()
            dial.setWrapping(True)
        else:
            dial = QLabeledSlider(Qt.Orientation.Horizontal)
        dial.setVisible(self._active is not None)
        dial.setRange(-180, 180)
        dial.valueChanged.connect(self._set_euler)
        return dial

    def _on_active_layer_change(self, event: Event):
        super()._on_active_layer_change(event)
        for d in self._dials:
            d.setVisible(event.value is not None)
        if event.value is not None:
            self._update_dials()

    def _on_layer_event(self, event):
        attr = event.type
        if attr == "affine":
            self._update_dials()

    def _update_dials(self):
        for dial, angle in zip(self._dials, self._get_euler()):
            with signals_blocked(dial):
                dial.setValue(np.rad2deg(angle))

    def _reset_rotation(self):
        self._active.affine = np.eye(4)

    def _reset_origin(self):
        if self._active is not None:
            for i, w in enumerate(self._orig):
                w.setValue(self._active.data.shape[i] // 2)

    def _get_euler(self) -> List[float]:
        aff = self._active.affine.affine_matrix
        return list(rotations.euler_xyz_from_matrix(aff[:3, :3]))

    def _set_euler(self) -> None:
        v = [np.deg2rad(d.value()) for d in self._dials]
        M = np.eye(4)
        M[:3, :3] = rotations.matrix_from_euler_xyz(v)

        T = np.eye(4)
        T[:3, -1] = np.array([d.value() for d in self._orig])
        self._active.affine = T @ M @ np.linalg.inv(T)
