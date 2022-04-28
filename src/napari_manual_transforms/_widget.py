from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Optional

import numpy as np
from napari.layers import Image
from qtpy.QtWidgets import QLabel, QPushButton, QWidget
from vispy.util.keys import ALT

from napari_manual_transforms._tform_widget import RotationView
from napari_manual_transforms._util import _Quaternion, transform_array_3d

if TYPE_CHECKING:
    import napari.layers
    import napari.viewer
    from napari.utils.events import Event


class LayerFollower(QWidget):
    def __init__(self, viewer: napari.viewer.Viewer, parent=None) -> None:
        assert viewer
        self._viewer: Optional[napari.Viewer] = None
        self._active: Optional[napari.layers.Layer] = None
        super().__init__(parent)
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


class RotationWidget(LayerFollower, RotationView):
    def __init__(self, viewer: napari.viewer.Viewer = None, parent=None):
        self._help = QLabel("(hold alt while dragging canvas to edit)")
        self._help.setStyleSheet(
            "font-size: 9pt; color: #AAA; text-align: center;"
        )
        super().__init__(viewer, parent)

        self.layout().insertWidget(0, self._help)

        self._center_orig = QPushButton("center origin")
        self._center_orig.clicked.connect(self._center_origin)
        self.layout().addWidget(self._center_orig)

        # self._resample_btn = QPushButton("resample")
        # self._resample_btn.clicked.connect(self._resample)
        # self.layout().addWidget(self._resample_btn)

        # try:
        #     self._layer = viewer.layers["rotation axis"]
        # except (AttributeError, KeyError):
        #     self._layer = viewer.add_vectors(
        #         self._model.rotation_vector, name="rotation axis"
        #     )

        self._model.valueChanged.connect(self._on_model_changed)
        self._on_model_changed()

    def _connect_viewer(self, viewer: napari.Viewer):
        super()._connect_viewer(viewer)
        viewer.mouse_drag_callbacks.append(self._on_mouse_drag)

    def _disconnect_viewer(self):
        if self._viewer is not None:
            with contextlib.suppress(Exception):
                self._viewer.mouse_drag_callbacks.remove(self._on_mouse_drag)
        super()._disconnect_viewer()

    def _on_model_changed(self):
        self._update_active()

    def _center_origin(self):
        if self._active is not None:
            self._model.origin = np.asarray(self._active.data.shape) // 2

    def _update_active(self) -> None:
        # TODO... add support for 2d
        if not isinstance(self._active, Image) or self._active.data.ndim < 3:
            self.setEnabled(False)
            return
        self.setEnabled(True)
        with self._model.valueChanged.blocked():
            self._active.affine = self._model.transform

    def _on_mouse_drag(self, viewer, event):
        """update layer affine when alt-dragging."""
        if self._active is None or ALT not in event.modifiers:
            return

        q = _Quaternion(*self._model.quaternion)
        p2 = None
        wh = event.source.size
        yield

        while event.type == "mouse_move":
            p1, p2 = p2, event.pos
            if p1 is None:
                p1 = p2

            qp2 = _Quaternion.from_arcball(p1, wh)
            qp1 = _Quaternion.from_arcball(p2, wh)
            q = qp2 * qp1 * q

            self._model.quaternion = (q.w, q.x, q.y, q.z)
            yield

    def _on_layer_event(self, event):
        if event.type == "affine":
            self._update_from_layer()

    def _update_from_layer(self):
        if self._active is not None:
            self._model.matrix = self._active.affine.affine_matrix[:3, :3]

    def _connect_layer(self, layer: napari.layers.Layer):
        super()._connect_layer(layer)
        self._update_from_layer()
        self._help.show()

    def _disconnect_layer(self):
        super()._disconnect_layer()
        self._help.hide()

    def _resample(self):
        if self._active is not None:
            data = transform_array_3d(self._active.data, self._model.matrix)
            new_layer = type(self._active)(
                data, name=f"resampled {self._active.name}"
            )
            self._viewer.add_layer(new_layer)


if __name__ == "__main__":
    import napari

    v = napari.Viewer()
    v.open_sample("napari", "cells3d")
    v.dims.ndisplay = 3
    wdg = RotationWidget(v)
    v.window.add_dock_widget(wdg)
    napari.run()
