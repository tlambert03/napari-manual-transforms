from typing import List, Optional, Sequence

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QLabeledDoubleSlider, QLabeledSlider, utils

from ._model import RotationModel


class _RotationComponent(QWidget):
    def __init__(self, model: RotationModel, parent=None):
        super().__init__(parent)

        lay = QFormLayout()
        lay.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )
        self.setLayout(lay)
        self._add_gui()
        self._model = model
        self._model.valueChanged.connect(self._update)
        self._update()

    def model(self):
        return self._model

    def _update(self):
        raise NotImplementedError()

    def _add_gui(self):
        raise NotImplementedError()


class QuaternionView(_RotationComponent):
    _q: Sequence[float]

    def _add_gui(self):
        self._sliders: List[QLabeledDoubleSlider] = []
        for i in "wxyz":
            wdg = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            wdg.setRange(-1, 1)
            wdg.valueChanged.connect(self._on_change)
            setattr(self, f"_{i}_sld", wdg)
            self.layout().addRow(i, wdg)
            self._sliders.append(wdg)

    def _update(self) -> None:
        for v, sld in zip(self._model.quaternion, self._sliders):
            with utils.signals_blocked(sld):
                sld.setValue(v)

    def _on_change(self, value):
        idx = self._sliders.index(self.sender())
        if not self._model._set_qwxyz(value, idx):
            self._update()


class EulerView(QWidget):
    def __init__(self, model: RotationModel, parent=None):
        super().__init__(parent)
        self._add_gui()
        self._model = model
        self._model.valueChanged.connect(self._update)
        self._update()

    def _add_gui(self):
        self.setLayout(QVBoxLayout())

        self._box = QComboBox()
        self._box.addItems(["XYZ", "ZYX"])
        self._box.currentTextChanged.connect(self._on_change)
        r1 = QHBoxLayout()
        lbl = QLabel("Order:")
        lbl.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        r1.addWidget(lbl)
        r1.addWidget(self._box)

        formlay = QFormLayout()
        formlay.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        self._sliders: List[QLabeledDoubleSlider] = []
        for i in "xyz":
            wdg = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            wdg.setRange(-180, 180)
            wdg.valueChanged.connect(self._on_change)
            setattr(self, f"_{i}_sld", wdg)
            formlay.addRow(i, wdg)
            self._sliders.append(wdg)

        self.layout().addLayout(r1)
        self.layout().addLayout(formlay)

    def _update(self) -> None:
        if getattr(self, "_is_updating", False):
            # if we're doing the changing, don't update the sliders.
            return
        if self._box.currentText().lower() == "xyz":
            xyz = self._model.euler_xyz
        else:
            xyz = self._model.euler_zyx[::-1]
        for v, sld in zip(xyz, self._sliders):
            with utils.signals_blocked(sld):
                sld.setValue(np.rad2deg(v))

    def _on_change(self):
        order = self._box.currentText().lower()
        xyz = [np.deg2rad(s.value()) for s in self._sliders]
        self._is_updating = True
        try:
            if order == "xyz":
                self._model.euler_xyz = xyz
            else:
                self._model.euler_zyx = xyz[::-1]
        finally:
            self._is_updating = False


class AxisAngleView(_RotationComponent):
    def _add_gui(self):
        self._sliders: List[QLabeledDoubleSlider] = []
        for i in "xyzθ":
            wdg = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            if i in "xyz":
                wdg.setRange(-1, 1)
            else:
                wdg.setRange(0, 180)
            wdg.valueChanged.connect(self._on_change)
            setattr(self, f"_{i}_sld", wdg)
            self.layout().addRow(i, wdg)
            self._sliders.append(wdg)

    def _update(self) -> None:
        for i, (v, sld) in enumerate(
            zip(self._model.axis_angle, self._sliders)
        ):
            with utils.signals_blocked(sld):
                sld.setValue(np.rad2deg(v) if i == 3 else v)

    def _on_change(self):
        xyzt = [s.value() for s in self._sliders]
        xyzt[-1] = np.deg2rad(xyzt[-1])
        self._model.axis_angle = xyzt


class OriginView(_RotationComponent):
    def _add_gui(self):
        self._sliders: List[Optional[QLabeledSlider]] = [None, None, None]
        for i, l in enumerate("ZYX"):
            wdg = QLabeledSlider(Qt.Orientation.Horizontal)
            wdg.setRange(-1000, 1000)  # TODO
            wdg.valueChanged.connect(self._on_change)
            self.layout().addRow(f"{l}", wdg)
            self._sliders[i] = wdg

    def _on_change(self):
        self._model.origin = [s.value() for s in self._sliders]

    def _update(self) -> None:
        for v, sld in zip(self._model.origin, self._sliders):
            with utils.signals_blocked(sld):
                sld.setValue(v)


class _Collapsible(QCollapsible):
    def __init__(self, title: str, widget, parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self.addWidget(widget)
        self.layout().setContentsMargins(0, 0, 0, 0)


class RotationView(QWidget):
    def __init__(self, model: Optional[RotationModel] = None, parent=None):
        super().__init__(parent)
        self._model: RotationModel = model or RotationModel()

        self._q_view = QuaternionView(self._model)
        self._e_view = EulerView(self._model)
        self._a_view = AxisAngleView(self._model)
        self._o_view = OriginView(self._model)
        self._q_view.layout().setContentsMargins(0, 0, 0, 0)
        self._e_view.layout().setContentsMargins(0, 0, 0, 0)
        self._a_view.layout().setContentsMargins(0, 0, 0, 0)
        self._o_view.layout().setContentsMargins(0, 0, 0, 0)

        qq = _Collapsible("Quaternion", self._q_view)
        qe = _Collapsible("Euler Angle", self._e_view)
        qa = _Collapsible("Axis && Angle", self._a_view)
        qo = _Collapsible("Origin", self._o_view)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(qq)
        self.layout().addWidget(qe)
        self.layout().addWidget(qa)
        self.layout().addWidget(qo)

        self._reset_rot = QPushButton("reset rotation")
        self._reset_rot.clicked.connect(self._reset_rotation)
        self.layout().addWidget(self._reset_rot)

    def _reset_rotation(self):
        self._model.quaternion = (1, 0, 0, 0)
