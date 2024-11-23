from typing import List, Optional, Sequence

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible, QLabeledDoubleSlider, QLabeledSlider, utils

from ._model import TransformationModel


class _TransformationComponent(QWidget):
    def __init__(self, model: TransformationModel, parent=None):
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


class QuaternionView(_TransformationComponent):
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
    def __init__(self, model: TransformationModel, parent=None):
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

        self._sliders: List[Optional[QLabeledSlider]] = []
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


class AxisAngleView(_TransformationComponent):
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


class OriginView(_TransformationComponent):
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


class TranslationView(_TransformationComponent):
    def _add_gui(self):
        self._spin_boxes: List[QDoubleSpinBox] = []
        for l in "ZYX":
            wdg = QDoubleSpinBox()
            wdg.valueChanged.connect(self._on_change)
            wdg.setMinimum(float("-inf"))
            wdg.setMaximum(float("inf"))
            self.layout().addRow(f"{l}", wdg)
            self._spin_boxes.append(wdg)

    def _on_change(self):
        self._model.translation = [s.value() for s in self._spin_boxes]

    def _update(self) -> None:
        for v, sld in zip(self._model.translation, self._spin_boxes):
            with utils.signals_blocked(sld):
                sld.setValue(v)


class ScaleView(_TransformationComponent):
    def _add_gui(self):
        self.spin_box: QDoubleSpinBox = QDoubleSpinBox()
        self.spin_box.setSingleStep(0.1)
        self.spin_box.valueChanged.connect(self._on_change)
        self.layout().addRow("Scale", self.spin_box)

    def _on_change(self):
        self._model.scale = self.spin_box.value()

    def _update(self) -> None:
        with utils.signals_blocked(self.spin_box):
            self.spin_box.setValue(self._model.scale)


class _Collapsible(QCollapsible):
    def __init__(self, title: str, widget, parent: Optional[QWidget] = None):
        super().__init__(title, parent)
        self.addWidget(widget)
        self.layout().setContentsMargins(0, 0, 0, 0)


class TransformationView(QWidget):
    def __init__(self, model: Optional[TransformationModel] = None, parent=None):
        super().__init__(parent)
        self._model: TransformationModel = model or TransformationModel()

        self._q_view = QuaternionView(self._model)
        self._e_view = EulerView(self._model)
        self._a_view = AxisAngleView(self._model)
        self._t_view = TranslationView(self._model)
        self._s_view = ScaleView(self._model)
        self._o_view = OriginView(self._model)
        self._q_view.layout().setContentsMargins(0, 0, 0, 0)
        self._e_view.layout().setContentsMargins(0, 0, 0, 0)
        self._a_view.layout().setContentsMargins(0, 0, 0, 0)
        self._t_view.layout().setContentsMargins(0, 0, 0, 0)
        self._s_view.layout().setContentsMargins(0, 0, 0, 0)
        self._o_view.layout().setContentsMargins(0, 0, 0, 0)

        qq = _Collapsible("Quaternion", self._q_view)
        qe = _Collapsible("Euler Angle", self._e_view)
        qa = _Collapsible("Axis && Angle", self._a_view)
        qt = _Collapsible("Translation", self._t_view)
        qs = _Collapsible("Scale", self._s_view)
        qo = _Collapsible("Origin", self._o_view)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(qq)
        self.layout().addWidget(qe)
        self.layout().addWidget(qa)
        self.layout().addWidget(qt)
        self.layout().addWidget(qs)
        self.layout().addWidget(qo)

        self._reset_tform = QPushButton("reset transformation")
        self._reset_tform.clicked.connect(self._reset_transformation)
        self.layout().addWidget(self._reset_tform)

    def _reset_transformation(self):
        self._model.reset()
