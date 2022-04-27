from typing import List, Optional, Sequence
from qtpy.QtWidgets import QWidget, QFormLayout, QComboBox
from qtpy.QtCore import Qt
from superqt import QLabeledDoubleSlider, utils
import numpy as np
from psygnal import Signal
from pytransform3d import rotations as rot


# code from three.js ... haven't dug into the differences between 
# it and pytransform3d, but this matches the (nice) behavior of quaternion.online
def _euler2quat(x, y, z, order="XYZ"):
    c1 = np.cos(x * 0.5)
    c2 = np.cos(y * 0.5)
    c3 = np.cos(z * 0.5)
    s1 = np.sin(x * 0.5)
    s2 = np.sin(y * 0.5)
    s3 = np.sin(z * 0.5)
    order = order.upper()
    if order == "XYZ":
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "YXZ":
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3
    elif order == "ZXY":
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "ZYX":
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3
    elif order == "YZX":
        _x = s1 * c2 * c3 + c1 * s2 * s3
        _y = c1 * s2 * c3 + s1 * c2 * s3
        _z = c1 * c2 * s3 - s1 * s2 * c3
        _w = c1 * c2 * c3 - s1 * s2 * s3
    elif order == "XZY":
        _x = s1 * c2 * c3 - c1 * s2 * s3
        _y = c1 * s2 * c3 - s1 * c2 * s3
        _z = c1 * c2 * s3 + s1 * s2 * c3
        _w = c1 * c2 * c3 + s1 * s2 * s3
    return (_w, _x, _y, _z)


def _mat2euler(R, order="XYZ"):
    if order == "XYZ":
        _y = np.arcsin(np.clip(R[0, 2], -1, 1))
        if np.abs(R[0, 2]) < 0.9999999:
            _x = np.arctan2(-R[1, 2], R[2, 2])
            _z = np.arctan2(-R[0, 1], R[0, 0])
        else:
            _x = np.arctan2(R[2, 1], R[1, 1])
            _z = 0
        return (_x, _y, _z)
    if order == "ZYX":
        _x = np.arcsin(np.clip(R[2, 1], -1, 1))
        if np.abs(R[2, 1]) < 0.9999999:
            _y = np.arctan2(-R[2, 0], R[2, 2])
            _z = np.arctan2(-R[0, 1], R[1, 1])
        else:
            _y = 0
            _z = np.arctan2(R[1, 0], R[0, 0])
        return (_z, _y, _x)


class RotationModel:
    valueChanged = Signal(tuple)

    def __init__(
        self,
        *,
        axis_angle: Optional[Sequence[float]] = None,
        quaternion: Optional[Sequence[float]] = None,
        matrix: Optional[np.ndarray] = None,
    ) -> None:
        self._q = (1, 0, 0, 0)
        self._rotation_axis = (0, 1, 0)
        if axis_angle is not None:
            self.axis_angle = axis_angle
        elif quaternion is not None:
            self.quaternion = quaternion
        elif matrix is not None:
            self.matrix = matrix

    def __repr__(self) -> str:
        return f"RotationModel(quaternion={self._q})"

    def _update_rotation_axis(self):
        w, x, y, z = self._q
        sin = np.sin(np.arccos(w))
        if sin >= 0.01 or sin <= -0.01:
            r = np.array([x, y, z]) / sin
            self._rotation_axis = r / np.linalg.norm(r)

    @property
    def rotation_axis(self):
        return self._rotation_axis

    @property
    def quaternion(self):
        return np.asarray(self._q)

    @quaternion.setter
    def quaternion(self, q: Sequence[float]):
        """w,x,y,z"""
        # self._q = tuple(rot.check_quaternion(q, unit=False))
        self._q = tuple(q)
        self._update_rotation_axis()
        self.valueChanged(self.quaternion)

    @property
    def axis_angle(self):
        a = rot.axis_angle_from_quaternion(self._q)
        a[:3] = self._rotation_axis
        return a

    @axis_angle.setter
    def axis_angle(self, a):
        """Axis of rotation and rotation angle: (x, y, z, angle)"""
        self.quaternion = rot.quaternion_from_axis_angle(a)

    @property
    def matrix(self):
        return rot.matrix_from_quaternion(self._q)

    @matrix.setter
    def matrix(self, R):
        self.quaternion = rot.quaternion_from_matrix(R)

    @property
    def euler_xyz(self):
        # return rot.euler_xyz_from_matrix(self.matrix)
        return _mat2euler(self.matrix, order="XYZ")

    @euler_xyz.setter
    def euler_xyz(self, e_xyz):
        # self.matrix = rot.matrix_from_euler_xyz(e_xyz)
        self.quaternion = _euler2quat(*e_xyz, order="xyz")

    @property
    def euler_zyx(self):
        return rot.euler_zyx_from_matrix(self.matrix)

    @euler_zyx.setter
    def euler_zyx(self, e_zyx):
        self.quaternion = _euler2quat(*list(e_zyx)[::-1], order="zyx")

    @property
    def compact_axis_angle(self):
        return rot.compact_axis_angle_from_quaternion(self._q)

    @compact_axis_angle.setter
    def compact_axis_angle(self, ca):
        self.quaternion = rot.quaternion_from_compact_axis_angle(ca)

    def set_qw(self, v: float):
        assert -1 <= v <= 1
        sint = np.sin((np.arccos(v) * 2) / 2)
        self.quaternion = (v,) + tuple(np.asarray(self.rotation_axis) * sint)

    def set_qx(self, v):
        self._set_qxyz(v, 1)

    def set_qy(self, v):
        self._set_qxyz(v, 2)

    def set_qz(self, v):
        self._set_qxyz(v, 3)

    def _set_qwxyz(self, v, idx):
        if idx == 0:
            self.set_qw(v)
            return

        idx -= 1
        w = self._q[0]
        theta = np.arccos(w) * 2
        sinarcw = np.sin(theta / 2)

        if -0.01 < sinarcw < 0.01:
            return False

        rot_axis = self.rotation_axis
        sinv = max(-1, min(1, v / sinarcw))
        rest = 1 - sinv ** 2
        i0, i1 = (i for i in range(3) if i != idx)
        a, b = rot_axis[i0], rot_axis[i1]
        if a + b == 0:
            rot_axis[i0] = np.sqrt(rest / 2)
            rot_axis[i1] = np.sqrt(rest / 2)
        else:
            ratio = a ** 2 / (b ** 2 + a ** 2)
            rot_axis[i0] = np.sign(a) * np.sqrt(rest * ratio)
            rot_axis[i1] = np.sign(b) * np.sqrt((1 - ratio) * rest)
        rot_axis[idx] = sinv

        self.quaternion = (w,) + tuple(np.asarray(rot_axis) * sinarcw)
        return True


class RotationView(QWidget):
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
        ...

    def _add_gui(self):
        ...


class QuaternionView(RotationView):
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
        q = self._model.quaternion
        for v, sld in zip(q, self._sliders):
            with utils.signals_blocked(sld):
                sld.setValue(v)

    def _on_change(self, value):
        idx = self._sliders.index(self.sender())
        if not self._model._set_qwxyz(value, idx):
            self._update()


class EulerView(RotationView):
    def _add_gui(self):
        self._box = QComboBox()
        self._box.addItems(["XYZ", "ZYX"])
        self._box.setCurrentText("XYZ")
        self._box.currentTextChanged.connect(self._on_change)
        self.layout().addRow("Order", self._box)

        self._sliders: List[QLabeledDoubleSlider] = []
        for i in "xyz":
            wdg = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
            wdg.setRange(-180, 180)
            wdg.valueChanged.connect(self._on_change)
            setattr(self, f"_{i}_sld", wdg)
            self.layout().addRow(i, wdg)
            self._sliders.append(wdg)

    def _update(self) -> None:
        if getattr(self, '_is_updating', False):
            # if we're doing the changing, don't update the sliders.
            return
        xyz = self._model.euler_xyz
        for v, sld in zip(xyz, self._sliders):
            with utils.signals_blocked(sld):
                sld.setValue(int(np.rad2deg(v)))

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

if __name__ == "__main__":
    from qtpy.QtWidgets import QApplication

    app = QApplication([])

    model = RotationModel()
    wdg = QuaternionView(model)
    wdg.show()
    wdg2 = EulerView(model)
    wdg2.show()

    app.exec_()
