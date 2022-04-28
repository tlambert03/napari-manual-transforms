import numpy as np
from vispy.util.quaternion import Quaternion


class _Quaternion(Quaternion):
    @classmethod
    def from_arcball(cls, xy, wh):
        """Convert x,y coordinates to w,x,y,z Quaternion parameters"""
        x, y = xy
        w, h = wh
        r = (w + h) / 2.0
        x, y = -(2.0 * x - w) / r, (2.0 * y - h) / r
        h = np.sqrt(x * x + y * y)
        args = (
            (0.0, x / h, y / h, 0.0)
            if h > 1.0
            else (0.0, x, y, np.sqrt(1.0 - h * h))
        )
        return cls(*args)

    def __iter__(self):
        yield from (self.w, self.x, self.y, self.z)


def transform_array_3d(ary: np.ndarray, matrix) -> np.ndarray:
    from scipy.ndimage import map_coordinates

    newshape = np.asarray(ary.shape) + np.array([100, 100, 100])
    coords = np.indices(newshape).reshape(3, -1) - np.array([[50, 50, 50]]).T
    matrix = np.asarray(matrix)

    if matrix.shape == (4, 4):
        R, T = matrix[:3, :3], np.expand_dims(matrix[:3, -1], 1)
    else:
        R, T = matrix, np.zeros((3, 1))

    transformed_coords = R @ (coords - T) + T

    new_values = map_coordinates(ary, transformed_coords, order=1)

    return new_values.reshape(newshape)
