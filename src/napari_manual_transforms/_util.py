from typing import Tuple

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


def _get_tform_bounds(input: np.ndarray, matrix: np.ndarray, full=True):
    if not full:
        return input.shape


def transform_array_3d(
    ary: np.ndarray, matrix, offset, reshape=True
) -> Tuple[np.ndarray, np.ndarray]:
    import scipy.ndimage as ndi

    if reshape:
        # Compute transformed input bounds
        iz, iy, ix = ary.shape
        out_bounds = matrix @ [
            [0, 0, 0, 0, iz, iz, iz, iz],
            [0, 0, iy, iy, 0, 0, iy, iy],
            [0, ix, 0, ix, 0, ix, 0, ix],
        ]
        # Compute the shape of the transformed input
        out_shape = (out_bounds.ptp(axis=1) + 0.5).astype(int)

        # make new larger input array.
        # FIXME: there has to be a more efficient way using output_shape and
        # offset, but so far I've failed...
        growth = np.array(out_shape) - np.array(ary.shape)
        half_g = np.abs(growth) // 2
        ary = np.pad(ary, tuple((int(i),) for i in half_g))

    else:
        half_g = np.zeros((3,))

    M = np.eye(4)
    M[:3, :3] = matrix
    T = np.eye(4)
    T[:3, -1] = np.array(offset) + half_g
    matrix = T @ M @ np.linalg.inv(T)

    data = ndi.affine_transform(
        ary, np.linalg.inv(matrix), order=1, prefilter=False
    )
    return (data, -half_g)
