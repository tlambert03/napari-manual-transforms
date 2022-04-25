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
