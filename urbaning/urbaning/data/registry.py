#    Urbaning
#    Copyright (C) 2025  Technische Hochschule Ingolstadt
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import numpy as np

GLOBAL: str = "global_coordinates"


def _transform_points(xTy: np.ndarray, y_points: np.ndarray) -> np.ndarray:
    """Apply a rigid-body transformation to 3D points.

    The function transforms points `y_points` from one coordinate frame into another using
    the 4x4 homogeneous transformation matrix `xTy`.

    Parameters
    ----------
    xTy : np.ndarray
        4x4 homogeneous transformation matrix representing the transformation from frame Y to frame X.
    y_points : np.ndarray
        Array of points in Y frame, shape (N, 3+), where the first 3 columns are (x, y, z).
        Additional columns (e.g., intensity, time) are preserved.

    Returns
    -------
    np.ndarray
        Transformed points in X frame, same shape as `y_points`.
    """
    x_points = y_points.copy()
    x_points[:, :3] = (xTy[:3, :3] @ y_points[:, :3].T + xTy[:3, 3:4]).T
    return x_points


class _XTG_Registry:
    """Registry for storing and resolving 4x4 homogeneous transformations between coordinate frames.

    This class maintains a mapping from frame names to 4x4 transformation matrices
    and provides a method to resolve relative transformations between frames.

    Attributes
    ----------
    xTg_registry : dict[str, np.ndarray]
        Dictionary mapping frame names to 4x4 homogeneous transformation matrices.
    """

    def __init__(self) -> None:
        """Initialize the registry with the global frame."""
        self.xTg_registry: dict[str, np.ndarray] = {GLOBAL: np.eye(4)}

    def __getitem__(self, key: str) -> np.ndarray:
        """Get the transformation matrix for a given frame.

        Parameters
        ----------
        key : str
            Name of the coordinate frame.

        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix associated with the frame.
        """
        return self.xTg_registry[key]

    def __setitem__(self, key: str, value: np.ndarray) -> None:
        """Set a transformation matrix for a given frame.

        Parameters
        ----------
        key : str
            Name of the coordinate frame.
        value : np.ndarray
            4x4 homogeneous transformation matrix.
        """
        self.xTg_registry[key] = value

    def resolve_aTb(self, a: str, b: str) -> np.ndarray:
        """Compute the transformation matrix from frame B to frame A.

        Parameters
        ----------
        a : str
            Name of the target coordinate frame.
        b : str
            Name of the source coordinate frame.

        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix from B to A.
        """
        return self.xTg_registry[a] @ np.linalg.inv(self.xTg_registry[b])


_xTg_registry = _XTG_Registry()
