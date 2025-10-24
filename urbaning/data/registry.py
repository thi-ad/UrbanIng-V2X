#    Urbaning
#    Copyright (C) 2025  Technische Hochschule Ingolstadt
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
