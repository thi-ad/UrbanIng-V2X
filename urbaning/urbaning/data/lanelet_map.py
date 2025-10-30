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

from __future__ import annotations
import numpy as np
try:
    from lanelet2.core import GPSPoint, BasicPoint3d
    from lanelet2.io import load, Origin
    from lanelet2.projection import UtmProjector
    LANELET2FOUND = True
except:
    pass
    LANELET2FOUND = False

from .registry import _xTg_registry, _transform_points


class LLMap:
    """Represents a Lanelet2 map with ground elevation modeling and coordinate transformations.

    The `LLMap` class provides access to map data loaded from a Lanelet2 `.osm` file and
    offers utilities for computing ground elevation (`z` values) for given (x, y) coordinates,
    as well as transforming those points into various coordinate systems.

    Attributes
    ----------
    map_file_name : str
        Path to the Lanelet2 map file (typically a `.osm` file).
    lato : float
        Latitude of the map's origin in degrees.
    lono : float
        Longitude of the map's origin in degrees.
    alto : float
        Altitude of the map's origin in meters.
    ground_params : tuple[float, float, float, float, float, float]
        Coefficients `(a, b, c, d, e, f)` for the quadratic ground elevation model:
        ``z = a*x² + b*y² + c*x*y + d*x + e*y + f``.
    projector : lanelet2.projection.UtmProjector
        The UTM projector used to convert between GPS and Cartesian coordinates.
    map : lanelet2.core.LaneletMap
        The loaded Lanelet2 map object.
    """

    def __init__(self, map_file: str, origin: tuple[float, float, float], ground_params: tuple[float, float, float, float, float, float]):
        """
        Initialize a `LLMap` instance.

        Parameters
        ----------
        map_file : str
            Path to the Lanelet2 map file (e.g., `'path/to/map.osm'`).
        origin : tuple[float, float, float]
            Tuple containing `(latitude, longitude, altitude)` of the map origin.
        ground_params : tuple[float, float, float, float, float, float]
            Coefficients `(a, b, c, d, e, f)` for the ground elevation model.
        """
        self.map_file_name: str = map_file
        self.lato, self.lono, self.alto = origin
        self.ground_params: tuple[float, float, float, float, float, float] = ground_params

        if LANELET2FOUND:
            self.projector: UtmProjector = UtmProjector(Origin(self.lato, self.lono))
            self.map = load(self.map_file_name, self.projector)

    def get_z_values_of_points(self, points: np.ndarray) -> np.ndarray:
        """Compute ground elevation (z-values) for given (x, y) points.

        The ground elevation is computed using a quadratic surface model:
        ``z = a*x² + b*y² + c*x*y + d*x + e*y + f``.

        Parameters
        ----------
        points : np.ndarray
            Array of 2D points of shape (N, 2), where each row represents `(x, y)` coordinates.

        Returns
        -------
        np.ndarray
            1D array of computed `z` values of shape (N,).
        """
        x = points[:, 0]
        y = points[:, 1]
        a, b, c, d, e, f = self.ground_params
        z = a * x * x + b * y * y + c * x * y + d * x + e * y + f
        return z

    def transform_points(self, g_points: np.ndarray, origin: str) -> np.ndarray:
        """Transform points from global (x, y) coordinates into another coordinate frame.

        The input points are augmented with estimated `z` values (based on the ground model),
        and then transformed using the corresponding transformation matrix stored in `_xTg_registry`.

        Parameters
        ----------
        g_points : np.ndarray
            Array of points in global coordinates of shape (N, 2), where each row is `(x, y)`.
        origin : str
            The target coordinate frame name (must exist in `_xTg_registry`).

        Returns
        -------
        np.ndarray
            Transformed 3D points in the target coordinate frame, of shape (N, 3).

        Raises
        ------
        KeyError
            If the specified `origin` does not exist in `_xTg_registry`.
        """
        xTg = _xTg_registry[origin]  # 4x4 transformation matrix

        pts = np.empty((g_points.shape[0], 3), dtype=g_points.dtype)
        pts[:, :2] = g_points[:, :2]
        pts[:, 2] = self.get_z_values_of_points(g_points)

        return _transform_points(xTg, pts)