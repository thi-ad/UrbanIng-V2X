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
from typing import TYPE_CHECKING
import numpy as np
from .registry import _xTg_registry, _transform_points

if TYPE_CHECKING:
    from .camera_data import CameraData
    from .infrastructure import Infrastructure
    from .vehicle import Vehicle


class LidarData:
    """Represents LiDAR sensor data, including transformation and coordinate conversion utilities.

    The `LidarData` class handles LiDAR point cloud data loading, caching, and coordinate
    transformations between LiDAR, global, and other reference frames such as vehicles,
    infrastructures, or cameras.

    Attributes
    ----------
    lidar_name : str
        The unique name of the LiDAR sensor.
    lidar_file_path : str
        Path to the `.npz` file containing LiDAR point cloud data.
    gTl : np.ndarray
        A 4x4 homogeneous transformation matrix from global coordinates to LiDAR coordinates.
    lidar_index : int
        Index of this LiDAR sensor (useful when multiple LiDARs are fused).
    LIDAR : str
        Identifier string for this LiDAR’s coordinate frame (e.g., `"lidar_front_lidar_coordinates"`).
    _lTg : Optional[np.ndarray]
        Cached inverse of `gTl` (LiDAR-to-global transform).
    _point_cloud_in_lidar_coordinates : Optional[np.ndarray]
        Cached LiDAR point cloud in the LiDAR's own coordinate frame.
    _point_cloud_in_global_coordinates : Optional[np.ndarray]
        Cached LiDAR point cloud transformed into the global coordinate frame.
    """

    def __init__(self, lidar_name: str, lidar_file_path: str, gTl: np.ndarray, lidar_index: int):
        """
        Initialize a LiDAR sensor data object.

        Parameters
        ----------
        lidar_name : str
            Name or identifier of the LiDAR sensor.
        lidar_file_path : str
            Path to the `.npz` file containing LiDAR data.
        gTl : np.ndarray
            A 4x4 transformation matrix from global to LiDAR coordinates.
        lidar_index : int
            Index used to identify this LiDAR among multiple sensors.
        """
        self.lidar_name: str = lidar_name
        self.lidar_file_path: str = lidar_file_path
        self.gTl: np.ndarray = gTl
        self.lidar_index: int = lidar_index

        self._lTg: np.ndarray | None = None
        self._point_cloud_in_lidar_coordinates: np.ndarray | None = None
        self._point_cloud_in_global_coordinates: np.ndarray | None = None

        self.LIDAR: str = f"{self.lidar_name}_lidar_coordinates"
        _xTg_registry[self.LIDAR] = self.lTg

    def add_coordinates(self, coordinate_name: str, x_T_LIDAR: np.ndarray) -> None:
        """Add a new coordinate frame relative to this LiDAR’s coordinate system.

        The transformation is registered in the global `_xTg_registry`.

        Parameters
        ----------
        coordinate_name : str
            Name of the new coordinate frame (e.g., `"horizontal"`, `"body"`).
        x_T_LIDAR : np.ndarray
            4x4 transformation matrix from LiDAR coordinates to the new coordinate system.
        """
        eval(f"self.{coordinate_name} = {self.lidar_name}_{coordinate_name}_lidar_coordinates")
        _xTg_registry[eval(f'self.{coordinate_name}')] = x_T_LIDAR @ _xTg_registry[self.LIDAR]

    @property
    def lTg(self) -> np.ndarray:
        """Compute or retrieve the LiDAR-to-global transformation matrix.

        Returns
        -------
        np.ndarray
            A 4x4 transformation matrix from LiDAR coordinates to global coordinates.
        """
        if self._lTg is None:
            self._lTg = np.linalg.inv(self.gTl)
        return self._lTg

    def point_cloud_in_lidar_coordinates(self) -> np.ndarray:
        """Load and cache the LiDAR point cloud in LiDAR coordinates.

        The `.npz` file is expected to contain structured arrays with keys
        `'x'`, `'y'`, `'z'`, `'intensity'`, and `'time_offset_ms'`.

        Returns
        -------
        np.ndarray
            Array of shape (N, 6) containing point cloud data as:
            `[x, y, z, intensity, time_offset_ms, lidar_index]`.
        """
        if self._point_cloud_in_lidar_coordinates is None:
            cali_array = np.load(self.lidar_file_path)
            cali_points = np.ones((len(cali_array['x']), 6), dtype=np.float32)
            cali_points[..., 0] = cali_array['x']
            cali_points[..., 1] = cali_array['y']
            cali_points[..., 2] = cali_array['z']
            cali_points[..., 3] = cali_array['intensity']
            cali_points[..., 4] = cali_array['time_offset_ms']
            cali_points[..., 5] = self.lidar_index
            self._point_cloud_in_lidar_coordinates = cali_points
        return self._point_cloud_in_lidar_coordinates

    def point_cloud_in_global_coordinates(self) -> np.ndarray:
        """Transform and cache the LiDAR point cloud into global coordinates.

        Returns
        -------
        np.ndarray
            Array of shape (N, 6), representing points transformed into the global coordinate frame.
        """
        if self._point_cloud_in_global_coordinates is None:
            point_cloud_in_global_coordinates = _transform_points(self.gTl, self.point_cloud_in_lidar_coordinates())
            self._point_cloud_in_global_coordinates = point_cloud_in_global_coordinates
        return self._point_cloud_in_global_coordinates

    def point_cloud_in_this_coordinates(self, origin: str) -> np.ndarray:
        """Transform the LiDAR point cloud from global coordinates into another coordinate system.

        Parameters
        ----------
        origin : str
            Target coordinate frame name (must exist in `_xTg_registry`).

        Returns
        -------
        np.ndarray
            Transformed LiDAR point cloud of shape (N, 6).
        """
        return _transform_points(_xTg_registry[origin], self.point_cloud_in_global_coordinates())

    def point_cloud_in_vehicle_body_coordinates(self, vehicle: Vehicle) -> np.ndarray:
        """Transform point cloud into a vehicle's body coordinate frame.

        Parameters
        ----------
        vehicle : Vehicle
            The target vehicle instance.

        Returns
        -------
        np.ndarray
            Transformed point cloud in the vehicle body coordinate frame.
        """
        return self.point_cloud_in_this_coordinates(vehicle.BODY)

    def point_cloud_in_vehicle_horizontal_coordinates(self, vehicle: Vehicle) -> np.ndarray:
        """Transform point cloud into a vehicle's horizontal coordinate frame.

        Parameters
        ----------
        vehicle : Vehicle
            The target vehicle instance.

        Returns
        -------
        np.ndarray
            Transformed point cloud in the vehicle horizontal coordinate frame.
        """
        return self.point_cloud_in_this_coordinates(vehicle.HORIZONTAL)

    def point_cloud_in_infrastructure_coordinates(self, infra: Infrastructure) -> np.ndarray:
        """Transform point cloud into an infrastructure coordinate frame.

        Parameters
        ----------
        infra : Infrastructure
            Target infrastructure instance.

        Returns
        -------
        np.ndarray
            Transformed point cloud in the infrastructure coordinate frame.
        """
        return self.point_cloud_in_this_coordinates(infra.INFRASTRUCTURE)

    def point_cloud_in_other_lidar_coordinates(self, other_lidar: LidarData) -> np.ndarray:
        """Transform point cloud into another LiDAR's coordinate frame.

        Parameters
        ----------
        other_lidar : LidarData
            Target LiDAR instance.

        Returns
        -------
        np.ndarray
            Transformed point cloud in the other LiDAR’s coordinate frame.
        """
        return self.point_cloud_in_this_coordinates(other_lidar.LIDAR)

    def point_cloud_in_camera_coordinates(self, camera: CameraData) -> np.ndarray:
        """Transform point cloud into a camera’s coordinate frame.

        Parameters
        ----------
        camera : CameraData
            Target camera instance.

        Returns
        -------
        np.ndarray
            Transformed point cloud in the camera coordinate frame.
        """
        return self.point_cloud_in_this_coordinates(camera.CAMERA)

    def point_cloud_in_image_coordinates(self, camera: CameraData) -> np.ndarray:
        """Project the LiDAR point cloud onto a camera image plane.

        Parameters
        ----------
        camera : CameraData
            Target camera instance.

        Returns
        -------
        np.ndarray
            2D pixel coordinates corresponding to projected LiDAR points.
        """
        point_cloud_in_camera_coordinates = self.point_cloud_in_camera_coordinates(camera)
        return camera.camera_coordinates_to_image_coordinates(point_cloud_in_camera_coordinates)
