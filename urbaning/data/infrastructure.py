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

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Tuple
import numpy as np

from .registry import _transform_points, _xTg_registry

if TYPE_CHECKING:
    from .camera_data import CameraData
    from .lidar_data import LidarData
    from .vehicle import Vehicle


class Infrastructure:
    """Represents a fixed infrastructure entity equipped with sensors.

    The `Infrastructure` class models a stationary structure (e.g., roadside unit,
    traffic light pole, or camera tower) containing multiple cameras and/or LiDARs.
    It manages coordinate systems for each sensor, fuses point clouds, and allows
    transformations between global, infrastructure, and other sensor coordinate frames.

    Attributes
    ----------
    infrastructure_name : str
        The unique name of the infrastructure instance.
    cameras : Dict[str, CameraData]
        Dictionary mapping camera identifiers to their associated `CameraData` objects.
    lidars : Dict[str, LidarData]
        Dictionary mapping LiDAR identifiers to their associated `LidarData` objects.
    _fused_point_cloud_in_global_coordinates : Optional[np.ndarray]
        Cached fused point cloud of all LiDARs in global coordinates. Computed lazily
        on first access.
    INFRASTRUCTURE : str
        String identifier representing this infrastructure's coordinate frame.
    """

    def __init__(self, infrastructure_name: str):
        """
        Initialize an `Infrastructure` instance.

        Parameters
        ----------
        infrastructure_name : str
            The unique name of the infrastructure (e.g., "intersection_1").
        """
        self.infrastructure_name: str = infrastructure_name

        self.cameras: Dict[str, CameraData] = {}
        self.lidars: Dict[str, LidarData] = {}

        self._fused_point_cloud_in_global_coordinates: Optional[np.ndarray] = None

        self.INFRASTRUCTURE: str = f"{self.infrastructure_name}_infrastructure_coordinates"
        _xTg_registry[self.INFRASTRUCTURE] = np.eye(4)

    def add_camera(self, name: str, camera_data: CameraData) -> None:
        """Add a camera to this infrastructure.

        Parameters
        ----------
        name : str
            Identifier for the camera.
        camera_data : CameraData
            The `CameraData` object representing the camera.
        """
        self.cameras[name] = camera_data

    def add_lidar(self, name: str, lidar_data: LidarData) -> None:
        """Add a LiDAR sensor to this infrastructure.

        Parameters
        ----------
        name : str
            Identifier for the LiDAR.
        lidar_data : LidarData
            The `LidarData` object representing the LiDAR sensor.
        """
        self.lidars[name] = lidar_data

    def add_coordinates(self, coordinate_name: str, x_T_INFRASTRUCTURE: np.ndarray) -> None:
        """Register a new coordinate frame relative to this infrastructure.

        Parameters
        ----------
        coordinate_name : str
            The name of the new coordinate system to register.
        x_T_INFRASTRUCTURE : np.ndarray
            A 4x4 homogeneous transformation matrix representing the transform
            from this infrastructure's coordinate system to the new one.
        """
        eval(f"self.{coordinate_name} = {self.infrastructure_name}_{coordinate_name}_infrastructure_coordinates")
        _xTg_registry[eval(f"self.{coordinate_name}")] = x_T_INFRASTRUCTURE @ _xTg_registry[self.INFRASTRUCTURE]

    def fused_point_cloud_in_global_coordinates(self) -> np.ndarray:
        """Fuse and return all LiDAR point clouds in global coordinates.

        Returns
        -------
        np.ndarray
            Fused LiDAR point cloud in global coordinates of shape (N, 3) or (N, 4).
        """
        if self._fused_point_cloud_in_global_coordinates is None:
            self._fused_point_cloud_in_global_coordinates = np.concatenate(
                [lidar.point_cloud_in_global_coordinates() for lidar in self.lidars.values()],
                axis=0,
            )
        return self._fused_point_cloud_in_global_coordinates

    def fused_point_cloud_in_this_coordinates(self, origin: str) -> np.ndarray:
        """Transform the fused point cloud from global to a specified coordinate frame.

        Parameters
        ----------
        origin : str
            The target coordinate frame name.

        Returns
        -------
        np.ndarray
            Fused point cloud transformed to the target coordinate frame.

        Raises
        ------
        KeyError
            If the target coordinate frame is not found in `_xTg_registry`.
        """
        return _transform_points(_xTg_registry[origin], self.fused_point_cloud_in_global_coordinates())

    def fused_point_cloud_in_vehicle_body_coordinates(self, vehicle: Vehicle) -> np.ndarray:
        """Get the fused point cloud in a vehicle's body coordinate frame.

        Parameters
        ----------
        vehicle : Vehicle
            The vehicle whose body coordinate system is used.

        Returns
        -------
        np.ndarray
            Fused point cloud in the vehicle's body coordinates.
        """
        return self.fused_point_cloud_in_this_coordinates(vehicle.BODY)

    def fused_point_cloud_in_vehicle_horizontal_coordinates(self, vehicle: Vehicle) -> np.ndarray:
        """Get the fused point cloud in a vehicle's horizontal coordinate frame.

        Parameters
        ----------
        vehicle : Vehicle
            The vehicle whose horizontal coordinate system is used.

        Returns
        -------
        np.ndarray
            Fused point cloud in the vehicle's horizontal coordinates.
        """
        return self.fused_point_cloud_in_this_coordinates(vehicle.HORIZONTAL)

    def fused_point_cloud_in_infrastructure_coordinates(self) -> np.ndarray:
        """Get the fused point cloud in this infrastructure's coordinate frame.

        Returns
        -------
        np.ndarray
            Fused point cloud in the infrastructure's local coordinates.
        """
        return self.fused_point_cloud_in_this_coordinates(self.INFRASTRUCTURE)

    def fused_point_cloud_in_other_infrastructure_coordinates(self, infra: Infrastructure) -> np.ndarray:
        """Get the fused point cloud in another infrastructure's coordinate frame.

        Parameters
        ----------
        infra : Infrastructure
            The target infrastructure whose coordinate system is used.

        Returns
        -------
        np.ndarray
            Fused point cloud in the target infrastructure's coordinates.
        """
        return self.fused_point_cloud_in_this_coordinates(infra.INFRASTRUCTURE)

    def fused_point_cloud_in_lidar_coordinates(self, other_lidar: LidarData) -> np.ndarray:
        """Get the fused point cloud in a LiDAR's coordinate frame.

        Parameters
        ----------
        other_lidar : LidarData
            The LiDAR whose coordinate system is used.

        Returns
        -------
        np.ndarray
            Fused point cloud in the LiDAR's coordinates.
        """
        return self.fused_point_cloud_in_this_coordinates(other_lidar.LIDAR)

    def fused_point_cloud_in_camera_coordinates(self, camera: CameraData) -> np.ndarray:
        """Get the fused point cloud in a camera's coordinate frame.

        Parameters
        ----------
        camera : CameraData
            The camera whose coordinate frame is used.

        Returns
        -------
        np.ndarray
            Fused point cloud in the camera's coordinates.
        """
        return self.fused_point_cloud_in_this_coordinates(camera.CAMERA)

    def fused_point_cloud_in_image_coordinates(
            self, camera: CameraData
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project the fused point cloud onto a camera image plane.

        Parameters
        ----------
        camera : CameraData
            The camera used for projection.

        Returns
        -------
        image_points_out : np.ndarray
            2D integer pixel coordinates of the projected points, shape (..., 2).
        valid_flag : np.ndarray
            Boolean mask indicating which points are in front of the camera (z > 0).
        inside_image_flag : np.ndarray
            Boolean mask indicating which projected points fall inside image boundaries.
        """
        fused_point_cloud_in_camera_coordinates = self.fused_point_cloud_in_camera_coordinates(camera)
        return camera.camera_coordinates_to_image_coordinates(fused_point_cloud_in_camera_coordinates)
