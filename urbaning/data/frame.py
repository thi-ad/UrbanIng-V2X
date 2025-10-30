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
from typing import Dict, Optional, Union, List, Mapping, Tuple

from .camera_data import CameraData
from .lidar_data import LidarData, _transform_points
from .labels import Labels
from .vehicle import Vehicle
from .infrastructure import Infrastructure
from .registry import GLOBAL, _xTg_registry


class Frame:
    """Represents a frame of multi-sensor data captured at a specific timestamp.

    A `Frame` aggregates all vehicles, infrastructures, and their corresponding
    sensor data (LiDAR, camera, etc.) for a given point in time. It provides
    functionality to fuse and transform point clouds between various coordinate systems.

    Attributes
    ----------
    timestamp : float
        The timestamp of the frame (in seconds or milliseconds, depending on dataset convention).
    vehicles : Dict[str, Vehicle]
        A dictionary mapping unique vehicle identifiers to `Vehicle` objects present in the frame.
    infrastructures : Dict[str, Infrastructure]
        A dictionary mapping unique infrastructure identifiers to `Infrastructure` objects.
    labels : Optional[Labels]
        Optional labels associated with the frame, typically containing ground-truth annotations.
    GLOBAL : str
        A string constant representing the global coordinate system name.
    _fused_point_cloud_in_global_coordinates : Optional[np.ndarray]
        Cached fused point cloud of all sources in global coordinates. Computed lazily
        when first accessed via :meth:`fused_point_cloud_in_global_coordinates`.
    """

    def __init__(self, timestamp: float):
        """
        Initialize a `Frame` instance.

        Parameters
        ----------
        timestamp : float
            The timestamp of the frame (in seconds or milliseconds, depending on dataset convention).
        """
        self.timestamp: float = timestamp

        self.vehicles: Dict[str, Vehicle] = {}
        self.infrastructures: Dict[str, Infrastructure] = {}
        self.labels: Optional[Labels] = None

        self.GLOBAL: str = GLOBAL

        self._fused_point_cloud_in_global_coordinates: Optional[np.ndarray] = None

    def add_coordinates(self, coordinate_name: str, GLOBAL_T_x: np.ndarray) -> None:
        """Register a new coordinate system relative to the global frame.

        Parameters
        ----------
        coordinate_name : str
            The name of the new coordinate system to register.
        GLOBAL_T_x : np.ndarray
            A 4x4 homogeneous transformation matrix representing the transform
            from the global coordinate system to this coordinate system.
        """
        coordinate_name_string = f"{coordinate_name}_coordinates"
        setattr(self, coordinate_name, coordinate_name_string)
        _xTg_registry[getattr(self, coordinate_name)] = np.linalg.inv(GLOBAL_T_x)

    def set_labels(self, labels_obj: Labels) -> None:
        """Attach a set of labels to the frame.

        Parameters
        ----------
        labels_obj : Labels
            The labels object containing annotated information for this frame.
        """
        self.labels = labels_obj

    def fused_point_cloud_in_global_coordinates(self) -> np.ndarray:
        """Fuse and return the combined point cloud of all sources in global coordinates.

        Combines all vehicle and infrastructure point clouds, transforming each
        into the global coordinate frame.

        Returns
        -------
        np.ndarray
            Fused point cloud of shape (N, 3) or (N, 4) in global coordinates.
        """
        if self._fused_point_cloud_in_global_coordinates is None:
            self._fused_point_cloud_in_global_coordinates = np.concatenate(
                [vehicle.fused_point_cloud_in_global_coordinates() for vehicle in self.vehicles.values()]
                + [infra.fused_point_cloud_in_global_coordinates() for infra in self.infrastructures.values()],
                axis=0,
            )
        return self._fused_point_cloud_in_global_coordinates

    def fused_point_cloud_in_this_coordinates(self, origin: str) -> np.ndarray:
        """Transform the fused point cloud from global to the specified coordinate frame.

        Parameters
        ----------
        origin : str
            The target coordinate frame identifier (must exist in `_xTg_registry`).

        Returns
        -------
        np.ndarray
            Transformed fused point cloud in the specified coordinate frame.

        Raises
        ------
        KeyError
            If the provided origin does not exist in the `_xTg_registry`.
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

    def fused_point_cloud_in_infrastructure_coordinates(self, infra: Infrastructure) -> np.ndarray:
        """Get the fused point cloud in an infrastructure's coordinate frame.

        Parameters
        ----------
        infra : Infrastructure
            The infrastructure whose coordinate system is used.

        Returns
        -------
        np.ndarray
            Fused point cloud in the infrastructure's coordinates.
        """
        return self.fused_point_cloud_in_this_coordinates(infra.INFRASTRUCTURE)

    def fused_point_cloud_in_lidar_coordinates(self, other_lidar: LidarData) -> np.ndarray:
        """Get the fused point cloud in a LiDAR's coordinate frame.

        Parameters
        ----------
        other_lidar : LidarData
            The LiDAR sensor whose coordinate frame is used.

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

    def fused_point_cloud_in_image_coordinates(self, camera: CameraData)  -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project the fused point cloud onto a camera image plane.

        Parameters
        ----------
        camera : CameraData
            The camera object that defines the projection parameters.

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

    @staticmethod
    def get_point_cloud_in_global_coordinates(
        sources: Union[Frame, List, Mapping, Vehicle, Infrastructure, LidarData]
    ) -> np.ndarray:
        """Get point cloud(s) in global coordinates from various input types.

        Parameters
        ----------
        sources : Union[Frame, List, Mapping, Vehicle, Infrastructure, LidarData]
            The source(s) from which to retrieve the point cloud(s). Can be a single object,
            list, or mapping of objects.

        Returns
        -------
        np.ndarray
            Concatenated point cloud(s) in global coordinates.

        Raises
        ------
        KeyError
            If an unsupported type is passed as `sources`.
        """
        if isinstance(sources, (Frame, Vehicle, Infrastructure)):
            return sources.fused_point_cloud_in_global_coordinates()
        elif isinstance(sources, List):
            return np.concatenate([Frame.get_point_cloud_in_global_coordinates(s) for s in sources], axis=0)
        elif isinstance(sources, Mapping):
            return np.concatenate([Frame.get_point_cloud_in_global_coordinates(s) for s in sources.values()], axis=0)
        elif isinstance(sources, LidarData):
            return sources.point_cloud_in_global_coordinates()
        else:
            raise KeyError(f"Unknown sources found: {sources}")

    def get_point_cloud(
        self,
        sources: Union[Frame, List, Mapping, Vehicle, Infrastructure, LidarData],
        origin: str = "global_coordinates"
    ) -> np.ndarray:
        """Get point cloud(s) transformed to a specified coordinate frame.

        Parameters
        ----------
        sources : Union[Frame, List, Mapping, Vehicle, Infrastructure, LidarData]
            The source(s) from which to retrieve the point cloud(s).
        origin : str, default="global_coordinates"
            The target coordinate frame to which points should be transformed.

        Returns
        -------
        np.ndarray
            Transformed point cloud(s) in the specified coordinate frame.

        Raises
        ------
        KeyError
            If the provided origin does not exist in the `_xTg_registry`.
        """
        point_cloud_in_global = self.get_point_cloud_in_global_coordinates(sources)
        return _transform_points(_xTg_registry[origin], point_cloud_in_global)