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
from typing import Dict, Optional
import numpy as np

from .camera_data import CameraData
from .lidar_data import LidarData
from .object_label import ObjectLabel
from .vehicle_state import VehicleState
from .infrastructure import Infrastructure
from .utils import get_vehicle_states_id_mapping
from .registry import _transform_points, _xTg_registry


class Vehicle:
    """Represents a vehicle with sensors, state, and fused point cloud capabilities.

    This class manages LiDARs, cameras, and vehicle state data. It also provides
    methods to retrieve fused point clouds in various coordinate frames and converts
    the vehicle state into an `ObjectLabel`.

    Attributes
    ----------
    vehicle_name : str
        Name of the vehicle.
    vehicle_id : int
        Unique identifier for the vehicle derived from vehicle state mapping.
    cameras : Dict[str, CameraData]
        Dictionary of camera sensors attached to the vehicle.
    lidars : Dict[str, LidarData]
        Dictionary of LiDAR sensors attached to the vehicle.
    state : Optional[VehicleState]
        Vehicle state information including position, orientation, and dimensions.
    BODY : str
        Name of the vehicle body coordinate frame.
    HORIZONTAL : str
        Name of the vehicle horizontal coordinate frame.
    _fused_point_cloud_in_global_coordinates : Optional[np.ndarray]
        Cached fused point cloud in global coordinates.
    _fused_point_cloud_in_vehicle_body_coordinates : Optional[np.ndarray]
        Cached fused point cloud in vehicle body coordinates.
    _fused_point_cloud_in_vehicle_horizontal_coordinates : Optional[np.ndarray]
        Cached fused point cloud in vehicle horizontal coordinates.
    _state_as_label : Optional[ObjectLabel]
        Cached ObjectLabel representation of the vehicle state.
    """

    def __init__(self, vehicle_name: str):
        """
        Initialize a Vehicle object.

        Parameters
        ----------
        vehicle_name : str
            Name of the vehicle.
        """
        self.vehicle_name: str = vehicle_name
        self.vehicle_id: int = get_vehicle_states_id_mapping()[self.vehicle_name + "_state"]

        self.cameras: Dict[str, CameraData] = {}
        self.lidars: Dict[str, LidarData] = {}
        self.state: Optional[VehicleState] = None

        self._fused_point_cloud_in_global_coordinates: Optional[np.ndarray] = None
        self._fused_point_cloud_in_vehicle_body_coordinates: Optional[np.ndarray] = None
        self._fused_point_cloud_in_vehicle_horizontal_coordinates: Optional[np.ndarray] = None
        self._state_as_label: Optional[ObjectLabel] = None

        self.BODY: str = f"{self.vehicle_name}_body_coordinates"
        self.HORIZONTAL: str = f"{self.vehicle_name}_horizontal_coordinates"

    def add_camera(self, name: str, camera_data: CameraData) -> None:
        """Add a camera to the vehicle."""
        self.cameras[name] = camera_data

    def add_lidar(self, name: str, lidar_data: LidarData) -> None:
        """Add a LiDAR to the vehicle."""
        self.lidars[name] = lidar_data

    def add_state(self, vehicle_state: VehicleState) -> None:
        """Add the vehicle's state and update the global registry for its frames."""
        self.state = vehicle_state
        _xTg_registry[self.BODY] = vehicle_state.vTg_body
        _xTg_registry[self.HORIZONTAL] = vehicle_state.vTg_horizontal

    def add_coordinates(self, coordinate_name: str, BODY_T_x: np.ndarray) -> None:
        """Add an arbitrary coordinate frame relative to the vehicle body.

        Parameters
        ----------
        coordinate_name : str
            Name of the new coordinate frame.
        BODY_T_x : np.ndarray
            4x4 transformation matrix from the new frame to the vehicle body.
        """
        coordinate_name_string = f"{self.vehicle_name}_{coordinate_name}_coordinates"
        setattr(self, coordinate_name, coordinate_name_string)
        _xTg_registry[getattr(self, coordinate_name)] = np.linalg.inv(BODY_T_x) @ _xTg_registry[self.BODY]

    def state_as_label(self) -> ObjectLabel:
        """Return the vehicle state as an `ObjectLabel`."""
        if self._state_as_label is None:
            self._state_as_label = ObjectLabel(
                track_id=self.vehicle_id,
                object_type="Car",
                position=self.state.position,
                quaternion=self.state.quaternion,
                dimension=self.state.dimension,
                attributes={},
            )
        return self._state_as_label

    def fused_point_cloud_in_global_coordinates(self) -> np.ndarray:
        """Return fused point cloud from all LiDARs in global coordinates."""
        if self._fused_point_cloud_in_global_coordinates is None:
            self._fused_point_cloud_in_global_coordinates = np.concatenate(
                [lidar.point_cloud_in_global_coordinates() for lidar in self.lidars.values()],
                axis=0,
            )
        return self._fused_point_cloud_in_global_coordinates

    def fused_point_cloud_in_this_coordinates(self, origin: str) -> np.ndarray:
        """Transform fused point cloud to the specified coordinate frame.

        Parameters
        ----------
        origin : str
            Target coordinate frame.

        Returns
        -------
        np.ndarray
            Fused point cloud transformed to the target frame.
        """
        return _transform_points(_xTg_registry[origin], self.fused_point_cloud_in_global_coordinates())

    def fused_point_cloud_in_vehicle_body_coordinates(self) -> np.ndarray:
        """Return fused point cloud in the vehicle body coordinate frame."""
        if self._fused_point_cloud_in_vehicle_body_coordinates is None:
            self._fused_point_cloud_in_vehicle_body_coordinates = self.fused_point_cloud_in_this_coordinates(self.BODY)
        return self._fused_point_cloud_in_vehicle_body_coordinates

    def fused_point_cloud_in_vehicle_horizontal_coordinates(self) -> np.ndarray:
        """Return fused point cloud in the vehicle horizontal coordinate frame."""
        if self._fused_point_cloud_in_vehicle_horizontal_coordinates is None:
            self._fused_point_cloud_in_vehicle_horizontal_coordinates = self.fused_point_cloud_in_this_coordinates(self.HORIZONTAL)
        return self._fused_point_cloud_in_vehicle_horizontal_coordinates

    def fused_point_cloud_in_other_vehicle_body_coordinates(self, other_vehicle: Vehicle) -> np.ndarray:
        """Return fused point cloud in another vehicle's body coordinate frame."""
        return self.fused_point_cloud_in_this_coordinates(other_vehicle.BODY)

    def fused_point_cloud_in_other_vehicle_horizontal_coordinates(self, other_vehicle: Vehicle) -> np.ndarray:
        """Return fused point cloud in another vehicle's horizontal coordinate frame."""
        return self.fused_point_cloud_in_this_coordinates(other_vehicle.HORIZONTAL)

    def fused_point_cloud_in_infrastructure_coordinates(self, infra: Infrastructure) -> np.ndarray:
        """Return fused point cloud in a given infrastructure's coordinate frame."""
        return self.fused_point_cloud_in_this_coordinates(infra.INFRASTRUCTURE)

    def fused_point_cloud_in_lidar_coordinates(self, lidar: LidarData) -> np.ndarray:
        """Return fused point cloud in a given LiDAR's coordinate frame."""
        return self.fused_point_cloud_in_this_coordinates(lidar.LIDAR)

    def fused_point_cloud_in_camera_coordinates(self, camera: CameraData) -> np.ndarray:
        """Return fused point cloud in a given camera's coordinate frame."""
        return self.fused_point_cloud_in_this_coordinates(camera.CAMERA)

    def fused_point_cloud_in_image_coordinates(self, camera: CameraData) -> np.ndarray:
        """Return fused point cloud in image coordinates of a given camera."""
        fused_points = self.fused_point_cloud_in_camera_coordinates(camera)
        return camera.camera_coordinates_to_image_coordinates(fused_points)
