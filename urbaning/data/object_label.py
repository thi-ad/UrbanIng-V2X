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
import numpy as np
from typing import Mapping
from scipy.spatial.transform import Rotation

from .registry import GLOBAL, _xTg_registry
from .utils import get_vPv_numpy

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .vehicle import Vehicle
    from .infrastructure import Infrastructure
    from .lidar_data import LidarData
    from .camera_data import CameraData


classes = [
    "Car", "Van", "Bus", "Truck", "Trailer", "OtherVehicle",
    "Cyclist", "Motorcycle", "EScooter", "Pedestrian",
    "OtherPedestrians", "Animal", "Other"
]


class ObjectLabel:
    """Represents a labeled 3D object with position, orientation, and dimensions.

    The `ObjectLabel` class encapsulates information about detected or annotated
    objects, including their 3D bounding box, coordinate transformations,
    and camera projection properties.

    Attributes
    ----------
    track_id : int
        Unique identifier for the tracked object.
    object_type : str
        Type or class of the object (e.g., 'Car', 'Pedestrian', etc.).
    position : np.ndarray
        3D position of the object in the origin coordinate frame, shape (3,).
    quaternion : np.ndarray
        Quaternion `(x, y, z, w)` representing the object's orientation.
    dimension : np.ndarray
        Dimensions `(length, width, height)` of the object in meters.
    attributes : Mapping
        Additional metadata or attributes describing the object.
    origin : str
        The coordinate frame in which this label is currently expressed.
    _globalTobject : np.ndarray or None
        Cached 4x4 transformation from global coordinates to object coordinates.
    _globalTorigin : np.ndarray or None
        Cached 4x4 transformation from global to origin coordinates.
    _originTobject : np.ndarray or None
        Cached 4x4 transformation from origin to object coordinates.
    _corners_in_object_coordinates : np.ndarray or None
        Cached array of 3D bounding box corners in object coordinates, shape (8, 4).
    _corners_in_current_coordinates : np.ndarray or None
        Cached array of corners transformed to the current origin coordinates.
    _corners_in_global_coordinates : np.ndarray or None
        Cached array of corners transformed to global coordinates.
    corners_in_image_coordinates : np.ndarray or None
        Projected corners in 2D image coordinates, if available.
    corners_in_image_valid_flag : np.ndarray or None
        Boolean mask indicating whether each projected corner is valid.
    corners_in_image_inside_flag : np.ndarray or None
        Boolean mask indicating whether each corner lies within the image frame.
    position_in_image_coordinates : np.ndarray or None
        Projected 2D position of the object center in image coordinates.
    position_in_image_valid_flag : bool or None
        Whether the projected object center is valid.
    position_in_image_inside_flag : bool or None
        Whether the projected object center is inside the image frame.
    """

    def __init__(
        self,
        track_id: int,
        object_type: str,
        position: np.ndarray,
        quaternion: np.ndarray,
        dimension: np.ndarray,
        attributes: Mapping,
        origin: str = GLOBAL,
    ) -> None:
        """
        Initialize an `ObjectLabel` instance.

        Parameters
        ----------
        track_id : int
            Unique track identifier for the object.
        object_type : str
            Category or class of the object.
        position : np.ndarray
            3D position vector of shape (3,).
        quaternion : np.ndarray
            Quaternion `(x, y, z, w)` representing the object's rotation.
        dimension : np.ndarray
            Object dimensions `(length, width, height)` in meters.
        attributes : Mapping
            Additional object-specific metadata (e.g., color, state, source).
        origin : str, optional
            Coordinate frame name where the label is currently expressed (default is `"global_coordinates"`).
        """
        self.track_id = track_id
        self.object_type = object_type
        self.position = position
        self.quaternion = quaternion
        self.dimension = dimension
        self.attributes = attributes
        self.origin = origin

        self._globalTobject = None
        self._globalTorigin = None
        self._originTobject = None

        self._corners_in_object_coordinates = None
        self._corners_in_current_coordinates = None
        self._corners_in_global_coordinates = None

        self.corners_in_image_coordinates = None
        self.corners_in_image_valid_flag = None
        self.corners_in_image_inside_flag = None

        self.position_in_image_coordinates = None
        self.position_in_image_valid_flag = None
        self.position_in_image_inside_flag = None

    def globalTobject(self) -> np.ndarray:
        """Return the transformation from global coordinates to object coordinates.

        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix.
        """
        if self._globalTobject is None:
            self._globalTobject = self.globalTorigin() @ self.originTobject()
        return self._globalTobject

    def globalTorigin(self) -> np.ndarray:
        """Return the transformation from global coordinates to the object's origin coordinates.

        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix.
        """
        if self._globalTorigin is None:
            self._globalTorigin = _xTg_registry.resolve_aTb(GLOBAL, self.origin)
        return self._globalTorigin

    def originTobject(self) -> np.ndarray:
        """Return the transformation from the origin coordinate frame to the object coordinate frame.

        Returns
        -------
        np.ndarray
            4x4 homogeneous transformation matrix representing translation and rotation.
        """
        if self._originTobject is None:
            originTobject = np.eye(4)
            originTobject[:3, 3] = self.position
            originTobject[:3, :3] = Rotation.from_quat(self.quaternion).as_matrix()
            self._originTobject = originTobject
        return self._originTobject

    def corners_in_object_coordinates(self) -> np.ndarray:
        """Return 3D bounding box corners in object coordinates.

        Returns
        -------
        np.ndarray
            Array of shape (8, 4) containing homogeneous 3D corner coordinates.
        """
        if self._corners_in_object_coordinates is None:
            self._corners_in_object_coordinates = get_vPv_numpy(*self.dimension)
        return self._corners_in_object_coordinates

    def corners_in_global_coordinates(self) -> np.ndarray:
        """Return 3D bounding box corners transformed into global coordinates.

        Returns
        -------
        np.ndarray
            Array of shape (8, 4) representing the corners in global coordinates.
        """
        if self._corners_in_global_coordinates is None:
            self._corners_in_global_coordinates = (
                self.corners_in_object_coordinates() @ self.globalTobject().T
            )
        return self._corners_in_global_coordinates

    def corners_in_current_coordinates(self) -> np.ndarray:
        """Return 3D bounding box corners in the current origin's coordinate frame.

        Returns
        -------
        np.ndarray
            Array of shape (8, 4) representing the corners in the current origin frame.
        """
        if self._corners_in_current_coordinates is None:
            self._corners_in_current_coordinates = (
                self.corners_in_object_coordinates() @ self.originTobject().T
            )
        return self._corners_in_current_coordinates

    def convert_to_this_coordinates(self, origin: str) -> ObjectLabel:
        """Convert this object label to another coordinate frame.

        Parameters
        ----------
        origin : str
            Target coordinate frame name.

        Returns
        -------
        ObjectLabel
            New object label expressed in the specified coordinate frame.
        """
        if origin == self.origin:
            return self

        neworiginTcurrentorigin = _xTg_registry.resolve_aTb(origin, self.origin)
        neworiginTobject = neworiginTcurrentorigin @ self.originTobject()

        position = neworiginTobject[:3, 3]
        quaternion = Rotation.from_matrix(neworiginTobject[:3, :3]).as_quat()

        new_object = ObjectLabel(
            track_id=self.track_id,
            object_type=self.object_type,
            position=position,
            quaternion=quaternion,
            dimension=self.dimension,
            attributes=self.attributes,
            origin=origin,
        )

        new_object._globalTobject = self._globalTobject
        new_object._corners_in_object_coordinates = self._corners_in_object_coordinates
        new_object._corners_in_global_coordinates = self._corners_in_global_coordinates

        return new_object

    def convert_to_global_coordinates(self) -> ObjectLabel:
        """Convert this object label to global coordinates.

        Returns
        -------
        ObjectLabel
            New object label in global coordinates.
        """
        if self.origin == GLOBAL:
            return self
        return self.convert_to_this_coordinates(GLOBAL)

    def convert_to_vehicle_body_coordinates(self, vehicle: Vehicle) -> ObjectLabel:
        """Convert this object label to a vehicle's body coordinate frame."""
        return self.convert_to_this_coordinates(vehicle.BODY)

    def convert_to_vehicle_horizontal_coordinates(self, vehicle: Vehicle) -> ObjectLabel:
        """Convert this object label to a vehicle's horizontal coordinate frame."""
        return self.convert_to_this_coordinates(vehicle.HORIZONTAL)

    def convert_to_infrastructure_coordinates(self, infra: Infrastructure) -> ObjectLabel:
        """Convert this object label to an infrastructure coordinate frame."""
        return self.convert_to_this_coordinates(infra.INFRASTRUCTURE)

    def convert_to_lidar_coordinates(self, lidar: LidarData) -> ObjectLabel:
        """Convert this object label to a LiDAR coordinate frame."""
        return self.convert_to_this_coordinates(lidar.LIDAR)

    def convert_to_camera_coordinates(self, camera: CameraData) -> ObjectLabel:
        """Convert this object label to a camera coordinate frame."""
        return self.convert_to_this_coordinates(camera.CAMERA)

    def convert_to_image_coordinates(self, camera: CameraData) -> ObjectLabel:
        """Project the object and its bounding box into image coordinates.

        Parameters
        ----------
        camera : CameraData
            Target camera instance.

        Returns
        -------
        ObjectLabel
            New object label with 2D image projections (position and corners) populated.
        """
        new_object = self.convert_to_this_coordinates(camera.CAMERA)
        image_points_position, valid_flag_position, inside_image_flag_position = (
            camera.camera_coordinates_to_image_coordinates(new_object.position[None, :])
        )

        new_object.position_in_image_coordinates = image_points_position[0]
        new_object.position_in_image_valid_flag = valid_flag_position[0]
        new_object.position_in_image_inside_flag = inside_image_flag_position[0]

        image_points_corners, valid_flag_corners, inside_image_flag_corners = (
            camera.camera_coordinates_to_image_coordinates(new_object.corners_in_current_coordinates())
        )

        new_object.corners_in_image_coordinates = image_points_corners
        new_object.corners_in_image_valid_flag = valid_flag_corners
        new_object.corners_in_image_inside_flag = inside_image_flag_corners

        return new_object

    def transform_points_from_object_coordinates_to_current_coordinates(
        self, points_in_object_coordinates: np.ndarray
    ) -> np.ndarray:
        """Transform arbitrary 3D points from object coordinates to the current origin coordinates.

        Parameters
        ----------
        points_in_object_coordinates : np.ndarray
            Array of shape (3, N) representing 3D points in the object coordinate frame.

        Returns
        -------
        np.ndarray
            Transformed points of shape (3, N) in the current origin coordinate frame.
        """
        originTobject = self.originTobject()
        points_in_current_coordinates = (
            originTobject[:3, :3] @ points_in_object_coordinates[:3, ...]
            + originTobject[:3, 3:4]
        )
        return points_in_current_coordinates
