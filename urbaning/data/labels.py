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
from typing import Dict, Optional

from .object_label import ObjectLabel
from .vehicle import Vehicle
from .infrastructure import Infrastructure
from .lidar_data import LidarData
from .camera_data import CameraData


class Labels:
    """Container class for managing and transforming labeled object data.

    The `Labels` class stores labeled objects (e.g., detected or annotated vehicles,
    pedestrians, etc.) and provides methods to transform them across coordinate systems,
    such as global, vehicle, infrastructure, LiDAR, or camera frames.

    It also supports filtering out labels corresponding to autonomous vehicles (AVs)
    or other designated tracked entities.

    Attributes
    ----------
    objects : Dict[str, dict]
        Dictionary mapping track IDs to raw object label dictionaries.
    labels_av_track_ids : Dict[str, str]
        Mapping of autonomous vehicle (AV) names to their associated track IDs.
    _objects_in_global_coordinates : Optional[Dict[str, ObjectLabel]]
        Cached dictionary mapping track IDs to `ObjectLabel` instances in global coordinates.
    """

    def __init__(self, objects: Dict[str, dict], labels_av_track_ids: Dict[str, str]):
        """
        Initialize a `Labels` instance.

        Parameters
        ----------
        objects : Dict[str, dict]
            Dictionary mapping track IDs to object label data.
        labels_av_track_ids : Dict[str, str]
            Dictionary mapping AV (autonomous vehicle) names to their label track IDs.
        """
        self.objects: Dict[str, dict] = objects
        self.labels_av_track_ids: Dict[str, str] = labels_av_track_ids

        self._objects_in_global_coordinates: Optional[Dict[str, ObjectLabel]] = None

    def objects_in_global_coordinates(self, remove_av_labels: bool = False) -> Dict[str, ObjectLabel]:
        """Return labeled objects in global coordinates.

        This method converts raw label dictionaries into `ObjectLabel` instances (once cached),
        and optionally removes labels corresponding to autonomous vehicles.

        Parameters
        ----------
        remove_av_labels : bool, default=False
            Whether to exclude labels belonging to AVs.

        Returns
        -------
        Dict[str, ObjectLabel]
            Dictionary mapping track IDs to `ObjectLabel` instances in global coordinates.
        """
        if self._objects_in_global_coordinates is None:
            self._objects_in_global_coordinates = {
                track_id: ObjectLabel(**obj) for track_id, obj in self.objects.items()
            }

        if remove_av_labels:
            objects_copy = self._objects_in_global_coordinates.copy()
            [objects_copy.pop(k) for k in self.labels_av_track_ids.values() if k in objects_copy]
            return objects_copy
        else:
            return self._objects_in_global_coordinates

    def objects_in_this_coordinates(
        self, origin: str, remove_av_labels: bool = False
    ) -> Dict[str, ObjectLabel]:
        """Transform labeled objects to a specified coordinate system.

        Parameters
        ----------
        origin : str
            The target coordinate frame (must exist in the global transformation registry).
        remove_av_labels : bool, default=False
            Whether to exclude labels belonging to AVs.

        Returns
        -------
        Dict[str, ObjectLabel]
            Dictionary mapping track IDs to transformed `ObjectLabel` instances.
        """
        return_objs: Dict[str, ObjectLabel] = {}
        for track_id, obj in self.objects_in_global_coordinates(remove_av_labels).items():
            return_objs[track_id] = obj.convert_to_this_coordinates(origin)
        return return_objs

    def objects_in_vehicle_body_coordinates(
        self, vehicle: Vehicle, remove_av_label: bool = False
    ) -> Dict[str, ObjectLabel]:
        """Return labeled objects in a vehicle's body coordinate frame.

        Parameters
        ----------
        vehicle : Vehicle
            The vehicle whose body coordinate system is used.
        remove_av_label : bool, default=False
            Whether to remove the label associated with this autonomous vehicle.

        Returns
        -------
        Dict[str, ObjectLabel]
            Dictionary of objects transformed into the vehicle's body coordinates.
        """
        objects = self.objects_in_this_coordinates(vehicle.BODY)
        if remove_av_label and vehicle.vehicle_name in self.labels_av_track_ids:
            objects.pop(self.labels_av_track_ids[vehicle.vehicle_name], None)
        return objects

    def objects_in_vehicle_horizontal_coordinates(
        self, vehicle: Vehicle, remove_av_label: bool = False
    ) -> Dict[str, ObjectLabel]:
        """Return labeled objects in a vehicle's horizontal coordinate frame.

        Parameters
        ----------
        vehicle : Vehicle
            The vehicle whose horizontal coordinate system is used.
        remove_av_label : bool, default=False
            Whether to remove the label associated with this autonomous vehicle.

        Returns
        -------
        Dict[str, ObjectLabel]
            Dictionary of objects transformed into the vehicle's horizontal coordinates.
        """
        objects = self.objects_in_this_coordinates(vehicle.HORIZONTAL)
        if remove_av_label and vehicle.vehicle_name in self.labels_av_track_ids:
            objects.pop(self.labels_av_track_ids[vehicle.vehicle_name], None)
        return objects

    def objects_in_infrastructure_coordinates(
        self, infra: Infrastructure, remove_av_labels: bool = False
    ) -> Dict[str, ObjectLabel]:
        """Return labeled objects in an infrastructure's coordinate frame.

        Parameters
        ----------
        infra : Infrastructure
            The infrastructure whose coordinate frame is used.
        remove_av_labels : bool, default=False
            Whether to exclude labels belonging to AVs.

        Returns
        -------
        Dict[str, ObjectLabel]
            Dictionary of objects transformed into the infrastructure's coordinates.
        """
        return self.objects_in_this_coordinates(infra.INFRASTRUCTURE, remove_av_labels)

    def objects_in_lidar_coordinates(
        self, lidar: LidarData, remove_av_labels: bool = False
    ) -> Dict[str, ObjectLabel]:
        """Return labeled objects in a LiDAR sensor's coordinate frame.

        Parameters
        ----------
        lidar : LidarData
            The LiDAR sensor whose coordinate frame is used.
        remove_av_labels : bool, default=False
            Whether to exclude labels belonging to AVs.

        Returns
        -------
        Dict[str, ObjectLabel]
            Dictionary of objects transformed into the LiDAR's coordinates.
        """
        return self.objects_in_this_coordinates(lidar.LIDAR, remove_av_labels)

    def objects_in_camera_coordinates(
        self, camera: CameraData, remove_av_labels: bool = False
    ) -> Dict[str, ObjectLabel]:
        """Return labeled objects in a camera's coordinate frame.

        Parameters
        ----------
        camera : CameraData
            The camera whose coordinate frame is used.
        remove_av_labels : bool, default=False
            Whether to exclude labels belonging to AVs.

        Returns
        -------
        Dict[str, ObjectLabel]
            Dictionary of objects transformed into the camera's coordinates.
        """
        return self.objects_in_this_coordinates(camera.CAMERA, remove_av_labels)

    def objects_in_image_coordinates(
        self, camera: CameraData, remove_av_labels: bool = False
    ) -> Dict[str, ObjectLabel]:
        """Project labeled objects from 3D world coordinates to 2D image coordinates.

        Parameters
        ----------
        camera : CameraData
            The camera used to perform the projection.
        remove_av_labels : bool, default=False
            Whether to exclude labels belonging to AVs.

        Returns
        -------
        Dict[str, ObjectLabel]
            Dictionary of objects transformed into the camera's coordinates and projected in image coordinates.
        """
        return_objs: Dict[str, ObjectLabel] = {}
        for track_id, obj in self.objects_in_global_coordinates(remove_av_labels).items():
            return_objs[track_id] = obj.convert_to_image_coordinates(camera)
        return return_objs