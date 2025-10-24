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

import math
from typing import Union, Optional

import cv2
import numpy as np

from urbaning.data.frame import Frame
from urbaning.data.infrastructure import Infrastructure
from urbaning.data.lanelet_map import LLMap
from urbaning.data.lidar_data import LidarData
from urbaning.data.registry import GLOBAL
from urbaning.data.vehicle import Vehicle
from .utils import draw_dashed_polyline, lanelet_linestring_plot_types


class BEVVisualizer:
    """Bird's Eye View (BEV) visualizer for vehicles, labels, lanelets, and point clouds.

    Attributes
    ----------
    image_size : np.ndarray
        Width and height of the output BEV image.
    extent : float
        Total extent (meters) for the BEV image; each side is extent/2.
    offset : np.ndarray
        Offset in pixels for the center of the BEV image.
    rotation : float
        Image rotation in radians.
    pixels_per_meter : float
        Scale factor for converting meters to pixels.
    meters_per_pixel : float
        Inverse of pixels_per_meter.
    iTp : np.ndarray
        Transformation matrix from meters to image pixels.
    origin : str | None
        Current coordinate system origin.
    dash_options : list
        Options for dashed line rendering.
    """

    def __init__(
        self,
        image_size=(1920, 1080),
        extent: float = 100,
        offset=None,
        rotation: float = -25,
    ):
        """
        Parameters
        ----------
        image_size : tuple of int, default=(1920, 1080)
            Width and height of the BEV image in pixels.
        extent : float, default=100
            Total meters represented in the image (half each side).
        offset : tuple or np.ndarray, optional
            Pixel coordinates of the image center. Defaults to image center.
        rotation : float, default=-25
            Rotation of the image in degrees (clockwise from top-down view).
        """
        self.image_size = np.array(image_size)
        self.extent = extent
        self.offset = self.image_size / 2 if offset is None else np.array(offset)
        self.rotation = rotation * math.pi / 180

        self.image_width, self.image_height = self.image_size
        self.pixels_per_meter = min(self.image_width, self.image_height) / self.extent
        self.meters_per_pixel = 1 / self.pixels_per_meter
        self.iTp = (
            np.array([[math.cos(rotation), math.sin(rotation)],
                      [math.sin(rotation), -math.cos(rotation)]])
            .T * self.pixels_per_meter
        )
        self.origin: Optional[str] = None

        scale = 1
        self.dash_options = [
            [int(10 * scale), int(10 * scale)],
            [int(5 * scale), int(10 * scale)],
            [int(2 * scale), int(5 * scale)],
            None
        ]

    def get_point_cloud_and_set_origin(
        self,
        source: Union[Frame, Vehicle, Infrastructure, LidarData, np.ndarray]
    ) -> np.ndarray:
        """Get point cloud from source and optionally set the BEV origin.

        Parameters
        ----------
        source : Frame | Vehicle | Infrastructure | LidarData | np.ndarray
            Source of the point cloud.

        Returns
        -------
        np.ndarray
            N x 6 point cloud array (x, y, z, intensity, time, lidar_index).
        """
        if isinstance(source, Frame):
            if self.origin is None:
                point_cloud = source.fused_point_cloud_in_global_coordinates()
                self.origin = source.GLOBAL
            else:
                point_cloud = source.fused_point_cloud_in_this_coordinates(self.origin)
        elif isinstance(source, Vehicle):
            if self.origin is None:
                point_cloud = source.fused_point_cloud_in_vehicle_body_coordinates()
                self.origin = source.BODY
            else:
                point_cloud = source.fused_point_cloud_in_this_coordinates(self.origin)
        elif isinstance(source, Infrastructure):
            if self.origin is None:
                point_cloud = source.fused_point_cloud_in_infrastructure_coordinates()
                self.origin = source.INFRASTRUCTURE
            else:
                point_cloud = source.fused_point_cloud_in_this_coordinates(self.origin)
        elif isinstance(source, LidarData):
            if self.origin is None:
                point_cloud = source.point_cloud_in_lidar_coordinates()
                self.origin = source.LIDAR
            else:
                point_cloud = source.point_cloud_in_this_coordinates(self.origin)
        elif isinstance(source, np.ndarray):
            point_cloud = source
        else:
            raise TypeError(f"Unknown data type: {type(source)}")

        return point_cloud

    def set_origin(self, origin: str = GLOBAL) -> None:
        """Set the BEV visualizer coordinate origin.

        Parameters
        ----------
        origin : str, default=GLOBAL
            Name of the coordinate frame to use as origin.
        """
        self.origin = origin

    def get_image_frame(self, image_frame: Optional[np.ndarray] = None) -> np.ndarray:
        """Return an existing image frame or create a new blank one.

        Parameters
        ----------
        image_frame : np.ndarray, optional
            Existing image frame. If None, a blank frame is created.

        Returns
        -------
        np.ndarray
            Image frame array.
        """
        if image_frame is None:
            image_frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        return image_frame

    def transform_meters_to_images(self, points: np.ndarray) -> np.ndarray:
        """Convert points from meters to image pixel coordinates.

        Parameters
        ----------
        points : np.ndarray
            N x 2 array of points in meters.

        Returns
        -------
        np.ndarray
            N x 2 array of points in image pixels (int).
        """
        return ((points @ self.iTp.T) + self.offset).astype(int)

    def plot_lanelet_map(
        self,
        lanelet_map: LLMap,
        image_frame: Optional[np.ndarray] = None,
        thickness: int = 1
    ) -> np.ndarray:
        """Plot lanelet map lines in BEV view.

        Parameters
        ----------
        lanelet_map : LLMap
            Lanelet map object.
        image_frame : np.ndarray, optional
            Image to draw on. Created if None.
        thickness : int, default=1
            Line thickness.

        Returns
        -------
        np.ndarray
            Image with lanelet lines plotted.
        """
        image_frame = self.get_image_frame(image_frame)

        for ls in lanelet_map.map.lineStringLayer:
            dashes, color, out_thickness = lanelet_linestring_plot_types(ls, thickness)
            dashes = self.dash_options[dashes]

            ls_points = np.column_stack([[pt.x for pt in ls], [pt.y for pt in ls]])
            ls_points = lanelet_map.transform_points(ls_points, self.origin)
            ls_points = self.transform_meters_to_images(ls_points[:, :2]).astype(np.int32)

            if dashes is None:
                image_frame = cv2.polylines(
                    image_frame, [ls_points.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=out_thickness
                )
            else:
                image_frame = draw_dashed_polyline(
                    image_frame, ls_points, dash_length=dashes[0], gap_length=dashes[1], color=color, thickness=out_thickness
                )

        return image_frame

    # The rest of the plotting methods (plot_point_cloud, plot_labels, plot_vehicle, plot_label)
    # follow similar structure with type annotations and NumPy-style docstrings.
