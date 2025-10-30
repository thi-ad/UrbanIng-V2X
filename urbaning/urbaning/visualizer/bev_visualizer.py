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

import math
from typing import Union, Optional, Mapping

import cv2
import matplotlib
import numpy as np

from urbaning.data import Frame, Infrastructure, LLMap, LidarData, Vehicle, ObjectLabel, Labels
from urbaning.data.registry import GLOBAL
from .utils import draw_dashed_polyline, lanelet_linestring_plot_types, get_color_function


class BEVVisualizer:
    """Bird's Eye View (BEV) visualizer for labels, lanelets, and point clouds of any source.

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
        rotation: float = 0,
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
        rotation : float, default=0
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
        self.origin_hard_set = False
        self.current_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)

        scale = 1
        self.dash_options = [
            [int(10 * scale), int(10 * scale)],
            [int(5 * scale), int(10 * scale)],
            [int(2 * scale), int(5 * scale)],
            None
        ]

    def reset(self):
        self.current_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        if not self.origin_hard_set:
            self.origin = None

    def result(self):
        return self.current_image

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
        self.origin_hard_set = True

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
        thickness: int = 1
    ) -> None:
        """Plot lanelet map lines in BEV view.

        Parameters
        ----------
        lanelet_map : LLMap
            Lanelet map object.
        thickness : int, default=1
            Line thickness.

        """

        for ls in lanelet_map.map.lineStringLayer:
            dashes, color, out_thickness = lanelet_linestring_plot_types(ls, thickness)
            dashes = self.dash_options[dashes]

            ls_points = np.column_stack([[pt.x for pt in ls], [pt.y for pt in ls]])
            ls_points = lanelet_map.transform_points(ls_points, self.origin or GLOBAL)
            ls_points = self.transform_meters_to_images(ls_points[:, :2]).astype(np.int32)

            if dashes is None:
                self.current_image = cv2.polylines(
                    self.current_image, [ls_points.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=out_thickness
                )
            else:
                self.current_image = draw_dashed_polyline(
                    self.current_image, ls_points, dash_length=dashes[0], gap_length=dashes[1], color=color, thickness=out_thickness
                )

    def plot_point_cloud(self, source: Union[Frame, Vehicle, Infrastructure, LidarData, np.ndarray],
                         color='intensity', color_map='gist_rainbow'):
        """
        Plot the point cloud onto the lidar frame.

        Args:
            source (Frame, Vehicle, Infrastructure, LidarData): source for point cloud and origin
            color (str | tuple | list | np.ndarray):
                - 'intensity', 'time_offset', or 'lidar_index' → use colormap.
                - (R, G, B) tuple or list → flat color for all points.
            color_map (str): Colormap name if using mapped colors.
        """

        point_cloud = self.get_point_cloud_and_set_origin(source)

        if isinstance(color, str):
            # --- Colormap mode ---
            if color == 'intensity':
                c = point_cloud[:, 3]
            elif color == 'time_offset':
                c = point_cloud[:, 4]
            elif color == 'lidar_index':
                c = point_cloud[:, 5]
            else:
                raise ValueError(f"Unknown color option: {color}")

            cmap = matplotlib.colormaps[color_map]
            c_norm = c / (c.max() if c.max() > 0 else 1)  # avoid div by zero
            pcd_colors = (cmap(c_norm)[:, [2, 1, 0]] * 255).astype(np.uint8)

        else:
            # --- Flat color mode ---
            flat_color = np.array(color, dtype=np.uint8)
            if flat_color.shape != (3,):
                raise ValueError("Flat color must be a 3-element RGB tuple/list/array")
            pcd_colors = np.tile(flat_color, (point_cloud.shape[0], 1))

        # xs, ys = ((point_cloud[:, :2] @ self.iTp.T)  + self.offset).astype(int).T
        xs, ys = self.transform_meters_to_images(point_cloud[:, :2]).T
        flag = (xs > 0) & (xs < self.image_width - 1) & (ys > 0) & (ys < self.image_height - 1)

        self.current_image[ys[flag], xs[flag]] = pcd_colors[flag]

    def plot_labels(
            self,
            object_labels:Union[Mapping[int, ObjectLabel], Labels],
            box_thickness=4,
            box_filled=False,
            box_color=(0, 100, 0),
            plot_text=True,
            text=None,
            text_scale=0.6,
            text_thickness=2,
            text_color=(255, 255, 255),
    ):
        text_color_function = get_color_function(text_color)
        box_color_function = get_color_function(box_color)

        if isinstance(object_labels, Labels):
            object_labels = object_labels.objects_in_global_coordinates()

        for track_id, object_label in object_labels.items():
            self.plot_label(
                object_label,
                box_thickness=box_thickness, box_filled=box_filled, box_color=box_color_function(object_label),
                plot_text=plot_text, text=text, text_scale=text_scale, text_thickness=text_thickness, text_color=text_color_function(object_label),
            )

    def plot_vehicle(
            self,
            vehicle:Vehicle,
            box_thickness=4,
            box_filled=True,
            box_color=(0, 100, 0),
            plot_text=True,
            text=None,
            text_scale=0.6,
            text_thickness=2,
            text_color=(255, 255, 255),
    ):
        text_color_function = get_color_function(text_color)
        box_color_function = get_color_function(box_color)

        vehicle_as_label = vehicle.state_as_label()
        self.plot_label(vehicle_as_label,
                        box_thickness=box_thickness, box_filled=box_filled, box_color=box_color_function(vehicle_as_label),
                        plot_text=plot_text, text=text, text_scale=text_scale, text_thickness=text_thickness, text_color=text_color_function(vehicle_as_label))

    def plot_label(
            self, object_label:ObjectLabel,
            box_thickness=4,
            box_filled=True,
            box_color=(0, 100, 0),
            plot_text=True,
            text=None,
            text_scale=0.6,
            text_thickness=2,
            text_color=(255, 255, 255),
    ):
        text_color = get_color_function(text_color)(object_label)
        box_color = get_color_function(box_color)(object_label)


        if self.origin is not None:
            object_label = object_label.convert_to_this_coordinates(self.origin)

        corners = object_label.corners_in_current_coordinates()[4:, :2]
        pos = object_label.position[None, :2]  # 1 x 2
        arrow_len = object_label.dimension[0] * 0.5 + 15 * self.meters_per_pixel
        arrow_tip = object_label.transform_points_from_object_coordinates_to_current_coordinates(
            np.array([[arrow_len], [0], [0]])).T[:, :2]  # 1 x 2

        all_points = np.concatenate([corners, pos, arrow_tip], axis=0)
        all_points = self.transform_meters_to_images(all_points)
        # all_points = ((all_points @ self.iTp.T) + self.offset).astype(int)

        if box_filled:
            self.current_image = cv2.fillPoly(self.current_image, [all_points[:4].reshape(4, 1, 2)], color=box_color)
        else:
            self.current_image = cv2.polylines(self.current_image, [all_points[:4].reshape(4, 1, 2)], isClosed=True, color=box_color, thickness=box_thickness)

        self.current_image = cv2.line(self.current_image, all_points[4], all_points[5], color=box_color, thickness=box_thickness)

        if plot_text:
            if text is None:
                text = object_label.object_type[0:3] + ' ' + str(object_label.track_id)
            cv2.putText(self.current_image, text, all_points[4], cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, color=text_color, thickness=text_thickness, )
