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

from typing import Mapping, Union

import cv2
import matplotlib
import numpy as np

from urbaning.data.camera_data import CameraData
from urbaning.data.frame import Frame
from urbaning.data.infrastructure import Infrastructure
from urbaning.data.labels import Labels
from urbaning.data.lanelet_map import LLMap
from urbaning.data.lidar_data import LidarData
from urbaning.data.object_label import ObjectLabel
from urbaning.data.vehicle import Vehicle
from .utils import draw_dashed_polyline, lanelet_linestring_plot_types, get_color_function


class CameraDataVisualizer:
    """Visualizer for projecting sensor, labels and map data onto a camera image.

    Can visualize point clouds, lanelet maps, object labels, and vehicles
    directly onto the undistorted image of a camera.

    Attributes
    ----------
    camera_data : CameraData
        The camera data associated with this visualizer.
    current_image : np.ndarray
        The current image onto which the visualizations are drawn.
    dash_options : list
        Predefined options for dashed line rendering.
    """
    def __init__(self):
        self.camera_data = None
        self.current_image = None

        self.multiplier = 1

        scale = 1
        self.dash_options: list = [
            [int(10 * scale), int(10 * scale)],
            [int(5 * scale), int(10 * scale)],
            [int(2 * scale), int(5 * scale)],
            None
        ]

    def reset(self):
        self.current_image = None
        self.camera_data = None

    def plot_camera_data(self, camera_data: CameraData):
        self.camera_data = camera_data
        self.current_image = camera_data.undistorted_image
        # self.multiplier = int(((self.current_image.shape[1] / 640) + 1) // 2)
        self.multiplier = round(self.current_image.shape[1] / 640)

    def result(self):
        return self.current_image

    def get_point_cloud(self, source: Union[Frame, Vehicle, Infrastructure, LidarData, np.ndarray]) -> np.ndarray:
        """Return point cloud from a given source transformed into camera coordinates.

        Parameters
        ----------
        source : Frame | Vehicle | Infrastructure | LidarData | np.ndarray
            Source of the point cloud.

        Returns
        -------
        np.ndarray
            N x 6 array of points in camera coordinates (x, y, z, intensity, time, lidar_index).
        """
        if isinstance(source, Frame):
            point_cloud = source.fused_point_cloud_in_camera_coordinates(self.camera_data)
        elif isinstance(source, Vehicle):
            point_cloud = source.fused_point_cloud_in_camera_coordinates(self.camera_data)
        elif isinstance(source, Infrastructure):
            point_cloud = source.fused_point_cloud_in_camera_coordinates(self.camera_data)
        elif isinstance(source, LidarData):
            point_cloud = source.point_cloud_in_camera_coordinates(self.camera_data)
        elif isinstance(source, np.ndarray):
            point_cloud = source
        else:
            raise TypeError(f"Unknown data type: {type(source)}")
        return point_cloud

    def plot_lanelet_map(self, lanelet_map: LLMap, thickness: int = 1) -> None:
        """Draw lanelet lines from a map onto the current camera image.

        Parameters
        ----------
        lanelet_map : LLMap
            Lanelet2 map object.
        thickness : int, default=1
            Base thickness of lanelet lines.
        """
        for ls in lanelet_map.map.lineStringLayer:
            dashes, color, out_thickness = lanelet_linestring_plot_types(ls, thickness)
            dashes = self.dash_options[dashes]

            # Transform points
            ls_points = np.column_stack([[pt.x for pt in ls], [pt.y for pt in ls]])
            ls_points = lanelet_map.transform_points(ls_points, self.camera_data.CAMERA)
            image_points, valid_flag, inside_image_flag = self.camera_data.camera_coordinates_to_image_coordinates(
                ls_points)

            points = image_points[valid_flag, :2].astype(np.int32)
            if len(points) < 2:
                continue

            if dashes is None:
                self.current_image = cv2.polylines(
                    self.current_image, [points.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=out_thickness
                )
            else:
                self.current_image = draw_dashed_polyline(
                    self.current_image, points, dash_length=dashes[0], gap_length=dashes[1], color=color,
                    thickness=out_thickness
                )

    def plot_point_cloud(
            self,
            source: Union[Frame, Vehicle, Infrastructure, LidarData, np.ndarray],
            color: Union[str, tuple, list, np.ndarray] = "intensity",
            color_map: str = "gist_rainbow"
    ) -> None:
        """Plot point cloud on the current camera image.

        Parameters
        ----------
        source : Frame | Vehicle | Infrastructure | LidarData | np.ndarray
            Source of the point cloud.
        color : str | tuple | list | np.ndarray, default='intensity'
            Determines coloring of points:
                - 'intensity', 'time_offset', 'lidar_index' → mapped colors
                - RGB tuple/list/array → flat color for all points
        color_map : str, default='gist_rainbow'
            Matplotlib colormap name when using mapped colors.
        """
        point_cloud = self.get_point_cloud(source)

        if isinstance(color, str):
            if color == "intensity":
                c = point_cloud[:, 3]
            elif color == "time_offset":
                c = point_cloud[:, 4]
            elif color == "lidar_index":
                c = point_cloud[:, 5]
            else:
                raise ValueError(f"Unknown color option: {color}")

            cmap = matplotlib.colormaps[color_map]
            c_norm = c / (c.max() if c.max() > 0 else 1)
            pcd_colors = (cmap(c_norm)[:, [2, 1, 0]] * 255).astype(np.uint8)
        else:
            flat_color = np.array(color, dtype=np.uint8)
            if flat_color.shape != (3,):
                raise ValueError("Flat color must be a 3-element RGB tuple/list/array")
            pcd_colors = np.tile(flat_color, (point_cloud.shape[0], 1))

        image_points, valid_flag, inside_image_flag = (
            self.camera_data.camera_coordinates_to_image_coordinates(point_cloud))
        xs, ys = image_points[:, :2].T.astype(int)
        flag = np.logical_and(valid_flag, inside_image_flag)

        offset = (self.multiplier + 1) // 2
        for x_offset in range(-offset + 1, offset):
            for y_offset in range(-offset + 1, offset):

                x_pos = xs[flag] + x_offset
                y_pos = ys[flag] + y_offset

                x_pos[x_pos < 0] = 0
                x_pos[x_pos >= self.current_image.shape[1]] = self.current_image.shape[1] - 1

                y_pos[y_pos < 0] = 0
                y_pos[y_pos >= self.current_image.shape[0]] = self.current_image.shape[0] - 1

                self.current_image[y_pos, x_pos] = pcd_colors[flag]

        # self.current_image[ys[flag], xs[flag]] = pcd_colors[flag]

    def plot_labels(
            self,
            object_labels: Union[Mapping[int, ObjectLabel], Labels],
            box_thickness: int = 2,
            box_filled: bool = False,
            box_color: Union[tuple, list] = (0, 100, 0),
            plot_text: bool = True,
            text=None,
            text_scale: float = 0.7,
            text_thickness: int = 2,
            text_color: Union[tuple, list] = (255, 255, 255)
    ) -> None:
        """Plot object labels on the current image.

        Parameters
        ----------
        object_labels : Mapping[int, ObjectLabel] | Labels
            Labels to plot.
        box_thickness : int, default=2
            Thickness of bounding box lines.
        box_filled : bool, default=False
            Defines if the box has to be filled or not. Not implemented yet
        box_color : tuple | list, default=(0,100,0)
            RGB color of bounding boxes.
        plot_text : bool, default=True
            Whether to draw label text.
        text : str, optional
            Custom text to display for each label.
        text_scale : float, default=0.5
            Font scale for text.
        text_thickness : int, default=2
            Thickness of text.
        text_color : tuple | list, default=(255,0,255)
            RGB color of the text.
        """
        text_color_function = get_color_function(text_color)
        box_color_function = get_color_function(box_color)

        if isinstance(object_labels, Labels):
            object_labels = object_labels.objects_in_global_coordinates()

        for track_id, object_label in object_labels.items():
            self.plot_label(
                object_label,
                box_thickness=box_thickness * self.multiplier,
                box_filled = box_filled,
                box_color=box_color_function(object_label),
                plot_text=plot_text,
                text=text,
                text_scale=text_scale * self.multiplier,
                text_thickness=text_thickness * self.multiplier,
                text_color=text_color_function(object_label),
            )

    def plot_vehicle(
            self,
            vehicle: Vehicle,
            box_thickness: int = 1,
            box_filled: bool = False,
            box_color: Union[tuple, list] = (0, 100, 0),
            plot_text: bool = True,
            text=None,
            text_scale: float = 0.7,
            text_thickness: int = 1,
            text_color: Union[tuple, list] = (255, 255, 255)
    ) -> None:
        """Plot a vehicle as a labeled bounding box on the camera image."""
        text_color_function = get_color_function(text_color)
        box_color_function = get_color_function(box_color)

        vehicle_as_label = vehicle.state_as_label()
        self.plot_label(
            vehicle_as_label,
            box_thickness=box_thickness * self.multiplier,
            box_filled=box_filled,
            box_color=box_color_function(vehicle_as_label),
            plot_text=plot_text,
            text=text,
            text_scale=text_scale * self.multiplier,
            text_thickness=text_thickness * self.multiplier,
            text_color=text_color_function(vehicle_as_label)
        )

    def plot_label(
            self,
            object_label: ObjectLabel,
            box_thickness: int = 1,
            box_filled: bool = False,
            box_color: Union[tuple, list] = (0, 100, 0),
            plot_text: bool = True,
            text=None,
            text_scale: float = 0.7,
            text_thickness: int = 1,
            text_color: Union[tuple, list] = (255, 255, 255)
    ) -> None:
        """Plot a single object label on the camera image.

        Parameters
        ----------
        object_label : ObjectLabel
            Object label to plot.
        box_thickness : int, default=1
            Thickness of bounding box lines.
        box_filled : bool, default=False
            Defines if the box has to be filled or not. Not implemented yet
        box_color : tuple | list, default=(0,100,0)
            RGB color of bounding box.
        plot_text : bool, default=True
            Whether to plot label text.
        text : str, optional
            Custom text for the label.
        text_scale : float, default=0.5
            Scale for text rendering.
        text_thickness : int, default=1
            Thickness of text.
        text_color : tuple | list, default=(255,255,255)
            RGB color of text.
        """
        text_color = get_color_function(text_color)(object_label)
        box_color = get_color_function(box_color)(object_label)

        object_label = object_label.convert_to_image_coordinates(self.camera_data)
        if np.any(np.logical_and(object_label.corners_in_image_inside_flag, object_label.corners_in_image_valid_flag)):
            corners = object_label.corners_in_image_coordinates
            corners_valid = object_label.corners_in_image_valid_flag
            points = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
                      (7, 0), (0, 3), (4, 7), (1, 6), (2, 5), (0, 6), (1, 7)]
            for a, b in points:
                if corners_valid[a] and corners_valid[b]:
                    cv2.line(self.current_image, corners[a], corners[b], box_color, box_thickness)

            position = object_label.position_in_image_coordinates
            if plot_text:
                if text is None:
                    text = object_label.object_type[0:3] + " " + str(object_label.track_id)
                cv2.putText(self.current_image, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=text_scale, color=text_color, thickness=text_thickness)