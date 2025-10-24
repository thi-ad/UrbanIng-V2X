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

import numpy as np
import cv2
from typing import Callable, Tuple, Optional, Dict

from .registry import _xTg_registry


class CameraData:
    """
    Container class for camera-related data, including intrinsic/extrinsic transformations,
    image data, and coordinate conversions.

    This class handles:
    - Registration of camera and image coordinate transformations.
    - Lazy loading of distorted and undistorted images.
    - Conversion of 3D points in camera coordinates to 2D image pixel coordinates.

    Parameters
    ----------
    camera_name : str
        Name identifier for the camera (e.g., 'front', 'rear', 'left').
    image_path : str
        Filesystem path to the distorted image captured by the camera.
    cTg : np.ndarray
        4×4 transformation matrix from the global frame to the camera frame.
    iTc : np.ndarray
        3×3 camera intrinsic matrix mapping camera coordinates to image coordinates.
    image_size : np.ndarray or tuple of int
        Size of the image in (height, width) format.
    undistort_function : Callable[[np.ndarray], np.ndarray]
        Function that takes a distorted image and returns an undistorted version.

    Attributes
    ----------
    camera_name : str
        Name of the camera.
    image_path : str
        Path to the image file.
    cTg : np.ndarray
        4×4 transformation matrix from global to camera coordinates.
    iTc : np.ndarray
        3×3 intrinsic camera matrix.
    image_size : np.ndarray
        Numpy array of (height, width).
    image_height : int
        Image height in pixels.
    image_width : int
        Image width in pixels.
    undistort_function : Callable[[np.ndarray], np.ndarray]
        Function to perform undistortion.
    CAMERA : str
        Registry key name for camera coordinates.
    IMAGE : str
        Registry key name for image coordinates.
    _distorted_image : Optional[np.ndarray]
        Cached distorted image (loaded lazily).
    _undistorted_image : Optional[np.ndarray]
        Cached undistorted image (computed lazily).
    """

    def __init__(
        self,
        camera_name: str,
        image_path: str,
        cTg: np.ndarray,
        iTc: np.ndarray,
        image_size: Tuple[int, int],
        undistort_function: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self.camera_name = camera_name
        self.image_path = image_path
        self.cTg = cTg
        self.iTc = iTc
        self.image_size = np.array(image_size)
        self.undistort_function = undistort_function

        self.image_height, self.image_width = image_size
        self._distorted_image: Optional[np.ndarray] = None
        self._undistorted_image: Optional[np.ndarray] = None

        # Embed intrinsic matrix into homogeneous 4×4 form
        iTc_eye = np.eye(4)
        iTc_eye[:3, :3] = iTc

        # Register coordinate transforms
        self.CAMERA = f"{self.camera_name}_camera_coordinates"
        self.IMAGE = f"{self.camera_name}_image_coordinates"
        _xTg_registry[self.CAMERA] = self.cTg
        _xTg_registry[self.IMAGE] = iTc_eye @ self.cTg

    def add_coordinates(self, coordinate_name: str, x_T_CAMERA: np.ndarray) -> None:
        """
        Add a new coordinate frame relative to the current camera coordinate system
        and register it globally in `_xTg_registry`.

        Parameters
        ----------
        coordinate_name : str
            Descriptive name for the new coordinate frame.
        x_T_CAMERA : np.ndarray
            4×4 transformation matrix from the camera frame to the new coordinate frame.
        """
        eval(f"self.{coordinate_name} = {self.camera_name}_{coordinate_name}_camera_coordinates")
        _xTg_registry[eval(f"self.{coordinate_name}")] = x_T_CAMERA @ _xTg_registry[self.CAMERA]

    @property
    def distorted_image(self) -> np.ndarray:
        """
        Load and return the distorted image.

        Returns
        -------
        np.ndarray
            Distorted image loaded from disk in BGR format.
        """
        if self._distorted_image is None:
            self._distorted_image = cv2.imread(self.image_path)
        return self._distorted_image

    @property
    def undistorted_image(self) -> np.ndarray:
        """
        Compute (once) and return the undistorted image.

        Returns
        -------
        np.ndarray
            Undistorted image obtained via `undistort_function`.
        """
        if self._undistorted_image is None:
            self._undistorted_image = self.undistort_function(self.distorted_image)
        return self._undistorted_image

    def camera_coordinates_to_image_coordinates(
        self, points_in_camera_coordinates: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Project 3D points from camera coordinates to 2D image coordinates.

        Parameters
        ----------
        points_in_camera_coordinates : np.ndarray
            Array of 3D points in the camera coordinate system with shape (..., 4) or (..., 3).
            The homogeneous component (if any) is ignored.

        Returns
        -------
        image_points_out : np.ndarray
            2D integer pixel coordinates of the projected points, shape (..., 2).
        valid_flag : np.ndarray
            Boolean mask indicating which points are in front of the camera (z > 0).
        inside_image_flag : np.ndarray
            Boolean mask indicating which projected points fall inside image boundaries.
        """
        image_points_unnorm = points_in_camera_coordinates[..., :3] @ self.iTc.T
        image_points_out = (image_points_unnorm[..., :2] / np.abs(image_points_unnorm[..., 2:3])).astype(int)

        valid_flag = image_points_unnorm[..., 2] > 0
        xs, ys = image_points_out[..., 0], image_points_out[..., 1]
        inside_image_flag = (xs > 0) & (xs < self.image_width) & (ys > 0) & (ys < self.image_height)

        return image_points_out, valid_flag, inside_image_flag