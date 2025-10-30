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

import json
import numpy as np
from scipy.spatial.transform import Rotation
from typing import Mapping


class VehicleState:
    """Represents the state of a vehicle at a given timestamp.

    Stores pose, velocity, acceleration, rotation rates, and other geometric
    parameters of the vehicle. Provides convenient properties for position,
    orientation (quaternion), and transformations to/from body and horizontal frames.

    Attributes
    ----------
    rotation_rates : np.ndarray
        Angular velocity (roll, pitch, yaw rates) of the vehicle in the global frame.
    acceleration : np.ndarray
        Linear acceleration of the vehicle in the global frame.
    velocity : np.ndarray
        Linear velocity of the vehicle in the global frame.
    gTv_body : np.ndarray
        4x4 homogeneous transformation matrix from vehicle body to global frame.
    dimension : np.ndarray
        Vehicle dimensions [length, width, height].
    dimension_with_mirror : np.ndarray
        Vehicle dimensions including mirrors.
    dx_frontaxle_rearaxle : float
        Distance between front and rear axles.
    dx_center_rearaxle : float
        Distance between vehicle center and rear axle.
    vT_CenterInFloor : np.ndarray
        4x4 transformation from vehicle floor to center.
    vT_RearAxleCenterInFloor : np.ndarray
        4x4 transformation from floor to rear axle center.
    vT_FrontAxleCenterInFloor : np.ndarray
        4x4 transformation from floor to front axle center.
    """

    def __init__(self, state_file_path: str, vehicle_data: Mapping):
        """
        Initialize VehicleState from a JSON file and vehicle metadata.

        Parameters
        ----------
        state_file_path : str
            Path to the JSON file containing vehicle state data.
        vehicle_data : Mapping
            Dictionary with vehicle-specific parameters such as dimensions and axle distances.
        """
        with open(state_file_path, "r") as f:
            state_data = json.load(f)

        self.rotation_rates: np.ndarray = np.array(state_data["W"])
        self.acceleration: np.ndarray = np.array(state_data["vA"])
        self.velocity: np.ndarray = np.array(state_data["vV"])
        self.gTv_body: np.ndarray = np.asarray(state_data["gTv"])

        self.dimension: np.ndarray = vehicle_data["size"]
        self.dimension_with_mirror: np.ndarray = vehicle_data["size_with_mirror"]
        self.dx_frontaxle_rearaxle: float = vehicle_data["dx_frontaxle_rearaxle"]
        self.dx_center_rearaxle: float = vehicle_data["dx_center_rearaxle"]
        self.vT_CenterInFloor: np.ndarray = vehicle_data["vT_CenterInFloor"]
        self.vT_RearAxleCenterInFloor: np.ndarray = vehicle_data["vT_RearAxleCenterInFloor"]
        self.vT_FrontAxleCenterInFloor: np.ndarray = vehicle_data["vT_FrontAxleCenterInFloor"]

        self._vTg_body: np.ndarray | None = None
        self._gTv_horizontal: np.ndarray | None = None
        self._vTg_horizontal: np.ndarray | None = None

        self._position: np.ndarray | None = None
        self._quaternion: np.ndarray | None = None

    @property
    def position(self) -> np.ndarray:
        """Vehicle position in global coordinates (3D vector)."""
        if self._position is None:
            self._position = self.gTv_body[:3, 3]
        return self._position

    @property
    def quaternion(self) -> np.ndarray:
        """Vehicle orientation as a quaternion [x, y, z, w]."""
        if self._quaternion is None:
            self._quaternion = Rotation.from_matrix(self.gTv_body[:3, :3]).as_quat()
        return self._quaternion

    @property
    def gTv_horizontal(self) -> np.ndarray:
        """4x4 transformation matrix from vehicle horizontal frame to global frame."""
        if self._gTv_horizontal is None:
            gTv_horizontal = self.gTv_body.copy()
            yaw, pitch, roll = Rotation.from_matrix(self.gTv_body[:3, :3]).as_euler("ZYX")
            gTv_horizontal[:3, :3] = Rotation.from_euler("Z", [yaw]).as_matrix()
            self._gTv_horizontal = gTv_horizontal
        return self._gTv_horizontal

    @property
    def vTg_horizontal(self) -> np.ndarray:
        """4x4 transformation matrix from global frame to vehicle horizontal frame."""
        if self._vTg_horizontal is None:
            self._vTg_horizontal = np.linalg.inv(self.gTv_horizontal)
        return self._vTg_horizontal

    @property
    def vTg_body(self) -> np.ndarray:
        """4x4 transformation matrix from global frame to vehicle body frame."""
        if self._vTg_body is None:
            self._vTg_body = np.linalg.inv(self.gTv_body)
        return self._vTg_body
