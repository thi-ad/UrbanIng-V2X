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

import os
import json
import numpy as np
import pandas as pd

from .frame import Frame
from .vehicle import Vehicle
from .infrastructure import Infrastructure
from .camera_data import CameraData
from .lidar_data import LidarData
from .labels import Labels
from .vehicle_state import VehicleState
from .lanelet_map import LLMap

from .registry import _xTg_registry

from .utils import (
    convert_labels_from_tracks_wise_to_timestamp_wise,
    redo_calib,
    redo_av_vehicle_data,
    get_wTg,
)

from .info import (
    lidar_index_mapping,
    vehicle_states_id_mapping,
    ground_params,
    gps_origins
)

class Sequence:
    """Represents a dataset sequence, providing frame-level access and sensor data parsing.

    The `Sequence` class handles loading all relevant data (LiDAR, cameras, vehicle and
    infrastructure states, labels, and calibration) for a given sequence. It allows
    iteration over timestamped frames, each represented as a `Frame` object.

    Attributes
    ----------
    sequence_root : str
        Path to the root folder of the sequence data.
    label_file : str
        Path to the JSON file containing labeled objects for this sequence.
    sequence_name : str
        Name of the sequence.
    calib_data : dict
        Calibration data for all sensors in the sequence.
    time_sync_df : pd.DataFrame
        DataFrame containing time-synchronized timestamps for all sensors.
    labels_av_track_ids : dict
        Mapping from autonomous vehicle (AV) vehicle names to track IDs in labels.
    av_vehicle_data : dict
        Preprocessed AV vehicle data loaded from JSON.
    labels : dict
        Labels converted from track-wise to timestamp-wise format.
    WORLD : str
        Name of the global world coordinate frame for this sequence.
    lanelet_map : LLMap
        Lanelet2 map object representing the environment.
    """

    def __init__(self, root_folder: str, sequence_name: str):
        """
        Initialize a Sequence instance by loading all sensor, calibration, and label data.

        Parameters
        ----------
        root_folder : str
            Root directory containing the dataset and supporting files.
        sequence_name : str
            Name of the sequence to load.
        """
        self.sequence_root = os.path.join(root_folder, "dataset", sequence_name)
        self.label_file = os.path.join(root_folder, "labels", sequence_name + ".json")
        self.sequence_name = sequence_name

        calibration_file = os.path.join(self.sequence_root, "calibration.json")
        time_sync_info_file = os.path.join(self.sequence_root, "timesync_info.csv")
        labels_av_track_ids_file = os.path.join(root_folder, "labels_av_track_ids.json")
        av_vehicle_data_file = os.path.join(root_folder, "av_vehicle_data.json")
        lanelet_map = os.path.join(root_folder, "crossings_lanelet2map.osm")

        # Load and preprocess calibration
        with open(calibration_file, "r") as f:
            calib_data = json.load(f)
        self.calib_data = redo_calib(calib_data)

        # Load time synchronization info
        self.time_sync_df = pd.read_csv(time_sync_info_file).set_index("Unnamed: 0")

        # Load label track IDs for AV vehicles
        with open(labels_av_track_ids_file, "r") as f:
            self.labels_av_track_ids = json.load(f)[sequence_name]

        # Load and preprocess AV vehicle data
        with open(av_vehicle_data_file, "r") as f:
            self.av_vehicle_data = redo_av_vehicle_data(json.load(f))

        # Load labels and convert to timestamp-wise format
        with open(self.label_file, "r") as f:
            labels = json.load(f)
        self.labels = convert_labels_from_tracks_wise_to_timestamp_wise(labels)

        # Initialize world frame for this sequence
        crossing_name = sequence_name.split("_")[2]
        self.WORLD = "world_coordinates"
        _xTg_registry[self.WORLD] = get_wTg(crossing_name)

        # Load Lanelet2 map
        self.lanelet_map = LLMap(
            lanelet_map, gps_origins[crossing_name], ground_params[crossing_name]
        )

    def __len__(self) -> int:
        """Return the number of time steps (columns) in the sequence."""
        return len(self.time_sync_df.columns)

    def __getitem__(self, item: int) -> Frame:
        """
        Get a `Frame` object for the specified index.

        Parameters
        ----------
        item : int
            Index of the frame to retrieve.

        Returns
        -------
        Frame
            Frame object containing all vehicle, infrastructure, and sensor data for the timestamp.
        """
        this_step_info = self.time_sync_df.iloc[:, item].to_dict()
        frame = self.extract_one_timestamp_data(this_step_info)
        return frame

    def __iter__(self):
        """Iterate over all frames in the sequence."""
        for i in range(len(self)):
            yield self[i]

    def extract_one_timestamp_data(self, this_step_info: dict) -> Frame:
        """
        Parse all sensor and label data for a single timestamp and return a `Frame`.

        Parameters
        ----------
        this_step_info : dict
            Dictionary containing sensor file paths and metadata for this timestamp.

        Returns
        -------
        Frame
            Frame object containing all loaded sensor data, vehicle states, and labels.
        """
        timestamp = int(this_step_info.pop("timestamp_ms")) / 1000
        frame = Frame(timestamp)

        # Set labels for this timestamp
        if timestamp in self.labels:
            frame.set_labels(Labels(self.labels[timestamp], self.labels_av_track_ids))

        # Parse vehicle and infrastructure data
        for k, v in this_step_info.items():
            # ---------------- Vehicle Sensors ----------------
            if k.startswith("vehicle"):
                vehicle_name = k.split("_")[0]
                vehicle_state_name = vehicle_name + "_state"

                if vehicle_name not in frame.vehicles:
                    frame.vehicles[vehicle_name] = Vehicle(vehicle_name, vehicle_states_id_mapping[vehicle_state_name])
                vehicle = frame.vehicles[vehicle_name]

                if k.endswith("camera"):
                    with open(
                        os.path.join(
                            self.sequence_root,
                            vehicle_state_name,
                            str((int(v.split(".")[0]) // 10) * 10) + ".json",
                        ),
                        "r",
                    ) as f:
                        vehicle_state = json.load(f)
                    gTv = np.asarray(vehicle_state["gTv"])
                    cTv = self.calib_data[k]["extrinsics"]["cTv"]
                    cTg = cTv.dot(np.linalg.inv(gTv))
                    iTc = self.calib_data[k]["intrinsics"]["IntrinsicMatrixNew"]
                    undistort_function = self.calib_data[k]["intrinsics"]["undistort_function"]
                    image_size = self.calib_data[k]["intrinsics"]["ImageSize"]
                    camera_data = CameraData(
                        k,
                        os.path.join(self.sequence_root, k, v),
                        cTg,
                        iTc,
                        image_size,
                        undistort_function,
                    )
                    vehicle.add_camera(k, camera_data)

                elif k.endswith("lidar"):
                    with open(
                        os.path.join(
                            self.sequence_root,
                            vehicle_state_name,
                            str((int(v.split(".")[0]) // 10) * 10) + ".json",
                        ),
                        "r",
                    ) as f:
                        vehicle_state = json.load(f)
                    gTv = np.asarray(vehicle_state["gTv"])
                    vTl = self.calib_data[k]["extrinsics"]["vTl"]
                    gTl = gTv.dot(vTl)
                    lidar_data = LidarData(
                        k, os.path.join(self.sequence_root, k, v), gTl, lidar_index_mapping[k]
                    )
                    vehicle.add_lidar(k, lidar_data)

                elif k.endswith("state"):
                    state_path = os.path.join(self.sequence_root, k, v)
                    vehicle_state = VehicleState(state_path, self.av_vehicle_data[vehicle.vehicle_name])
                    vehicle.add_state(vehicle_state)

            # ---------------- Infrastructure Sensors ----------------
            elif k.startswith("crossing"):
                infra_name = k.split("_")[0]
                if infra_name not in frame.infrastructures:
                    frame.infrastructures[infra_name] = Infrastructure(infra_name, vehicle_states_id_mapping[infra_name])
                infra = frame.infrastructures[infra_name]

                if k.endswith("camera"):
                    cTg = self.calib_data[k]["extrinsics"]["cTg"]
                    iTc = self.calib_data[k]["intrinsics"]["IntrinsicMatrixNew"]
                    undistort_function = self.calib_data[k]["intrinsics"]["undistort_function"]
                    image_size = self.calib_data[k]["intrinsics"]["ImageSize"]
                    camera_data = CameraData(
                        k,
                        os.path.join(self.sequence_root, k, v),
                        cTg,
                        iTc,
                        image_size,
                        undistort_function,
                    )
                    infra.add_camera(k, camera_data)

                elif k.endswith("lidar"):
                    gTl = self.calib_data[k]["extrinsics"]["gTl"]
                    lidar_data = LidarData(k, os.path.join(self.sequence_root, k, v), gTl, lidar_index_mapping[k])
                    infra.add_lidar(k, lidar_data)

            # ---------------- Miscellaneous ----------------
            else:
                raise Exception(f"Unrecognized data key: {k}")

        return frame
