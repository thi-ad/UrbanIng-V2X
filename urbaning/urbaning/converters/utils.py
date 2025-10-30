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

import glob
import math
import os
import json
import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


def get_pinhole_undistort_function(cameraParams_dict, alpha=0):

    intrinsicMatrix = np.asarray(cameraParams_dict['IntrinsicMatrix']).T

    radialDistortion = np.asarray(cameraParams_dict['RadialDistortion'])
    tangentialDistortion = np.asarray(cameraParams_dict['TangentialDistortion'])
    distortionCoefficients = np.insert(radialDistortion, 2, tangentialDistortion)

    imageSize = tuple(np.flip(np.asarray(cameraParams_dict['ImageSize'])))  # imageSize --> w x h (eg. 640 x 512)

    newIntrinsicMatrix = cv2.getOptimalNewCameraMatrix(intrinsicMatrix, distortionCoefficients, imageSize, alpha=alpha)[0] # alpha = 1 or 0 determines the undistortion coverage
    mapx, mapy = cv2.initUndistortRectifyMap(intrinsicMatrix, distortionCoefficients, None, newIntrinsicMatrix, imageSize, cv2.CV_32FC1)

    def undistort_function(frame):
        return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    return undistort_function


def get_pinhole_intrinsic_matrix(cameraParams_dict, alpha=0):
    intrinsicMatrix = np.asarray(cameraParams_dict['IntrinsicMatrix']).reshape(3, 3).T

    radialDistortion = np.asarray(cameraParams_dict['RadialDistortion'])
    tangentialDistortion = np.asarray(cameraParams_dict['TangentialDistortion'])
    distortionCoefficients = np.insert(radialDistortion, 2, tangentialDistortion)

    imageSize = tuple(np.flip(cameraParams_dict['ImageSize']))  # imageSize --> w x h (eg. 640 x 512)

    newIntrinsicMatrix = cv2.getOptimalNewCameraMatrix(intrinsicMatrix, distortionCoefficients, imageSize, alpha=alpha)[0]

    return newIntrinsicMatrix


def get_fisheye_undistort_function(cameraParams_dict, alpha=0):
    intrinsicMatrix = np.asarray(cameraParams_dict['IntrinsicMatrix']).T
    distortionCoefficients = np.asarray(cameraParams_dict['DistortionCoefficients'])
    imageSize = tuple(np.flip(np.asarray(cameraParams_dict['ImageSize'])))  # imageSize --> w x h (e.g. 640 x 512)

    newIntrinsicMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsicMatrix, distortionCoefficients, imageSize, np.eye(3), balance=alpha)

    mapx, mapy = cv2.fisheye.initUndistortRectifyMap(
        intrinsicMatrix, distortionCoefficients, np.eye(3), newIntrinsicMatrix, imageSize, cv2.CV_32FC1)

    def undistort_function(frame):
        return cv2.remap(frame, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    return undistort_function


def get_fisheye_intrinsic_matrix(cameraParams_dict, alpha=0):
    intrinsicMatrix = np.asarray(cameraParams_dict['IntrinsicMatrix']).reshape(3, 3).T
    distortionCoefficients = np.asarray(cameraParams_dict['DistortionCoefficients'])

    imageSize = tuple(np.flip(cameraParams_dict['ImageSize']))  # imageSize --> w x h (e.g. 640 x 512)

    newIntrinsicMatrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        intrinsicMatrix, distortionCoefficients, imageSize, np.eye(3), balance=alpha)

    return newIntrinsicMatrix


def redo_calib(calib):
    for topic in calib.keys():
        if topic.endswith('camera'):
            if topic.startswith('vehicle'):
                calib[topic]['extrinsics']['cTv'] = np.asarray(calib[topic]['extrinsics']['cTv'])
                camera_model = 'fisheye' if 'DistortionCoefficients' in calib[topic]['intrinsics'].keys() else 'pinhole'
            else:
                calib[topic]['extrinsics']['cTg'] = np.asarray(calib[topic]['extrinsics']['cTg'])
                camera_model = 'pinhole'
            if camera_model == 'pinhole':
                calib[topic]['intrinsics']['undistort_function'] = get_pinhole_undistort_function(calib[topic]['intrinsics'], alpha=0)
                calib[topic]['intrinsics']['IntrinsicMatrixNew'] = get_pinhole_intrinsic_matrix(calib[topic]['intrinsics'], alpha=0)
            else:
                calib[topic]['intrinsics']['undistort_function'] = get_fisheye_undistort_function(calib[topic]['intrinsics'], alpha=0)
                calib[topic]['intrinsics']['IntrinsicMatrixNew'] = get_fisheye_intrinsic_matrix(calib[topic]['intrinsics'], alpha=0)
        elif topic.endswith('lidar'):
            if topic.startswith('vehicle'):
                calib[topic]['extrinsics']['vTl'] = np.asarray(calib[topic]['extrinsics']['vTl'])
            else:
                calib[topic]['extrinsics']['gTl'] = np.asarray(calib[topic]['extrinsics']['gTl'])
    return calib


def convert_labels_from_tracks_wise_to_timestamp_wise(labels, skip_tracks=[]):
    labels_new = {}

    for track in labels['tracks']:
        if track['track_id'] in skip_tracks:
            continue
        for timestamp_index, timestamp in enumerate(track['timestamps']):
            if timestamp not in labels_new.keys():
                labels_new[timestamp] = []

            this_dict = {
                'track_id': track['track_id'],
                'object_type': track['object_type'],
                'position': track['positions'][timestamp_index],
                'orientation': track['orientations'][timestamp_index],
                'dimension': track['dimensions'][0] if len(track['dimensions']) == 1 else track['dimensions'][timestamp_index],
                'attributes': dict(),
            }

            for k, v in track['attributes'].items():
                if isinstance(v, list):
                    this_dict['attributes'][k] = v[timestamp_index]
                else:
                    this_dict['attributes'][k] = v

            labels_new[timestamp].append(this_dict)

    return labels_new


def get_general_folder_information():
    crossing_cameras = {
        'crossing1': [
            'crossing1_13_thermal_camera',
            'crossing1_14_thermal_camera',
            'crossing1_15_thermal_camera',
            'crossing1_33_thermal_camera',
            'crossing1_34_thermal_camera',
            'crossing1_53_thermal_camera'
        ],
        'crossing2': [
            'crossing2_13_thermal_camera',
            'crossing2_14_thermal_camera',
            'crossing2_15_thermal_camera',
            'crossing2_33_thermal_camera',
            'crossing2_34_thermal_camera',
            'none'
        ],
        'crossing3': [
            'crossing3_13_thermal_camera',
            'crossing3_14_thermal_camera',
            'crossing3_15_thermal_camera',
            'crossing3_23_thermal_camera',
            'crossing3_24_thermal_camera',
            'crossing3_25_thermal_camera'
        ]
    }

    crossing_lidars = {
        'crossing1': [
            'crossing1_11_lidar',
            'crossing1_12_lidar',
            'crossing1_31_lidar',
            'crossing1_32_lidar'
        ],
        'crossing2': [
            'crossing2_11_lidar',
            'crossing2_12_lidar',
            'crossing2_31_lidar',
            'crossing2_32_lidar'
        ],
        'crossing3': [
            'crossing3_11_lidar',
            'crossing3_12_lidar',
            'crossing3_21_lidar',
            'crossing3_22_lidar'
        ]
    }

    vehicle_cameras = {
        'vehicle1': [
            'vehicle1_back_left_camera',
            'vehicle1_left_camera',
            'vehicle1_front_left_camera',
            'vehicle1_front_right_camera',
            'vehicle1_right_camera',
            'vehicle1_back_right_camera',
        ],
        'vehicle2': [
            'vehicle2_back_left_camera',
            'vehicle2_left_camera',
            'vehicle2_front_left_camera',
            'vehicle2_front_right_camera',
            'vehicle2_right_camera',
            'vehicle2_back_right_camera',
        ]
    }

    vehicle_lidars = {
        'vehicle1': [
            'vehicle1_middle_lidar',
        ],
        'vehicle2': [
            'vehicle2_middle_lidar',
        ]
    }

    vehicle_states = {
        'vehicle1': 'vehicle1_state',
        'vehicle2': 'vehicle2_state'
    }

    return crossing_cameras, crossing_lidars, vehicle_cameras, vehicle_lidars, vehicle_states


def extract_one_timestep_information(this_step_info, root_folder, sequence, calib_data, lidar_index_mapping,
                                     camera=True, lidar=True, state=True):
    extracted_data = {}

    for k, v in this_step_info.items():
        if k.endswith('camera') and camera:
            if k.startswith('vehicle'):
                vehicle = k.split('_')[0]
                with open(os.path.join(root_folder, sequence, vehicle + '_state',
                                       str((int(v.split('.')[0]) // 10) * 10) + '.json'), 'r') as f:
                    vehicle_state = json.load(f)
                gTv = np.asarray(vehicle_state['gTv'])
                cTv = calib_data[k]['extrinsics']['cTv']
                cTg = cTv.dot(np.linalg.inv(gTv))
            else:
                cTg = calib_data[k]['extrinsics']['cTg']
            extracted_data[k] = {
                'data': cv2.imread(os.path.join(root_folder, sequence, k, v)),
                'cTg': cTg
            }
        elif k.endswith('lidar') and lidar:
            cali_array = np.load(os.path.join(root_folder, sequence, k, v))
            cali_points = np.ones((len(cali_array['x']), 18), dtype=np.float32)
            cali_points[..., 0] = cali_array['x']
            cali_points[..., 1] = cali_array['y']
            cali_points[..., 2] = cali_array['z']
            range_values = np.linalg.norm(cali_points[..., :3], axis=1)
            cali_points[..., 17] = range_values
            if k.startswith('vehicle'):
                vehicle = k.split('_')[0]
                with open(os.path.join(root_folder, sequence, vehicle + '_state',
                                       str((int(v.split('.')[0]) // 10) * 10) + '.json'), 'r') as f:
                    vehicle_state = json.load(f)
                gTv = np.asarray(vehicle_state['gTv'])
                vTl = calib_data[k]['extrinsics']['vTl']
                gTl = gTv.dot(vTl)
            else:
                gTl = calib_data[k]['extrinsics']['gTl']
            cali_points[..., :4] = cali_points[..., :4].dot(gTl.T)
            cali_points[..., 4] = cali_array['intensity']
            cali_points[..., 5] = cali_array['time_offset_ms']
            cali_points[..., 6] = np.ones(len(cali_points)) * lidar_index_mapping[k]
            extracted_data[k] = {
                'data': cali_points[range_values > 1],
                # pd.DataFrame(cali_points, columns=['x', 'y', 'z', 'ones', 'intensity', 'time_offset', 'lidar_index']),
                'gTl': gTl
            }
        elif k.endswith('state') and state:
            with open(os.path.join(root_folder, sequence, k, v), 'r') as f:
                extracted_data[k] = json.load(f)
            extracted_data[k]['gTv'] = np.asarray(extracted_data[k]['gTv'])
        else:
            extracted_data[k] = v

    return extracted_data


def get_additional_dataset_information():
    crossing_cameras, crossing_lidars, vehicle_cameras, vehicle_lidars, vehicle_states = get_general_folder_information()

    vehicle1_lidar_index = [1]  # [1, 7, 13, 19]
    outdoor_lidar_index = [2, 8, 14, 20]
    crossing1_lidar_index = [3, 9, 15, 21]
    crossing2_lidar_index = [4, 10, 16, 22]
    crossing3_lidar_index = [5, 11, 17, 23]
    vehicle2_lidar_index = [6]  # [6, 12, 18, 24]

    lidar_index_mapping = {
        **{t: i for t, i in zip(crossing_lidars['crossing1'], crossing1_lidar_index)},
        **{t: i for t, i in zip(crossing_lidars['crossing2'], crossing2_lidar_index)},
        **{t: i for t, i in zip(crossing_lidars['crossing3'], crossing3_lidar_index)},
        **{t: i for t, i in zip(vehicle_lidars['vehicle1'], vehicle1_lidar_index)},
        **{t: i for t, i in zip(vehicle_lidars['vehicle2'], vehicle2_lidar_index)},
    }
    vehicle_states_id_mapping = {
        'vehicle1_state': 100000,
        'vehicle2_state': 200000,
        'crossing1': -1,
        'crossing2': -2,
        'crossing3': -3
    }

    return lidar_index_mapping, vehicle_states_id_mapping


def x_to_world(pose):
    """
    The transformation matrix from x-coordinate system to carla world system

    Parameters
    ----------
    pose : list
        [x, y, z, roll, yaw, pitch]

    Returns
    -------
    matrix : np.ndarray
        The transformation matrix.
    """
    x, y, z, roll, yaw, pitch = pose[:]

    # used for rotation matrix
    c_y = np.cos(np.radians(yaw))
    s_y = np.sin(np.radians(yaw))
    c_r = np.cos(np.radians(roll))
    s_r = np.sin(np.radians(roll))
    c_p = np.cos(np.radians(pitch))
    s_p = np.sin(np.radians(pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def world_to_x(matrix):
    """
    The transformation matrix from carla world system to x-coordinate system

    Parameters
    ----------
    matrix : np.ndarray
        The transformation matrix.

    Returns
    -------
    pose : list
        [x, y, z, roll, yaw, pitch]
    """
    # Extract translation components
    x = matrix[0, 3].tolist()
    y = matrix[1, 3].tolist()
    z = matrix[2, 3].tolist()

    # Extract rotation components
    pitch = np.degrees(np.arcsin(matrix[2, 0])).tolist()
    roll = np.degrees(np.arctan2(-matrix[2, 1], matrix[2, 2])).tolist()
    yaw = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])).tolist()

    return [x, y, z, roll, yaw, pitch]
