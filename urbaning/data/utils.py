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


import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from pyproj import Proj


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


def convert_labels_from_tracks_wise_to_timestamp_wise(labels, skip_tracks=[]):
    labels_new = {}

    for track in labels['tracks']:
        if track['track_id'] in skip_tracks:
            continue

        positions = np.array(track['positions'])
        orientations = np.array(track['orientations'])
        quaternions = Rotation.from_euler('Z', orientations).as_quat()

        for timestamp_index, timestamp in enumerate(track['timestamps']):
            if timestamp not in labels_new.keys():
                labels_new[timestamp] = {}

            this_dict = {
                'track_id': track['track_id'],
                'object_type': track['object_type'],
                'position': positions[timestamp_index],
                'quaternion': quaternions[timestamp_index],
                'dimension': track['dimensions'][0] if len(track['dimensions']) == 1 else track['dimensions'][timestamp_index],
                'attributes': dict(),
            }

            for k, v in track['attributes'].items():
                if isinstance(v, list):
                    this_dict['attributes'][k] = v[timestamp_index]
                else:
                    this_dict['attributes'][k] = v

            labels_new[timestamp][track['track_id']] = this_dict

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


def get_vehicle_states_id_mapping():
    vehicle_states_id_mapping = {
        'vehicle1_state': 100000,
        'vehicle2_state': 200000,
        'crossing1': -1,
        'crossing2': -2,
        'crossing3': -3
    }
    return vehicle_states_id_mapping


def get_lidar_index_mapping():
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

    return lidar_index_mapping


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


def get_vPv_numpy(l, w, h):
    l_b2, w_b2, h_b2 = l / 2.0, w / 2.0, h / 2.0
    return np.asarray([
        [l_b2, w_b2, h_b2, 1],  # flt
        [l_b2, -w_b2, h_b2, 1],  # frt
        [-l_b2, -w_b2, h_b2, 1],  # rrt
        [-l_b2, w_b2, h_b2, 1],  # rlt
        [-l_b2, w_b2, -h_b2, 1],  # rlb
        [-l_b2, -w_b2, -h_b2, 1],  # rrb
        [l_b2, -w_b2, -h_b2, 1],  # frb
        [l_b2, w_b2, -h_b2, 1]  # flb
    ])  # 8 x 4  # homogenous coordinate system

def redo_av_vehicle_data(av_vehicle_data):
    for vehicle in av_vehicle_data:
        for key in av_vehicle_data[vehicle]:
            av_vehicle_data[vehicle][key] = np.array(av_vehicle_data[vehicle][key])

        av_vehicle_data[vehicle]['vT_CenterInFloor'] = np.eye(4)
        av_vehicle_data[vehicle]['vT_CenterInFloor'][2, 3] = -av_vehicle_data[vehicle]['size'][2] / 2.0

        av_vehicle_data[vehicle]['vT_RearAxleCenterInFloor'] = np.eye(4)
        av_vehicle_data[vehicle]['vT_RearAxleCenterInFloor'][0, 3] = -av_vehicle_data[vehicle]['dx_center_rearaxle']
        av_vehicle_data[vehicle]['vT_RearAxleCenterInFloor'][2, 3] = -av_vehicle_data[vehicle]['size'][2] / 2.0

        av_vehicle_data[vehicle]['vT_FrontAxleCenterInFloor'] = np.eye(4)
        av_vehicle_data[vehicle]['vT_FrontAxleCenterInFloor'][0, 3] = (av_vehicle_data[vehicle]['dx_frontaxle_rearaxle']
                                                                       - av_vehicle_data[vehicle]['dx_center_rearaxle'])
        av_vehicle_data[vehicle]['vT_FrontAxleCenterInFloor'][2, 3] = -av_vehicle_data[vehicle]['size'][2] / 2.0
    return av_vehicle_data


gps_origins = {
    'crossing1': (48.771731, 11.438043, 419),
    'crossing2': (48.772450, 11.441743, 419),
    'crossing3': (48.769060, 11.438518, 419),
}


# ground_params = {  # with only states
#     'crossing1': [-6.32662449e-05, -2.01889628e-05, -2.02631123e-05,  1.36400142e-03, 7.26723337e-05,  3.80466627e-01],
#     'crossing2': [ 5.22864204e-06,  5.61826591e-06, -4.59121710e-05, -4.05358545e-03, 1.54556150e-03,  4.26684155e-01],
#     'crossing3': [ 5.43821942e-05, -2.27764165e-05,  6.22164573e-05, -4.43234139e-03, 2.13904875e-03, -2.05336238e-01],
# }

ground_params = {  # states and labels
    'crossing1': [-2.17646045e-05, -2.54249987e-05, -8.18657797e-05,  3.81083323e-03, 8.94509838e-04,  4.57465017e-01],
    'crossing2': [-1.06222423e-05, -2.00344008e-05,  2.03389314e-05, -1.28649645e-03, -3.67867551e-04,  5.02631044e-01],
    'crossing3': [-1.21443378e-05, -1.32533272e-05,  3.58418801e-05,  1.78362570e-03, 2.67349162e-03, -7.98810367e-02],
}

# ground_params = { # states < 30m
#     'crossing1': [-0.0002519077294041521, -0.0001985239030466994, -4.2491469767478034e-05, -0.001394149018872422, 0.001348526777358048, 0.4975215230229767],
#     'crossing2': [-8.08576690589262e-05, -0.00027696961944360704, 0.0001751439465562411, -0.005629239959437079, 0.004883528296162676, 0.4775998681257498],
#     'crossing3': [-0.00038634689300494357, -0.00011367482201832447, 0.00012049498162828162, -0.009142698566119543, 0.002231060796454622, -0.11701059595772732]
# }

def get_wTg(crossing_number):
    lato, lono, alto = gps_origins[crossing_number]
    lato_c1, lono_c1, alto_c1 = gps_origins['crossing1']

    utm_projector = Proj(proj="utm", zone=32)
    x_or, y_or = utm_projector(lono, lato)
    x_c1, y_c1 = utm_projector(lono_c1, lato_c1)

    wTg = np.eye(4)
    wTg[0, 3] = x_or - x_c1
    wTg[1, 3] = y_or - y_c1
    wTg[2, 3] = alto - alto_c1

    return wTg