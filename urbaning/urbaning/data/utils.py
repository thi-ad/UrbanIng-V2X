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


import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from pyproj import Proj

from .info import gps_origins


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
