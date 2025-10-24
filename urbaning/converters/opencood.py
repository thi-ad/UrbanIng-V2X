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


import os
import json
import os
import yaml
from tqdm import tqdm

from multiprocessing import Pool, freeze_support
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import cv2

import open3d as o3d

from .utils import (redo_calib, convert_labels_from_tracks_wise_to_timestamp_wise, get_general_folder_information,
                   extract_one_timestep_information, get_additional_dataset_information, x_to_world, world_to_x)


R1 = np.eye(4)
R1[1, 1] = -1

R2 = np.zeros((4, 4))
R2[0, 1] = -1
R2[1, 2] = -1
R2[2, 0] = 1
R2[3, 3] = 1

R3 = np.zeros((4, 4))
R3[0, 1] = -1
R3[1, 0] = 1
R3[2, 2] = 1
R3[3, 3] = 1


def get_camera_cords_from_cTg(cTg):
    # this function is not used for plotting the objects
    gTc = np.linalg.inv(cTg)
    gTc_lh = R1 @ gTc @ R2 @ R1
    return world_to_x(gTc_lh)


def get_lidar_pose_from_gTl(gTl):
    # this function is used for plotting the objects
    gTl_lh = R1 @ gTl @ R3 @ R1
    return world_to_x(gTl_lh)


def get_ego_pose_from_gTv(gTv):
    # this function is not used for plotting the objects
    gTv_lh = R1 @ gTv
    return world_to_x(gTv_lh)


def get_camera_extrinsic_from_cTg_and_gTl(cTg, gTl=None):  # need cTl
    # this function is used for plotting the objects
    if gTl is None:
        return (R1 @ np.linalg.inv(R2) @ cTg @ R3 @ R1).tolist()
    else:
        return (R1 @ np.linalg.inv(R2) @ cTg.dot(gTl @ R3) @ R1).tolist()


def get_o3d_pcd_from_pcd_data(pcd_data_original, gTl):
    lTg_lh = R1 @ np.linalg.inv(gTl @ R3)

    pcd_data = pcd_data_original.copy()
    pcd_data[:, :4] = pcd_data[:, :4].dot(lTg_lh.T)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.tile(pcd_data[:, 4:5], (1, 3)))
    return pcd


def get_object_dict_from_object_dict(obj):
    # gTv is required only for vehicle ; as for infrastructure, global coordinate is the vehicle coordinate

    gTo = np.eye(4)
    # obj['orientation'] = 0
    gTo[:-1, :-1] = Rotation.from_euler('xyz', [0, 0, obj['orientation']], degrees=False).as_matrix()
    gTo[:-1, -1] = obj['position']  # [0, 0, 0]  #obj['position'] # [0, 0, 0]  #

    gTo_lh = R1 @ gTo
    new_pos = world_to_x(gTo_lh)

    this_dict = {
        'angle': new_pos[3:],
        'center': [0, 0, 0],
        'extent': [i / 2.0 for i in obj['dimension']],
        'location': new_pos[:3],
    }
    return this_dict


def get_object_dict_for_other_avs(other_vehicles, vehicle_ids, root_folder, sequence, timestamp_ms, pcd_data, dimensions):
    this_dicts = {}
    for v, v_id, dimension in zip(other_vehicles, vehicle_ids, dimensions):
        with open(os.path.join(root_folder, sequence, v + '_state', str(timestamp_ms) + '.json'), 'r') as f:
            v_state = json.load(f)
        gTv = np.asarray(v_state['gTv'])
        new_points_in_v_coordinates = pcd_data[:, :4].dot(np.linalg.inv(gTv).T)
        new_points_in_normalized_v_coordinates = np.abs(new_points_in_v_coordinates[:, :3]) * 2 / np.asarray(dimension)

        _find_all = np.all(new_points_in_normalized_v_coordinates < 1, axis=1)
        has_points = bool(np.any(_find_all))
        if has_points:
            gTv_lh = R1 @ gTv
            new_pos = world_to_x(gTv_lh)

            this_dict = {
                'angle': new_pos[3:],
                'center': [0, 0, 0],
                'extent': [i / 2.0 for i in dimension],
                'location': new_pos[:3],
                'point_cloud_hits': int(np.sum(_find_all))
            }

            this_dicts[v_id] = this_dict
    return this_dicts


def is_there_a_point_inside_object(pcd_data, object_dict):
    # return True
    gTv = np.eye(4)
    gTv[:-1, -1] = object_dict['position']
    gTv[:-1, :-1] = Rotation.from_euler('xyz', [0, 0, object_dict['orientation']], degrees=False).as_matrix()
    new_points_in_v_coordinates = pcd_data[:, :4].dot(np.linalg.inv(gTv).T)
    new_points_in_normalized_v_coordinates = np.abs(new_points_in_v_coordinates[:, :3]) * 2 / np.asarray(
        object_dict['dimension'])

    _find_all = np.all(new_points_in_normalized_v_coordinates < 1, axis=1)
    has_points = bool(np.any(_find_all))
    point_count = int(np.sum(_find_all))

    return has_points, point_count


def do_one_sequence(args):
    sequence, root_folder, labels_folder, target_folder = args
    crossing_number = sequence.split('_')[2]

    calibration_file = os.path.join(root_folder, sequence, 'calibration.json')
    time_sync_info_file = os.path.join(root_folder, sequence, 'timesync_info.csv')

    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)

    time_sync_df = pd.read_csv(time_sync_info_file).set_index('Unnamed: 0')

    calib_data = redo_calib(calib_data)
    with open(os.path.join(labels_folder, sequence + '.json'), 'r') as f:
        labels = json.load(f)
    with open(labels_folder + '_av_track_ids.json', 'r') as f:
        av_track_id_map = json.load(f)

    labels = convert_labels_from_tracks_wise_to_timestamp_wise(labels,
                                                               skip_tracks=[av_track_id_map[sequence]['vehicle1'],
                                                                            av_track_id_map[sequence]['vehicle2']])

    crossing_cameras, crossing_lidars, vehicle_cameras, vehicle_lidars, vehicle_states = get_general_folder_information()
    lidar_index_mapping, vehicle_states_id_mapping = get_additional_dataset_information()

    save_path_crossing = os.path.join(target_folder, sequence, str(vehicle_states_id_mapping[crossing_number]))
    save_path_vehicle1 = os.path.join(target_folder, sequence, str(vehicle_states_id_mapping['vehicle1_state']))
    save_path_vehicle2 = os.path.join(target_folder, sequence, str(vehicle_states_id_mapping['vehicle2_state']))

    os.makedirs(save_path_crossing, exist_ok=True)
    os.makedirs(save_path_vehicle1, exist_ok=True)
    os.makedirs(save_path_vehicle2, exist_ok=True)

    for col in tqdm(time_sync_df.columns):
        this_step_info = time_sync_df[col]

        this_ts = round(int(this_step_info['timestamp_ms']) * 0.001, 2)
        if this_ts not in labels.keys():
            continue

        extracted_data = extract_one_timestep_information(
            this_step_info, root_folder, sequence, calib_data, lidar_index_mapping)

        for vehicle, other_vehicles, save_folder in zip(['vehicle1', 'vehicle2'], [['vehicle2'], ['vehicle1']],
                                                        [save_path_vehicle1, save_path_vehicle2]):
            vehicle_middle_lidar = vehicle + '_middle_lidar'
            vehicle_state = vehicle + '_state'
            this_ts_yaml_info = {}

            for i, topic in enumerate(vehicle_cameras[vehicle]):
                this_camera_config = {
                    'cords': get_camera_cords_from_cTg(extracted_data[topic]['cTg']),
                    'extrinsic': get_camera_extrinsic_from_cTg_and_gTl(extracted_data[topic]['cTg'],
                                                                       extracted_data[vehicle_middle_lidar]['gTl']),
                    'intrinsic': (calib_data[topic]['intrinsics']['IntrinsicMatrixNew']).tolist()
                }
                this_ts_yaml_info[f'camera{i}'] = this_camera_config

                image_distorted = extracted_data[topic]['data']
                image = calib_data[topic]['intrinsics']['undistort_function'](image_distorted)
                cv2.imwrite(os.path.join(save_folder, f'{str(col).zfill(6)}_camera{i}.png'), image)

            pcd_data = np.concatenate([extracted_data[t]['data'] for t in vehicle_lidars[vehicle]])
            pcd = get_o3d_pcd_from_pcd_data(pcd_data, extracted_data[vehicle_middle_lidar]['gTl'])
            o3d.io.write_point_cloud(os.path.join(save_folder, f'{str(col).zfill(6)}.pcd'), pcd)

            this_ts_yaml_info['ego_speed'] = (np.sign(extracted_data[vehicle_state]['vV'][0]) * np.linalg.norm(
                extracted_data[vehicle_state]['vV'])).tolist()
            this_ts_yaml_info['lidar_pose'] = get_lidar_pose_from_gTl(extracted_data[vehicle_middle_lidar]['gTl'])
            this_ts_yaml_info['predicted_ego_pos'] = get_ego_pose_from_gTv(extracted_data[vehicle_state]['gTv'])
            this_ts_yaml_info['true_ego_pos'] = get_ego_pose_from_gTv(extracted_data[vehicle_state]['gTv'])

            object_dict = {}
            for object in labels[this_ts]:
                has_points, point_count = is_there_a_point_inside_object(pcd_data, object)
                if not has_points:
                    continue
                if object['track_id'] == av_track_id_map[sequence][vehicle]:
                    continue
                if object['object_type'] not in object_dict.keys():
                    object_dict[object['object_type']] = {}

                # get usual opencood object info
                opencood_object_info = get_object_dict_from_object_dict(object)
                # add lidar points count
                opencood_object_info['point_cloud_hits'] = point_count
                object_dict[object['object_type']][object['track_id']] = opencood_object_info  # , extracted_data[vehicle_middle_lidar]['gTl'])

            additional_dict = get_object_dict_for_other_avs(other_vehicles,
                                                            [vehicle_states_id_mapping[v + '_state'] for v in
                                                             other_vehicles], root_folder, sequence,
                                                            this_step_info['timestamp_ms'], pcd_data,
                                                            [calib_data[v + '_state']['dimension'] for v in
                                                             other_vehicles])

            if 'Car' not in object_dict.keys():
                print('Sequence may be wrong:', sequence, 'Vehicle:', vehicle, 'Timestamp:', this_ts)
                object_dict['Car'] = additional_dict
            else:
                object_dict['Car'] = {**additional_dict, **object_dict['Car']}

            this_ts_yaml_info = {**this_ts_yaml_info, **object_dict}

            with open(os.path.join(save_folder, f'{str(col).zfill(6)}.yaml'), 'w') as f:
                yaml.dump(this_ts_yaml_info, f, sort_keys=False)

        this_ts_yaml_info = {}

        for i, topic in enumerate(crossing_cameras[crossing_number]):
            if topic == 'none':
                continue
            this_camera_config = {
                'cords': get_camera_cords_from_cTg(extracted_data[topic]['cTg']),
                'extrinsic': get_camera_extrinsic_from_cTg_and_gTl(extracted_data[topic]['cTg']),
                'intrinsic': (calib_data[topic]['intrinsics']['IntrinsicMatrixNew']).tolist()
            }
            this_ts_yaml_info[f'camera{i}'] = this_camera_config

            image_distorted = extracted_data[topic]['data']
            image = calib_data[topic]['intrinsics']['undistort_function'](image_distorted)
            cv2.imwrite(os.path.join(save_path_crossing, f'{str(col).zfill(6)}_camera{i}.png'), image)

        pcd_data = np.concatenate([extracted_data[t]['data'] for t in crossing_lidars[crossing_number]])
        pcd = get_o3d_pcd_from_pcd_data(pcd_data, np.eye(4))
        o3d.io.write_point_cloud(os.path.join(save_path_crossing, f'{str(col).zfill(6)}.pcd'), pcd)

        this_ts_yaml_info['ego_speed'] = 0
        this_ts_yaml_info['lidar_pose'] = get_lidar_pose_from_gTl(np.eye(4))  # [0, 0, 0, 0, 0, 0]
        this_ts_yaml_info['predicted_ego_pos'] = get_ego_pose_from_gTv(np.eye(4))  # [0, 0, 0, 0, 0, 0]
        this_ts_yaml_info['true_ego_pos'] = get_ego_pose_from_gTv(np.eye(4))  # [0, 0, 0, 0, 0, 0]

        object_dict = {}
        for object in labels[this_ts]:
            has_points, point_count = is_there_a_point_inside_object(pcd_data, object)
            if not has_points:
                continue
            if object['object_type'] not in object_dict.keys():
                object_dict[object['object_type']] = {}

            # get usual opencood object info
            opencood_object_info = get_object_dict_from_object_dict(object)
            # add lidar points count
            opencood_object_info['point_cloud_hits'] = point_count
            object_dict[object['object_type']][object['track_id']] = opencood_object_info  # , np.eye(4))

        other_vehicles = ['vehicle1', 'vehicle2']
        additional_dict = get_object_dict_for_other_avs(other_vehicles,
                                                        [vehicle_states_id_mapping[v + '_state'] for v in
                                                         other_vehicles], root_folder, sequence,
                                                        this_step_info['timestamp_ms'], pcd_data,
                                                        [calib_data[v + '_state']['dimension'] for v in other_vehicles])

        object_dict['Car'] = {**additional_dict, **object_dict['Car']}
        this_ts_yaml_info = {**this_ts_yaml_info, **object_dict}

        with open(os.path.join(save_path_crossing, f'{str(col).zfill(6)}.yaml'), 'w') as f:
            yaml.dump(this_ts_yaml_info, f, sort_keys=False)


def urbaning_to_opencood_format(source_folder, target_folder, use_multiprocessing=True):
    if not os.path.exists(source_folder):
        raise "source_folder does not exist"

    os.makedirs(target_folder, exist_ok=True)

    dataset_folder = os.path.join(source_folder, 'dataset')
    labels_folder = os.path.join(source_folder, 'labels')

    sequences = sorted(os.listdir(dataset_folder))

    args = [[sequence, dataset_folder, labels_folder, target_folder] for sequence in sequences]

    if not use_multiprocessing:
        freeze_support()
        for arg in args:
            do_one_sequence(arg)
    else:

        with Pool() as pool:
            pool.map(do_one_sequence, args)

if __name__ == "__main__":
    freeze_support()

    source_folder = r'/path/to/source_folder'
    target_folder = r'/path/to/target_folder'

    urbaning_to_opencood_format(source_folder, target_folder)