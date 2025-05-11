import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'

import glob
import json

from scipy.spatial.transform import Rotation
import numpy as np
import math
import cv2

import matplotlib
from tqdm import tqdm

from multiprocessing import freeze_support, Pool

from utils import (extract_separate_source_folders_from_root_folder, get_general_folder_information,
                   redo_calib, convert_labels_from_tracks_wise_to_timestamp_wise, get_additional_dataset_information,
                   extract_one_timestep_information)

from plot_utils import (plot_pcd_in_image_np, plot_labels_in_image, plot_states_in_image, plot_pcd_in_lidar_frame_np,
                        plot_labels_in_lidar_frame, plot_states_in_lidar_frame)

def video_generation_config():
    video_config = {
        'final_width': 1920,
        'final_height': 1080,
        'image_config': {
            'sources': 'all',
            # 'all' 'crossing' 'vehicle1' 'vehicle2' 'vehicles' ['specific_camera1', 'sc2'] 'crossing_vehicle1' 'crossing_vehicle2' --> crossing and one vehicle
            'draw_order': ['pcd', 'labels'],  # first pcd is plotted .. and state is plotted last
            'labels_config': {
                'line_thickness': 3,
                'text_scale': 1,
            },
            'pcd_config': {
                'sources': 'all',
                # 'none' 'all' 'crossing' 'vehicle1' 'vehicle2' 'vehicles' ['specific_lidar1', 'sl2] 'crossing_vehicle1' 'crossing_vehicle2' --> crossing and one vehicle
                'color': 'intensity',  # 'z' 'time_offset', 'lidar_index'
                'color_map': 'gist_rainbow',
            },
            'states_config': {
                'sources': ['vehicle1_state', 'vehicle2_state'],  # ['vehicle1_state'], ['vehicle2_state']
                'observation_window': 0,  # number of time instances to show from history
                'prediction_window': 0,  # number of time instances to show from future
                'line_thickness': 3,
                'text_scale': 1,
                'plot_self': False
            }
        },
        'pcd_config': {
            'sources': 'all',
            # 'none' 'all' 'crossing' 'vehicle1' 'vehicle2' 'vehicles' ['specific_lidar1', 'sl2] 'crossing_vehicle1' 'crossing_vehicle2' --> crossing and one vehicle
            'draw_order': ['pcd', 'labels', 'states'],  # first pcd is plotted .. and state is plotted last
            'color': 'intensity',  # 'z' 'time_offset', 'lidar_index'
            'color_map': 'gist_rainbow',
            'origin': 'crossing',  # 'vehicle1, vehicle2, crossing
            'extent': 100,  # in meters -x until +x
            'labels_config': {
                'line_thickness': 2,
                'text_scale': 1,
                'filled': False
            },
            'states_config': {
                'sources': ['vehicle1_state', 'vehicle2_state'],  # ['vehicle1'], ['vehicle2']
                'observation_window': 0,  # number of time instances to show from history
                'prediction_window': 0,  # number of time instances to show from future
                'line_thickness': 2,
                'text_scale': 1,
                'filled': True,
            }
        },
    }
    return video_config


def redo_video_generation_config(video_config, crossing_number):
    crossing_cameras, crossing_lidars, vehicle_cameras, vehicle_lidars, vehicle_states = get_general_folder_information()

    camera_source_mapping = {
        'none': [],
        'all': vehicle_cameras['vehicle1'] + vehicle_cameras['vehicle2'] + crossing_cameras[crossing_number],
        'crossing': crossing_cameras[crossing_number],
        'vehicles': vehicle_cameras['vehicle1'] + vehicle_cameras['vehicle2'],
        'vehicle1': vehicle_cameras['vehicle1'],
        'vehicle2': vehicle_cameras['vehicle2'],
        'crossing_vehicle1': vehicle_cameras['vehicle1'] + crossing_cameras[crossing_number],
        'crossing_vehicle2': vehicle_cameras['vehicle2'] + crossing_cameras[crossing_number],
    }

    lidar_source_mapping = {
        'none': [],
        'all': vehicle_lidars['vehicle1'] + vehicle_lidars['vehicle2'] + crossing_lidars[crossing_number],
        'crossing': crossing_lidars[crossing_number],
        'vehicles': vehicle_lidars['vehicle1'] + vehicle_lidars['vehicle2'],
        'vehicle1': vehicle_lidars['vehicle1'],
        'vehicle2': vehicle_lidars['vehicle2'],
        'crossing_vehicle1': vehicle_lidars['vehicle1'] + crossing_lidars[crossing_number],
        'crossing_vehicle2': vehicle_lidars['vehicle2'] + crossing_lidars[crossing_number],
    }

    sources = video_config['image_config']['sources']
    if isinstance(sources, list):
        if len(sources):
            assert all(f in camera_source_mapping['all'] for f in sources)
    else:
        if sources in camera_source_mapping:
            video_config['image_config']['sources'] = camera_source_mapping[sources]
        else:
            raise 'Unknown camera source specified'

    sources = video_config['image_config']['pcd_config']['sources']
    if isinstance(sources, list):
        if len(sources):
            assert all(f in lidar_source_mapping['all'] for f in sources)
    else:
        if sources in lidar_source_mapping:
            video_config['image_config']['pcd_config']['sources'] = lidar_source_mapping[sources]
        else:
            raise 'Unknown lidar source specified'

    sources = video_config['pcd_config']['sources']
    if isinstance(sources, list):
        if len(sources):
            assert all(f in lidar_source_mapping['all'] for f in sources)
    else:
        if sources in lidar_source_mapping:
            video_config['pcd_config']['sources'] = lidar_source_mapping[sources]
        else:
            raise 'Unknown lidar source specified'

    has_lidar = bool(len(video_config['pcd_config']['sources']))
    num_camera_sources = len(video_config['image_config']['sources'])
    position_information = {}

    cams = video_config['image_config']['sources']
    if has_lidar:
        if num_camera_sources == 0:
            position_information['lidar'] = {'x': 0, 'y': 0, 'w': video_config['final_width'],
                                             'h': video_config['final_height']}
        elif num_camera_sources == 6:
            w = (video_config['final_width'] / 4)
            h = (video_config['final_height'] / 4)
            bl1, l1, fl1, fr1, r1, br1 = cams
            pos_info = {
                fl1: (0, 0.5), fr1: (1, 0.5),
                l1: (0, 1.5), r1: (1, 1.5),
                bl1: (0, 2.5), br1: (1, 2.5),
            }
            position_information = {k: {'x': x * w, 'y': y * h, 'w': w, 'h': h} for k, (x, y) in pos_info.items()}
            position_information['lidar'] = {'x': 2 * w, 'y': 0, 'w': 2 * w, 'h': 4 * h}
        elif num_camera_sources == 12:
            w = (video_config['final_width'] / 6)
            h = (video_config['final_height'] / 6)
            bl1, l1, fl1, fr1, r1, br1, bl2, l2, fl2, fr2, r2, br2, = cams
            pos_info = {
                **{bl1: (0, 0), l1: (1, 0), fl1: (2, 0), fr1: (3, 0), r1: (4, 0), br1: (5, 0)},
                **{bl2: (0, 1), l2: (1, 1), fl2: (2, 1), fr2: (3, 1), r2: (4, 1), br2: (5, 1)},
            }
            position_information = {k: {'x': x * w, 'y': y * h, 'w': w, 'h': h} for k, (x, y) in pos_info.items()}
            position_information['lidar'] = {'x': 0 * w, 'y': 2 * h, 'w': 6 * w, 'h': 4 * h}
        elif num_camera_sources == 18:
            w = (video_config['final_width'] / 6)
            h = (video_config['final_height'] / 5)
            bl1, l1, fl1, fr1, r1, br1, bl2, l2, fl2, fr2, r2, br2, bl3, l3, fl3, fr3, r3, br3 = cams
            pos_info = {
                **{bl1: (0, 0), l1: (1, 0), fl1: (2, 0), fr1: (3, 0), r1: (4, 0), br1: (5, 0)},
                **{bl2: (0, 1), l2: (1, 1), fl2: (2, 1), fr2: (3, 1), r2: (4, 1), br2: (5, 1)},
                **{bl3: (0, 2), l3: (1, 2)},
                **{fl3: (0, 3), fr3: (1, 3)},
                **{r3: (0, 4), br3: (1, 4)},
            }
            position_information = {k: {'x': x * w, 'y': y * h, 'w': w, 'h': h} for k, (x, y) in pos_info.items()}
            position_information['lidar'] = {'x': 2 * w, 'y': 2 * h, 'w': 4 * w, 'h': 3 * h}

        else:
            raise "Unknown number of cameras requested"

        lp = position_information['lidar']
        pixels_per_meter = min(lp['w'], lp['h']) / video_config['pcd_config']['extent']
        offset = np.array([[lp['w'] / 2, lp['h'] / 2]])
        video_config['pcd_config']['pixels_per_meter'] = pixels_per_meter
        video_config['pcd_config']['offset'] = offset

    else:
        if num_camera_sources == 1:
            w = video_config['final_width']
            h = video_config['final_height']
            position_information[cams[0]] = {'x': 0, 'y': 0, 'w': w, 'h': h}
        elif num_camera_sources == 6:  # 2 x 3
            w = (video_config['final_width'] / 3)
            h = (video_config['final_height'] / 3)
            bl1, l1, fl1, fr1, r1, br1 = cams
            pos_info = {
                **{bl1: (0, 0.5), l1: (1, 0.5), fl1: (2, 0.5)},
                **{fr1: (0, 1.5), r1: (1, 1.5), br1: (2, 1.5)},
            }
            position_information = {k: {'x': x * w, 'y': y * h, 'w': w, 'h': h} for k, (x, y) in pos_info.items()}

        elif num_camera_sources == 12:  # 3 x 4
            w = (video_config['final_width'] / 4)
            h = (video_config['final_height'] / 4)
            bl1, l1, fl1, fr1, r1, br1, bl2, l2, fl2, fr2, r2, br2 = cams
            pos_info = {
                **{fl1: (0, 0.5), fr1: (1, 0.5), fl2: (2, 0.5), fr2: (3, 0.5)},
                **{l1: (0, 1.5), r1: (1, 1.5), l2: (2, 1.5), r2: (3, 1.5)},
                **{bl1: (0, 2.5), br1: (1, 2.5), bl2: (2, 2.5), br2: (3, 2.5)},
            }
            position_information = {k: {'x': x * w, 'y': y * h, 'w': w, 'h': h} for k, (x, y) in pos_info.items()}

        elif num_camera_sources == 18:  # 3 x 6
            w = (video_config['final_width'] / 6)
            h = (video_config['final_height'] / 6)
            bl1, l1, fl1, fr1, r1, br1, bl2, l2, fl2, fr2, r2, br2, bl3, l3, fl3, fr3, r3, br3 = cams
            pos_info = {
                **{bl1: (0, 1.5), l1: (1, 1.5), fl1: (2, 1.5), fr1: (3, 1.5), r1: (4, 1.5), br1: (5, 1.5)},
                **{bl2: (0, 2.5), l2: (1, 2.5), fl2: (2, 2.5), fr2: (3, 2.5), r2: (4, 2.5), br2: (5, 2.5)},
                **{bl3: (0, 3.5), l3: (1, 3.5), fl3: (2, 3.5), fr3: (3, 3.5), r3: (4, 3.5), br3: (5, 3.5)},
            }
            position_information = {k: {'x': x * w, 'y': y * h, 'w': w, 'h': h} for k, (x, y) in pos_info.items()}

        else:
            raise "Unknown number of cameras requested"

    for k, v in position_information.items():
        position_information[k] = {
            'x': math.floor(v['x']),
            'y': math.floor(v['y']),
            'w': math.floor(v['w']),
            'h': math.floor(v['h'])
        }
    video_config['position_information'] = position_information

    video_config['image_config']['pcd_config']['color_map'] = matplotlib.colormaps[video_config['image_config']['pcd_config']['color_map']]
    video_config['pcd_config']['color_map'] = matplotlib.colormaps[video_config['pcd_config']['color_map']]

    return video_config


def all_data_in_one_video(root_folder, labels_folder, sequence, video_config, save_file):
    """
    The output video will have 2 car camera images + infrastructure camera images +
    separate fused lidar point cloud data video
    """

    (
        crossing_camera_folders, crossing_lidar_folders,
        vehicle1_camera_folders, vehicle1_lidar_folders, vehicle1_state_folder,
        vehicle2_camera_folders, vehicle2_lidar_folders, vehicle2_state_folder,
        calib_data, time_sync_df
    ) = extract_separate_source_folders_from_root_folder(root_folder, sequence)

    calib_data = redo_calib(calib_data)
    labels_file = os.path.join(labels_folder, sequence + '.json')
    if not os.path.isfile(labels_file):
        print(f'Label file {labels_file} not found!')
        return
    # return

    with open(labels_file, 'r') as f:
        labels = json.load(f)
    labels = convert_labels_from_tracks_wise_to_timestamp_wise(labels)

    lidar_index_mapping, vehicle_states_id_mapping = get_additional_dataset_information()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 10
    frame_size = (video_config['final_width'], video_config['final_height'])
    writer = cv2.VideoWriter(save_file, fourcc, fps, frame_size)
    max_color_number = 0

    for col in tqdm(time_sync_df.columns):
        this_step_info = time_sync_df[col]

        this_ts = round(int(this_step_info['timestamp_ms']) * 0.001, 2)
        if this_ts not in labels.keys():
            continue

        extracted_data = extract_one_timestep_information(
            this_step_info, root_folder, sequence, calib_data, lidar_index_mapping)

        if 'pcd' in video_config['image_config']['draw_order']:
            # pcd_for_image_original = pd.concat([extracted_data[t]['data'] for t in video_config['image_config']['pcd_config']['sources']])
            # c = pcd_for_image_original[video_config['image_config']['pcd_config']['color']]
            # max_color_number = max(max_color_number, max(c))
            # pcd_for_image_original[['r', 'g', 'b', 'a']] = video_config['image_config']['pcd_config']['color_map'](c / max_color_number) * 255
            pcd_for_image_original = np.concatenate(
                [extracted_data[t]['data'] for t in video_config['image_config']['pcd_config']['sources']])

            if video_config['image_config']['pcd_config']['color'] == 'intensity':
                c = pcd_for_image_original[:, 4]
            elif video_config['image_config']['pcd_config']['color'] == 'lidar_index':
                c = pcd_for_image_original[:, 6]
            elif video_config['image_config']['pcd_config']['color'] == 'time_offset':
                c = pcd_for_image_original[:, 5]

            max_color_number = max(max_color_number, max(c))
            pcd_for_image_original[:, 7:11] = video_config['image_config']['pcd_config']['color_map'](
                c / max_color_number) * 255

        constructed_image = np.zeros((video_config['final_height'], video_config['final_width'], 3), dtype=np.uint8)

        for topic in video_config['image_config']['sources']:
            if topic == 'none':
                continue
            image_distorted = extracted_data[topic]['data']
            image = calib_data[topic]['intrinsics']['undistort_function'](image_distorted)
            height, width, _ = image.shape

            cTg = extracted_data[topic]['cTg']
            iTc = calib_data[topic]['intrinsics']['IntrinsicMatrixNew']

            for d in video_config['image_config']['draw_order']:
                if d == 'pcd':
                    plot_pcd_in_image_np(image, pcd_for_image_original, cTg, iTc)

                elif d == 'labels':
                    text_scale = video_config['image_config']['labels_config']['text_scale'] * width / 640
                    line_thickness = round(
                        video_config['image_config']['labels_config']['line_thickness'] * width / 640)
                    plot_labels_in_image(image, labels[this_ts], cTg, iTc, text_scale=text_scale,
                                         thickness=line_thickness)

                elif d == 'states':
                    if video_config['image_config']['states_config']['plot_self']:
                        ignore_key = topic.split('_')[0] + '_state' if topic.startswith('vehicle') else ''
                    else:
                        ignore_key = ''

                    state_informations = {
                        k: {**v, **calib_data[k], 'id': vehicle_states_id_mapping[k]}
                        for k, v in extracted_data.items() if k.endswith('state') and not k == ignore_key
                    }
                    text_scale = video_config['image_config']['states_config']['text_scale'] * width / 640
                    line_thickness = round(
                        video_config['image_config']['states_config']['line_thickness'] * width / 640)
                    plot_states_in_image(image, state_informations, cTg, iTc, text_scale=text_scale,
                                         thickness=line_thickness)

            pos_dict = video_config['position_information'][topic]
            x, y, w, h = pos_dict['x'], pos_dict['y'], pos_dict['w'], pos_dict['h']
            constructed_image[y:y + h, x:x + w] = cv2.resize(image, (w, h))

        if len(video_config['pcd_config']['sources']):
            # pcd_for_lidar = pd.concat([extracted_data[t]['data'] for t in video_config['pcd_config']['sources']])
            # c = pcd_for_lidar[video_config['pcd_config']['color']]
            # max_color_number = max(max_color_number, max(c))
            # pcd_for_lidar[['r', 'g', 'b', 'a']] = video_config['pcd_config']['color_map'](c / max_color_number) * 255

            pcd_for_lidar = np.concatenate([extracted_data[t]['data'] for t in video_config['pcd_config']['sources']])
            if video_config['image_config']['pcd_config']['color'] == 'intensity':
                c = pcd_for_image_original[:, 4]
            elif video_config['image_config']['pcd_config']['color'] == 'lidar_index':
                c = pcd_for_image_original[:, 6]
            elif video_config['image_config']['pcd_config']['color'] == 'time_offset':
                c = pcd_for_image_original[:, 5]
            
            c = pcd_for_lidar[:, 4]
            max_color_number = max(max_color_number, max(c))
            pcd_for_lidar[:, 7:11] = video_config['pcd_config']['color_map'](c / max_color_number) * 255

            pixels_per_meter = video_config['pcd_config']['pixels_per_meter']
            offset = video_config['pcd_config']['offset']
            pos_dict = video_config['position_information']['lidar']
            x, y, w, h = pos_dict['x'], pos_dict['y'], pos_dict['w'], pos_dict['h']
            lidar_frame = np.zeros((h, w, 3), dtype=np.uint8)

            state_informations = {
                k: {**v, **calib_data[k], 'id': vehicle_states_id_mapping[k]}
                for k, v in extracted_data.items() if k.endswith('state')
            }

            vTg = np.eye(4)
            if 'crossing1' in sequence:
                vTg[:-1, :-1] = Rotation.from_euler('xyz', [0, 0, -45], degrees=True).as_matrix()
            elif 'crossing2' in sequence:
                vTg[:-1, :-1] = Rotation.from_euler('xyz', [0, 0, -25], degrees=True).as_matrix()
            elif 'crossing3' in sequence:
                vTg[:-1, :-1] = Rotation.from_euler('xyz', [0, 0, -90], degrees=True).as_matrix()

            if video_config['pcd_config']['origin'] == 'vehicle1':
                vTg = np.linalg.inv(state_informations['vehicle1_state']['gTv'])
            elif video_config['pcd_config']['origin'] == 'vehicle2':
                vTg = np.linalg.inv(state_informations['vehicle2_state']['gTv'])

            for d in video_config['pcd_config']['draw_order']:
                if d == 'pcd':
                    plot_pcd_in_lidar_frame_np(lidar_frame, pcd_for_lidar, pixels_per_meter, offset, vTg)
                elif d == 'labels':
                    text_scale = video_config['pcd_config']['labels_config']['text_scale'] / 2 # * 100 / video_config['pcd_config']['extent']
                    line_thickness = round(video_config['pcd_config']['labels_config']['line_thickness']) # * 100 / video_config['pcd_config']['extent'])
                    filled = video_config['pcd_config']['labels_config']['filled']
                    plot_labels_in_lidar_frame(lidar_frame, labels[this_ts], pixels_per_meter, offset, text_scale=text_scale, thickness=line_thickness, filled=filled, vTg=vTg)

                elif d == 'states':
                    text_scale = video_config['pcd_config']['states_config']['text_scale'] / 2  #  * video_config['pcd_config']['extent'] / 200
                    line_thickness = round(video_config['pcd_config']['states_config']['line_thickness']) # * video_config['pcd_config']['extent'] / 100)
                    filled = video_config['pcd_config']['states_config']['filled']
                    plot_states_in_lidar_frame(lidar_frame, state_informations, pixels_per_meter, offset,text_scale=text_scale, thickness=line_thickness, filled=filled, vTg=vTg)

            constructed_image[y:y + h, x:x + w] = lidar_frame

        writer.write(constructed_image)
        # break
    writer.release()



def do_one_sequence(args):

    root_folder, labels_folder, sequence, videos_dir = args
    crossing_number = sequence.split('_')[2]
    video_config = video_generation_config()
    video_config = redo_video_generation_config(video_config, crossing_number)
    video_save_file = os.path.join(videos_dir, f'{sequence}.mp4')
    all_data_in_one_video(root_folder, labels_folder, sequence, video_config, video_save_file)


def process_all_sequence(root_folder, labels_folder, save_folder):
    sequences = os.listdir(root_folder)

    args = [[root_folder, labels_folder, sequence, save_folder] for sequence in sequences]

    for arg in args:
        do_one_sequence(arg)


def main():
    root_folder = r"/path/to/dataset"
    labels_folder = r"/path/to/labels"
    save_folder = r"/path/to/save/videos"

    # To create videos for all the sequences
    # process_all_sequence(root_folder, labels_folder, save_folder)

    # To create videos for the selected sequences
    sequence = "20241126_0024_crossing1_18"
    do_one_sequence([root_folder, labels_folder, sequence, save_folder])

if __name__ == "__main__":
    freeze_support()
    main()

