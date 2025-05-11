import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'

import open3d as o3d
import os
import json

from utils import (extract_separate_source_folders_from_root_folder, redo_calib,
                   convert_labels_from_tracks_wise_to_timestamp_wise, get_additional_dataset_information,
                   extract_one_timestep_information)

import numpy as np
import matplotlib
from multiprocessing import Pool, freeze_support, cpu_count


def create_text_3d(text, pos, depth=1, scale=0.2):
    hello_open3d_mesh = o3d.t.geometry.TriangleMesh.create_text(text, depth=depth).to_legacy()
    hello_open3d_mesh.paint_uniform_color((0, 0, 0))
    hello_open3d_mesh.transform([[scale, 0, 0, pos[0]], [0, scale, 0, pos[1]], [0, 0, scale, pos[2]], [0, 0, 0, 1]])
    return hello_open3d_mesh


class PcdVisualizer:
    def __init__(self, shared_dict):
        self.shared_dict = shared_dict
        self.current_index = 0
        self.color_index = 0

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        render_option = self.vis.get_render_option()
        render_option.point_size = 2.0  # Adjust this value as needed

        self.load_current_pcd(True)

        self.vis.register_key_callback(ord('1'), lambda vis: self.different_pcd(-20))  # 'A' key for previous PCD
        self.vis.register_key_callback(ord('2'), lambda vis: self.different_pcd(-10))  # 'A' key for previous PCD
        self.vis.register_key_callback(ord('A'), lambda vis: self.different_pcd(-1))  # 'D' key for next PCD
        self.vis.register_key_callback(ord('D'), lambda vis: self.different_pcd(1))  # 'A' key for previous PCD
        self.vis.register_key_callback(ord('3'), lambda vis: self.different_pcd(10))  # 'A' key for previous PCD
        self.vis.register_key_callback(ord('4'), lambda vis: self.different_pcd(20))  # 'A' key for previous PCD

        self.vis.register_key_callback(ord('E'), lambda vis: self.change_color_index())  # 'A' key for previous PCD

    def change_color_index(self):
        self.color_index = (self.color_index + 1) % 3
        self.load_current_pcd()

    def load_current_pcd(self, reset_bounding_box=False):
        pcd_for_lidar_points, colors, objects = self.shared_dict[self.current_index]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_for_lidar_points)
        pcd.colors = o3d.utility.Vector3dVector(colors[self.color_index])

        self.vis.clear_geometries()
        self.vis.add_geometry(pcd, reset_bounding_box=reset_bounding_box)

        for center, R, extent, track_id in objects:
            obb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
            obb.color = (0, 0, 0)  # generate_colors(track_id)  # (0, 1, 0)  # Green color

            self.vis.add_geometry(obb, reset_bounding_box=reset_bounding_box)

            text_mesh = create_text_3d(str(track_id), center + np.array([0, 0, extent[2] / 2 + 0.5]))
            # text_mesh.paint_uniform_color((1, 0, 0))  # Red text
            self.vis.add_geometry(text_mesh, reset_bounding_box=reset_bounding_box)

            # here visualize the track_id text at the center location

    def different_pcd(self, step):
        if self.current_index + step in self.shared_dict.keys():
            self.current_index = self.current_index + step
            self.load_current_pcd()

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


def load_one_frame(args):
    current_index, time_sync_df, labels, root_folder, sequence, calib_data, lidar_index_mapping = args
    calib_data = redo_calib(calib_data)

    this_step_info = time_sync_df[time_sync_df.columns[current_index]]

    this_ts = round(int(this_step_info['timestamp_ms']) * 0.001, 2)
    if this_ts not in labels.keys():
        return (current_index, None)

    extracted_data = extract_one_timestep_information(
        this_step_info, root_folder, sequence, calib_data, lidar_index_mapping, camera=False)

    pcd_for_lidar = np.concatenate([extracted_data[t]['data'] for t in extracted_data.keys() if t.endswith('_lidar')])

    intensity_color = matplotlib.colormaps['gist_rainbow'](pcd_for_lidar[:, 4] / max(pcd_for_lidar[:, 4]))[:, :3]
    time_offset_color = matplotlib.colormaps['gist_rainbow'](pcd_for_lidar[:, 5] / max(pcd_for_lidar[:, 5]))[:, :3]
    lidar_index_color = matplotlib.colormaps['gist_rainbow'](pcd_for_lidar[:, 6] / max(pcd_for_lidar[:, 6]))[:, :3]

    objects = []

    for obj in labels[this_ts]:
        psi = obj['orientation']
        track_id = obj['track_id']
        gRv = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

        objects.append([obj['position'], gRv, obj['dimension'], track_id])

    return (current_index, [pcd_for_lidar[:, :3], [intensity_color, time_offset_color, lidar_index_color], objects])


def main_process(root_folder, labels_folder, sequence):
    (
        crossing_camera_folders, crossing_lidar_folders,
        vehicle1_camera_folders, vehicle1_lidar_folders, vehicle1_state_folder,
        vehicle2_camera_folders, vehicle2_lidar_folders, vehicle2_state_folder,
        calib_data, time_sync_df
    ) = extract_separate_source_folders_from_root_folder(root_folder, sequence)

    labels_file = os.path.join(labels_folder, sequence + '.json')
    if not os.path.isfile(labels_file):
        print(f'Label file {labels_file} not found!')
        return

    with open(labels_file, 'r') as f:
        labels = json.load(f)
    labels = convert_labels_from_tracks_wise_to_timestamp_wise(labels)

    lidar_index_mapping, vehicle_states_id_mapping = get_additional_dataset_information()

    num_frames = len(time_sync_df.columns)

    args_list = [
        (idx, time_sync_df, labels, root_folder, sequence, calib_data, lidar_index_mapping)
        for idx in range(num_frames)
    ]

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(load_one_frame, args_list)

    results_dict = {idx: data for idx, data in results if data is not None}

    visualizer = PcdVisualizer(results_dict)
    visualizer.run()


def main():
    root_folder = r"/path/to/dataset"
    labels_folder = r"/path/to/labels"

    sequence = "20241126_0024_crossing1_18"

    main_process(root_folder, labels_folder, sequence)

if __name__ == "__main__":
    freeze_support()
    main()
