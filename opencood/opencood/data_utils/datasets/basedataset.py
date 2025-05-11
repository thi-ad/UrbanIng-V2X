# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Basedataset class for all kinds of fusion.
"""

import os
import copy
import math
import random
from collections import OrderedDict

import torch
import numpy as np
import cv2
from torch.utils.data import Dataset

from opencood.utils.pcd_utils import pcd_to_np
from opencood.utils.camera_utils import load_rgb_from_files
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to initialize the
    database and associate the __getitem__ index with the correct timestamp
    and scenario.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the raw point cloud will be saved in the memory
        for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    pre_processor : opencood.pre_processor
        Used to preprocess the raw data.

    post_processor : opencood.post_processor
        Used to generate training labels and convert the model outputs to
        bbx formats.

    data_augmentor : opencood.data_augmentor
        Used to augment data.

    """

    def __init__(self, params, visualize=False, train=True, specific_scenario=None):
        self.params = params
        self.visualize = visualize
        self.train = train

        self.pre_processor = None
        self.post_processor = None

        self.gt_range = params['preprocess']['cav_lidar_range'] if train else params['preprocess']['cav_lidar_range_validation']

        if 'data_augment' in params:
            self.data_augmentor = DataAugmentor(params['data_augment'], train)
        else:
            self.data_augmentor = None

        # if the training/testing include noisy setting
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            # whether to add time delay
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            # localization error
            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']

            # transmission data size
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed'] \
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0

        else:
            self.async_flag = False
            self.async_overhead = 0  # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0  # Mb (Megabits)
            self.transmission_speed = 27  # Mbps
            self.backbone_delay = 0  # ms

        if self.train:
            root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']

        if 'max_cav' not in params['train_params']:
            self.max_cav = 3
        else:
            self.max_cav = params['train_params']['max_cav']
        
        if 'min_lidar_hits_per_object' not in params['train_params']:
            self.min_lidar_hits_per_object = 5
        else:
            self.min_lidar_hits_per_object = params['train_params']['min_lidar_hits_per_object']

        if 'include_infrastructure' in params['train_params']:
            self.include_infrastructure = params['train_params']['include_infrastructure']
        else:
            self.include_infrastructure = False
        
        if 'only_infrastructure' in params['train_params']:
            self.only_infrastructure = params['train_params']['only_infrastructure']
        else:
            self.only_infrastructure = False
        
        if 'specific_ego_id' in params['train_params']:
            self.specific_ego_id = params['train_params']['specific_ego_id']
        else:
            self.specific_ego_id = None
        
        # This is new compared to the original OpenCOOD
        if 'dataset_labels' in params:
            self.dataset_labels = params['dataset_labels']
        else:
            self.dataset_labels = ['vehicles']

        
        # by default, we load lidar, camera and metadata. But users may
        # define additional inputs/tasks
        self.add_data_extension = \
            params['add_data_extension'] if 'add_data_extension' \
                                            in params else []

        self.all_yaml_files = dict()  # gets filled during runtime (key: timestamp, value: yaml content)

        # first load all paths of different scenarios
        self.scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir)
                                   if os.path.isdir(os.path.join(root_dir, x))])
        # resolve symlinks
        self.scenario_folders = [os.path.realpath(x) for x in self.scenario_folders]

        if specific_scenario is not None:
            # filter out the specific scenario
            self.scenario_folders = [x for x in self.scenario_folders if os.path.basename(x) == specific_scenario]

        self.object_types = self._prepare_object_types(params['preprocess']['anchor_generator_config'])
        self.build_class_name2int_map()
        
        self.reinitialize()
    
    def _prepare_object_types(self, class_config):
        final_obj_types = {} # superclass: [subclasses]
        for obj_type in class_config:
            class_name = obj_type['class_name']
            final_obj_types[class_name] = []
            includes_classes = obj_type.get('includes_classes', [])
            if includes_classes:
                for sub_class in includes_classes:
                    final_obj_types[class_name].append(sub_class)
            else:
                final_obj_types[class_name].append(class_name)
        
        # check if no duplicates exists
        all_subclasses = []
        for obj_type in final_obj_types:
            all_subclasses += final_obj_types[obj_type]
        if len(all_subclasses) != len(set(all_subclasses)):
            raise ValueError('Duplicate subclasses found in object types.')
        
        return final_obj_types


    def _load_yaml(self, file_path: str):
        if file_path in self.all_yaml_files:
            return self.all_yaml_files[file_path]
        else:
            yaml_content = load_yaml(file_path)
            self.all_yaml_files[file_path] = yaml_content
            return yaml_content
    
    def build_class_name2int_map(self):
        self.class_name2int = {}
        for i, class_name in enumerate(self.object_types):
            self.class_name2int[class_name] = i + 1

    def reinitialize(self):
        # Structure: {scenario_id : {cav_1 : {timestamp1 : {yaml: path,
        # lidar: path, cameras:list of path}}}}
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            cav_list = sorted([x for x in os.listdir(scenario_folder)
                               if os.path.isdir(
                    os.path.join(scenario_folder, x))])
            
            if self.only_infrastructure:
                cav_list = [x for x in cav_list if int(x) < 0]
            elif self.include_infrastructure:
                # roadside unit data's id is always negative, so here we want to
                # make sure they will be in the end of the list as they shouldn't
                # be ego vehicle.
                if int(cav_list[0]) < 0:
                    cav_list = cav_list[1:] + [cav_list[0]]
            else:
                # remove infrastructure data
                cav_list = [x for x in cav_list if int(x) >= 0]
            
            if self.train:
                random.shuffle(cav_list)

            if self.specific_ego_id:
                # make this id the first one
                if str(self.specific_ego_id) in cav_list:
                    cav_list.remove(str(self.specific_ego_id))
                    cav_list = [str(self.specific_ego_id)] + cav_list
                else:
                    raise ValueError(f"Specific ego id {self.specific_ego_id} not found in scenario {scenario_folder}")
            
            assert len(cav_list) > 0, f'No CAVs in {scenario_folder}'

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    self.scenario_database[i][cav_id][timestamp]['yaml'] = os.path.join(cav_path, timestamp + '.yaml')
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = os.path.join(cav_path, timestamp + '.pcd')
                    self.scenario_database[i][cav_id][timestamp]['cameras'] = self.load_camera_files(cav_path, timestamp)
                    self.scenario_database[i][cav_id][timestamp]['scenario_folder'] = scenario_folder

                    # load extra data
                    for file_extension in self.add_data_extension:
                        file_name = \
                            os.path.join(cav_path,
                                         timestamp + '_' + file_extension)

                        self.scenario_database[i][cav_id][timestamp][
                            file_extension] = file_name

                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    # we regard the agent with the minimum id as the ego
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        return self.retrieve_base_data(idx, load_lidar_data=True)
        # raise NotImplementedError

    def retrieve_by_idx(self, idx):
        """
        Retrieve the scenario index and timstamp by a single idx
        .
        Parameters
        ----------
        idx : int
            Idx among all frames.

        Returns
        -------
        scenario database and timestamp.
        """
        # we loop the accumulated length list to see get the scenario index
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]

        return scenario_database, timestamp_index

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True, load_lidar_data: bool = False, load_camera_data: bool = False):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix. If set to false, meaning when other cavs
            project their LiDAR point cloud to ego, they are projecting to
            past ego pose.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        assert load_lidar_data or load_camera_data, 'At least one of the data should be loaded.'

        # we loop the accumulated length list to see get the scenario index
        if isinstance(idx, int):
            scenario_database, timestamp_index = self.retrieve_by_idx(idx)
        elif isinstance(idx, tuple):
            scenario_database = self.scenario_database[idx[0]]
            timestamp_index = idx[1]
        else:
            import sys
            sys.exit('Index has to be a int or tuple')

        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        # calculate distance to ego for each cav
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()
        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']

            # calculate delay for this vehicle
            timestamp_delay = \
                self.time_delay_calculation(cav_content['ego'])

            if timestamp_index - timestamp_delay <= 0:
                timestamp_delay = timestamp_index
            timestamp_index_delay = max(0, timestamp_index - timestamp_delay)
            timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                            timestamp_index_delay)
            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            # load the corresponding data into the dictionary
            data[cav_id]['params'] = self.reform_param(
                cav_content, ego_cav_content,
                timestamp_key, timestamp_key_delay,
                cur_ego_pose_flag)
            data[cav_id]['camera_params'] = self.reform_camera_param(
                cav_content, ego_cav_content,
                timestamp_key, is_infrastructure=True if int(cav_id) < 0 else False
            )
            if load_lidar_data:
                data[cav_id]['lidar_np'] = pcd_to_np(cav_content[timestamp_key_delay]['lidar'])
            if load_camera_data:
                data[cav_id]['camera_np'] = load_rgb_from_files(cav_content[timestamp_key_delay]['cameras'])
            
            for file_extension in self.add_data_extension:
                # todo: currently not considering delay!
                # output should be only yaml or image
                if '.yaml' in file_extension:
                    _file = load_yaml(cav_content[timestamp_key][file_extension])
                    if _file is not None:
                        data[cav_id][file_extension] = _file
                else:
                    img_file = cv2.imread(cav_content[timestamp_key][file_extension])
                    if img_file is not None:
                        data[cav_id][file_extension] = img_file

        return data

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = os.path.basename(file)

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = \
                    self._load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            if int(cav_id) < 0:
                continue
            cur_lidar_pose = \
                self._load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = \
                math.sqrt((cur_lidar_pose[0] -
                           ego_lidar_pose[0]) ** 2 +
                          (cur_lidar_pose[1] - ego_lidar_pose[1]) ** 2)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content

    @staticmethod
    def find_ego_pose(base_data_dict):
        """
        Find the ego vehicle id and corresponding LiDAR pose from all cavs.

        Parameters
        ----------
        base_data_dict : dict
            The dictionary contains all basic information of all cavs.

        Returns
        -------
        ego vehicle id and the corresponding lidar pose.
        """

        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        return ego_id, ego_lidar_pose

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # in the real mode, time delay = systematic async time + data
            # transmission time + backbone computation time
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            # in the simulation mode, the time delay is constant
            time_delay = np.abs(self.async_overhead)

        # the data is 10 hz for both opv2v and v2x-set
        # todo: it may not be true for other dataset like DAIR-V2X and V2X-Sim
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0

    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        np.random.seed(self.seed)
        xyz_noise = np.random.normal(0, xyz_std, 3)
        ryp_std = np.random.normal(0, ryp_std, 3)
        noise_pose = [pose[0] + xyz_noise[0],
                      pose[1] + xyz_noise[1],
                      pose[2] + xyz_noise[2],
                      pose[3],
                      pose[4] + ryp_std[1],
                      pose[5]]
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose for other CAVs.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.

        timestamp_delay : str
            The delayed timestamp.

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = self._load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = self._load_yaml(cav_content[timestamp_delay]['yaml'])

        cur_ego_params = self._load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = self._load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params['lidar_pose']

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and self.loc_err_flag:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)
            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # find objects in cur_params
        object_dict = self.map_class_name_to_super_class_name(cur_params)
        delay_params['vehicles'] = self.filter_boxes_by_class(object_dict)
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params

    def filter_boxes_by_class(self, object_dict):
        filtered_object_dict = OrderedDict()
        for obj_type, obj in object_dict.items():
            for obj_id in list(obj.keys()):
                obj[obj_id]['obj_type'] = np.array(
                    [self.class_name2int[obj_type]])
                filtered_object_dict[obj_id] = obj[obj_id]
        return filtered_object_dict

    def map_class_name_to_super_class_name(self, cur_params):
        new_object_dict = OrderedDict()
        for obj_type, objects in cur_params.items():
            # Find the superclass name
            for super_class, sub_classes in self.object_types.items():
                if super_class not in new_object_dict:
                    new_object_dict[super_class] = {}
                if obj_type in sub_classes:
                    # Create a new object with the superclass name
                    new_objects = objects.copy()
                    new_object_dict[super_class].update(new_objects)
                    break

        return new_object_dict

    def reform_camera_param(self, cav_content, ego_content, timestamp, is_infrastructure=False):
        """
        Load camera extrinsic and intrinsic into a propoer format. todo:
        Enable delay and localization error.

        Returns
        -------
        The camera params dictionary.
        """
        camera_params = OrderedDict()
        number_of_cameras = len(cav_content[timestamp]['cameras'])
        cav_params = self._load_yaml(cav_content[timestamp]['yaml'])
        ego_params = self._load_yaml(ego_content[timestamp]['yaml'])
        ego_lidar_pose = ego_params['lidar_pose']
        ego_pose = ego_params['true_ego_pos']

        # load each camera's world coordinates, extrinsic (lidar to camera)
        # pose and intrinsics (the same for all cameras).

        for i in range(number_of_cameras):
            camera_coords = cav_params['camera%d' % i]['cords']
            if is_infrastructure:
                # create extrinsic from camera coords (x, y, z, roll, pitch, yaw)
                # camera_extrinsic = self.create_extrinsic_from_coords(camera_coords)
                camera_extrinsic = np.array(
                    cav_params['camera%d' % i]['extrinsic'])
            else:
                camera_extrinsic = np.array(
                    cav_params['camera%d' % i]['extrinsic'])
            camera_extrinsic_to_ego_lidar = x1_to_x2(camera_coords,
                                                     ego_lidar_pose)
            camera_extrinsic_to_ego = x1_to_x2(camera_coords,
                                               ego_pose)

            camera_intrinsic = np.array(
                cav_params['camera%d' % i]['intrinsic'])

            cur_camera_param = {'camera_coords': camera_coords,
                                'camera_extrinsic': camera_extrinsic,
                                'camera_intrinsic': camera_intrinsic,
                                'camera_extrinsic_to_ego_lidar':
                                    camera_extrinsic_to_ego_lidar,
                                'camera_extrinsic_to_ego':
                                    camera_extrinsic_to_ego,
                                'image_path': cav_content[timestamp]['cameras'][i]}
            camera_params.update({'camera%d' % i: cur_camera_param})

        return camera_params

    def create_extrinsic_from_coords(self, camera_coords):
        """
        Create extrinsic matrix from camera coordinates.

        Parameters
        ----------
        camera_coords : list
            x, y, z, roll, pitch, yaw

        Returns
        -------
        Extrinsic matrix.
        """
        # Convert degrees to radians
        x, y, z = camera_coords[:3]
        roll, pitch, yaw = camera_coords[3:]
        roll, pitch, yaw = np.radians([roll, pitch, yaw])
        
        # Rotation Matrices
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        R_y = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        R_z = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Compute final rotation matrix
        R = R_z @ R_y @ R_x  # Matrix multiplication in order: Rz * Ry * Rx

        # Construct extrinsic matrix
        extrinsic = np.eye(4)  # Initialize 4x4 identity matrix
        extrinsic[:3, :3] = R  # Set rotation
        extrinsic[:3, 3] = [x, y, z]  # Set translation
        return extrinsic

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """

        # find all camera files (_camera{id}.png)

        # Initialize an empty list for camera files
        camera_files = []
        
        # Iterate through camera numbers (up to 5)
        for i in range(10):  # 0 to 5 for 6 cameras
            camera_file = os.path.join(cav_path, f'{timestamp}_camera{i}.png')
            if os.path.isfile(camera_file):  # Check if the file exists
                camera_files.append(camera_file)
            else:
                break

        return camera_files

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask,
                flip=None, rotation=None, scale=None):
        """
        Given the raw point cloud, augment by flipping and rotation.

        Parameters
        ----------
        lidar_np : np.ndarray
            (n, 4) shape

        object_bbx_center : np.ndarray
            (n, 8) shape to represent bbx's x, y, z, h, w, l, yaw, class

        object_bbx_mask : np.ndarray
            Indicate which elements in object_bbx_center are padded.
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center[:, :7],
                    'object_bbx_mask': object_bbx_mask,
                    'flip': flip,
                    'noise_rotation': rotation,
                    'noise_scale': scale}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center[:, :7] = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def collate_batch(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for early and late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         map_lidar,
                         show_vis,
                         save_path,
                         dataset=None):
        # visualize the model output
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      map_lidar,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)
