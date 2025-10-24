import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_MAX_THREADS'] = '1'

from collections import defaultdict
import json
import os
from tqdm import tqdm
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multiprocessing import Pool, freeze_support, cpu_count
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import uuid

from utils import (redo_calib, convert_labels_from_tracks_wise_to_timestamp_wise, get_general_folder_information,
                   extract_one_timestep_information, get_additional_dataset_information)


object_type2nuscenes = {
    "Car": "vehicle.car",
    "Van": "vehicle.car",

    "Bus": "vehicle.truck",
    "Truck": "vehicle.truck",
    "OtherVehicle": "vehicle.truck",
    "Trailer": "vehicle.truck",

    "EScooter": "vehicle.bicycle",
    "Motorcycle": "vehicle.bicycle",
    "Cyclist": "vehicle.bicycle",

    "Pedestrian": "human.pedestrian.adult",
    "OtherPedestrians": "human.pedestrian.adult",
    "Animal": "animal",
    "Other": "movable_object.barrier",
}

def process_one_sequence(arg):
    sensor_tokens, category_tokens, visibility_token_dict, root_folder, labels_folder, target_sample_folder, fused_lidar_topic, agent_token, agent_name, sequence, store_raw_data = arg
    scene_token = create_token()
    log_token = create_token()
    
    seq_obj = ExtractNuscenesSequence(
        sensor_tokens, category_tokens, agent_token, scene_token, visibility_token_dict,
        agent_name, root_folder, labels_folder, target_sample_folder,
        sequence, store_raw_data=store_raw_data, fused_lidar_topic=fused_lidar_topic
    )
    seq_sample_tokens = sorted(list(seq_obj.sample_tokens.items()), key=lambda x: x[0])
    
    return {
        "scene_token": scene_token,
        "log_token": log_token,
        "agent_token": agent_token,
        "sequence": sequence + '_' + agent_name,
        "seq_sample_tokens": seq_sample_tokens,
        "sample_table": seq_obj.sample_table,
        "sample_data_table": seq_obj.sample_data_table,
        "ego_pose_table": seq_obj.ego_pose_table,
        "calibrated_sensor_table": seq_obj.calibrated_sensor_table,
        "sample_annotation_table": seq_obj.sample_annotation_table,
        "instance_table": seq_obj.instance_table,
    }

class ExtractCustomNuscenes():
    def __init__(self, root_folder, labels_folder, sequences, target_nuscenes_folder, ego_agents = ["vehicle1"], version='v1.0-trainval', train_seq_names=None, use_multiprocessing=False): # , "vehicle2", "crossing"
        self.root_folder = root_folder
        self.labels_folder = labels_folder
        self.vaL_seq_names = [sequence for sequence in sequences if sequence not in train_seq_names]
        self.target_nuscenes_folder = target_nuscenes_folder 
        self.target_table_folder = os.path.join(self.target_nuscenes_folder, version)
        self.target_sample_folder = os.path.join(self.target_nuscenes_folder, 'samples')
        self.version = version
        self.train_seq_names = train_seq_names
        os.makedirs(self.target_table_folder, exist_ok=True)
        os.makedirs(self.target_sample_folder, exist_ok=True)

        self.use_multiprocessing = use_multiprocessing
        self.ego_agent_table, self.ego_agent_tokens = self.create_agent_information(ego_agents) 
        self.fused_lidar_topic = "fusedlidar"
        self.category_table, self.category_tokens, self.attribute_table, self.attribute_tokens = self.create_category_and_attribute_information()
        self.sensor_table, self.sensor_tokens = self.create_sensor_information()
             
        self.scene_table = []
        self.log_table = []  

        self.sample_table = []
        self.sample_data_table= []
        self.ego_pose_table = []
        self.calibrated_sensor_table = []
        self.sample_annotation_table = []
        self.instance_table = []

        self.visibility_table = [self.create_visibility_entry(dummy=True)]
        self.visibility_token_dict = {vis["level"]: vis["token"] for vis in self.visibility_table}
        
        if version == 'v1.0-trainval':
            print("Process training sequences:")
            self.create_sequence_dependent_information(self.train_seq_names, ego_agents)
            
            print("Process validation sequences:")
            self.create_sequence_dependent_information(self.vaL_seq_names,  ['vehicle1'])

        elif version == 'v1.0-test':
            print("Process test sequences:")
            self.create_sequence_dependent_information(self.vaL_seq_names,  ['vehicle1'])
        else:
            print(f"{version} is invalid. Please utilze 'v1.0-trainval' or 'v1.0-test'")
        
        self.map_table = []
        self.map_table.append(self.create_map_entry(dummy=True))

    def create_sensor_information(self):
        sensor_table = []
        sensor_token_dict = {}
        crossing_cameras, crossing_lidars, vehicle_cameras, vehicle_lidars, _ = get_general_folder_information()
        all_camera_sensor_topics = []
        for crossing_camera in crossing_cameras.values():
            all_camera_sensor_topics += crossing_camera
        for vehicle_camera in vehicle_cameras.values():
            all_camera_sensor_topics += vehicle_camera
        for topic in all_camera_sensor_topics:
            if topic == "none":
                continue
            sensor_token = create_token()
            sensor_token_dict[topic] = sensor_token
            modality = "camera" 
            sensor_table.append(self.create_sensor_entry(sensor_token, topic, modality))
            sensor_path = os.path.join(self.target_sample_folder, topic)
            os.makedirs(sensor_path, exist_ok=True)
        
        sensor_token = create_token()
        sensor_token_dict[self.fused_lidar_topic] = sensor_token
        sensor_table.append(self.create_sensor_entry(sensor_token, self.fused_lidar_topic, "lidar"))
        sensor_path = os.path.join(self.target_sample_folder, self.fused_lidar_topic)
        os.makedirs(sensor_path, exist_ok=True)
        return sensor_table, sensor_token_dict
    
    def create_category_and_attribute_information(self):
        category_table = []
        category_token_dict = {}

        attribute_table = []
        attribute_token_dict = {}
        for sequence in self.vaL_seq_names + self.train_seq_names:
            with open(os.path.join(self.labels_folder, sequence + '.json'), 'r') as f:
                labels_trackwise = json.load(f)
            for track in labels_trackwise['tracks']:
                obj_type = object_type2nuscenes[track['object_type']]
                if obj_type not in category_token_dict.keys():
                    category_token = create_token()
                    category_token_dict[obj_type] = category_token
                    category_table.append(self.create_category_entry(category_token, obj_type)) 
        
        return category_table, category_token_dict, attribute_table, attribute_token_dict

        # initialize attributes?
    def create_agent_information(self, agents):
        ego_agent_table = []
        ego_agent_token = {}
        for agent in agents:
            agent_token = create_token()
            ego_agent_table.append(self.create_agent_entry(agent_token, agent))
            ego_agent_token[agent] = agent_token
        
        return ego_agent_table, ego_agent_token

    def create_sequence_dependent_information(self, sequences, considered_agents):
        store_raw_data = True
        for agent in self.ego_agent_table:
            if not agent['name'] in considered_agents:
                continue
            results = []
            args_list = []
            for sequence in sequences:
                args_list.append((
                    self.sensor_tokens,
                    self.category_tokens,
                    self.visibility_token_dict,
                    self.root_folder,
                    self.labels_folder,
                    self.target_sample_folder,
                    self.fused_lidar_topic,
                    agent["token"],
                    agent["name"],
                    sequence,
                    store_raw_data
                ))
            if self.use_multiprocessing:
                with Pool(processes=cpu_count()) as pool:
                   results = pool.map(process_one_sequence, args_list)
        
            else:
                for arg in args_list:
                    results.append(process_one_sequence(arg))
            for result in results:
                scene_entry = self.create_scene_entry(
                    result["scene_token"], result["log_token"], result["agent_token"],
                    result["sequence"], len(result["seq_sample_tokens"]),
                    result["seq_sample_tokens"][0][1], result["seq_sample_tokens"][-1][1]
                )
                log_entry = self.create_log_entry(result["log_token"], date=result["sequence"].split('_')[0])

                self.scene_table.append(scene_entry)
                self.log_table.append(log_entry)
                self.sample_table += result["sample_table"]
                self.sample_data_table += result["sample_data_table"]
                self.ego_pose_table += result["ego_pose_table"]
                self.calibrated_sensor_table += result["calibrated_sensor_table"]
                self.sample_annotation_table += result["sample_annotation_table"]
                self.instance_table += result["instance_table"]

            store_raw_data = False
 
    def create_visibility_entry(self, dummy=False):
        if dummy:
            return {
                "token": create_token(),
                "level": "1",
                "description": "Fully visible"
            }
        
    def create_map_entry(self, dummy=False):
        if dummy:
            return {
                "category": "",
                "token": create_token(),
                "filename": "",
                "log_tokens": [log["token"] for log in self.log_table],
            }
        
    def create_category_entry(self, category_token, category, description = None):
        return {
            "token": category_token,
            "name": category,
            "description": description,
        }

    def create_sensor_entry(self, sensor_token, topic, modality):
       if modality == None:
            return None
       return {
            "token": sensor_token,
            "channel": topic,
            "modality": modality
        }
    def create_agent_entry(self, agent_token, agent):
        return {
            "token": agent_token,
            "name": agent
        }
    
    def create_scene_entry(self, scene_token, log_token, agent_token, sequence_name, nbr_samples, first_sample_token, last_sample_token):
        return {
            "token": scene_token,
            "log_token": log_token,
            "nbr_samples": nbr_samples,
            "agent_token": agent_token,
            "first_sample_token": first_sample_token,
            "last_sample_token": last_sample_token,
            "name": sequence_name, 
            "description": ""
        }

    def create_log_entry(self, log_token, date):
        return {
            "token": log_token,
            "logfile": "",
            "date_captured": date,
            "location": "Ingolstadt"
        }
                
    def write_table(self, name, data):
        with open(os.path.join(self.target_table_folder, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=2)

    def write_split(self):
        scene_names = [scene['name'] for scene in self.scene_table]
        split_path = os.path.join(self.target_table_folder, "split")
        os.makedirs(split_path, exist_ok=True)
        if "test" in self.version:
            file_name = os.path.join(split_path, 'test.txt')
            with open(file_name, 'w') as f:
                for name in scene_names:
                    f.write(name + '\n')
        else:
            val_scenes = [scene_name for scene_name in scene_names if scene_name not in self.train_seq_names]
            train_scenes = self.train_seq_names 
            with open(os.path.join(split_path, 'train.txt'), 'w') as f:
                for name in train_scenes:
                    f.write(name + '\n')

            with open(os.path.join(split_path, 'val.txt'), 'w') as f:
                for name in val_scenes:
                    f.write(name + '\n')

    def token_exists(self, token, table):
        exists = any([True if entry['token'] == token else False for entry in table])
        return exists
    
    def prev_and_next_exist(self, table):
        for entry in tqdm(table):
            prev_token = entry['prev']
            prev_exists = self.token_exists(prev_token, table) if prev_token != '' else True
            next_token = entry['next']
            next_exists = self.token_exists(next_token, table) if next_token != '' else True
            if not (prev_exists and next_exists):
                return False
        return True

    def first_and_last_ann_of_instance_exist(self):
        for entry in tqdm(self.instance_table):
            first_ann_token = entry["first_annotation_token"]
            first_exists = self.token_exists(first_ann_token, self.sample_annotation_table)
            last_ann_token = entry["last_annotation_token"]
            last_exists = self.token_exists(last_ann_token, self.sample_annotation_table)
            if not (first_exists and last_exists):
                return False
        return True
    
    def test_tables(self):
        ann_references_good = self.prev_and_next_exist(self.sample_annotation_table)
        sample_references_good = self.prev_and_next_exist(self.sample_table)
        sample_data_references_good = self.prev_and_next_exist(self.sample_data_table)

        instances_good = self.first_and_last_ann_of_instance_exist()

        all_references_good = ann_references_good and sample_data_references_good and sample_references_good and instances_good

        return all_references_good
    def write_tables(self):
        tables = {
            "ego_agents": self.ego_agent_table,
            "category": self.category_table,
            "sample": self.sample_table,
            "sample_data": self.sample_data_table,
            "ego_pose": self.ego_pose_table,
            "calibrated_sensor": self.calibrated_sensor_table,
            "sample_annotation": self.sample_annotation_table,
            "instance": self.instance_table,
            "sensor": self.sensor_table,
            "scene": self.scene_table,
            "log": self.log_table,
            "visibility": self.visibility_table,
            "map": self.map_table,
            "attribute": self.attribute_table,
        }
        for name, data in tables.items():
            self.write_table(name, data)

class ExtractNuscenesSequence():
    def __init__(self, sensor_tokens, category_tokens, agent_token, scene_token, visibility_tokens, ego, root_folder, labels_folder, target_sample_folder, sequence, store_raw_data=False, fused_lidar_topic=None):
        self.ego = ego
        self.fused_lidar_topic = fused_lidar_topic
        self.sensor_tokens = sensor_tokens
        self.category_tokens = category_tokens
        self.root_folder = root_folder
        self.labels_folder = labels_folder
        self.target_sample_folder = target_sample_folder
        self.store_raw_data = store_raw_data
        self.sequence = sequence
        self.agent_token = agent_token
        self.scene_token = scene_token
        self.crossing_number = sequence.split('_')[2]
        self.visibility_tokens = visibility_tokens
        self.load_dataset_information()
        self.initialize_tokens_of_sequence()

        self.sample_table = []
        self.sample_data_table = []
        self.ego_pose_table = []
        self.calibrated_sensor_table = []
        self.sample_annotation_table = []
        self.instance_table = []
        self.fill_raw_data_independent_information()
        self.fill_raw_data_dependent_information()

    def initialize_tokens_of_sequence(self):
        self.instance_tokens = {}
        self.sample_annotation_tokens = defaultdict(dict)
        self.calib_tokens = defaultdict(dict)
        self.sample_tokens = {}
        self.sample_data_tokens = defaultdict(dict)
        self.pose_tokens = {}
        self.track_id2category = {}
        for timestamp_ms in self.time_sync_df.loc["timestamp_ms"]:
            labels_of_ts = self.labels_tswise[float(timestamp_ms) / 1000]
            timestamp_ms = int(timestamp_ms)
            for track in labels_of_ts:
                track_id = track['track_id']
                ann_token = create_token()
                self.sample_annotation_tokens[track_id][timestamp_ms] = ann_token
                if track_id not in self.track_id2category.keys():
                    self.track_id2category[track_id]= track['object_type']

        for track_id in self.track_id2category.keys():
            self.instance_tokens[track_id] = create_token()
         
        for timestamp_ms in self.time_sync_df.loc["timestamp_ms"]:
            timestamp_ms = int(timestamp_ms)
            sample_token = create_token()
            self.sample_tokens[timestamp_ms] = sample_token
            pose_token = create_token()
            self.pose_tokens[timestamp_ms] = pose_token

            for sensor in self.sensor_tokens.keys():
                sample_data_token = create_token()
                calib_token = create_token()
                self.sample_data_tokens[sensor][timestamp_ms] = sample_data_token
                self.calib_tokens[sensor][timestamp_ms] = calib_token 

    def fill_raw_data_independent_information(self):
        for track_id, category in self.track_id2category.items():
            obj_type = object_type2nuscenes[category]
            self.instance_table.append(self.create_instance_entry(track_id, obj_type))
        
        for col in self.time_sync_df.columns:
            this_step_info = self.time_sync_df[col] 
            timestamp_ms = int(this_step_info['timestamp_ms']) 
            self.sample_table.append(self.create_sample_entry(timestamp_ms))

    def load_dataset_information(self):
        calibration_file = os.path.join(self.root_folder, self.sequence, 'calibration.json')
        time_sync_info_file = os.path.join(self.root_folder, self.sequence, 'timesync_info.csv')

        with open(calibration_file, 'r') as f:
            self.calib_data = json.load(f)

        self.time_sync_df = pd.read_csv(time_sync_info_file).set_index('Unnamed: 0')

        self.calib_data = redo_calib(self.calib_data)
        with open(os.path.join(self.labels_folder, self.sequence + '.json'), 'r') as f:
            labels_trackwise = json.load(f)
        with open(self.labels_folder + '_av_track_ids.json', 'r') as f:
            self.av_track_id_map = json.load(f)

        crossing_cameras, crossing_lidars, self.vehicle_cameras, self.vehicle_lidars, self.vehicle_states = get_general_folder_information()
        self.crossing_cameras = [camera for camera in crossing_cameras[self.crossing_number] if "crossing" in camera]
        self.crossing_lidars = crossing_lidars[self.crossing_number]
        self.lidar_sources = [lidar for lidar in self.vehicle_lidars["vehicle1"] + self.vehicle_lidars["vehicle2"] + self.crossing_lidars]
        self.lidar_index_mapping, self.vehicle_states_id_mapping = get_additional_dataset_information()
        
        skipped_tracks = [self.av_track_id_map[self.sequence][self.ego]] if "vehicle" in self.ego else []
        
        
        self.labels_tswise = convert_labels_from_tracks_wise_to_timestamp_wise(labels_trackwise, skip_tracks=skipped_tracks) # no tracks skipped 

    def fill_raw_data_dependent_information(self):           
        for col in tqdm(self.time_sync_df.columns, desc= f"Processing all tracks of sequence {self.sequence} of {self.ego}"):
            
            this_step_info = self.time_sync_df[col] 
            frame_timestamp_ms = int(this_step_info['timestamp_ms'])          

            extracted_data = extract_one_timestep_information(
                this_step_info, self.root_folder, self.sequence, self.calib_data, self.lidar_index_mapping)

            gTagent_dict = {key.split('_')[0]: value['gTv'] for key, value in extracted_data.items() if key.endswith('state')}
            gTagent_dict['crossing'] = np.eye(4)

            self.ego_pose_table.append(self.create_pose_entry(frame_timestamp_ms, gTagent_dict[self.ego]))

            crossing_lidar_done = False
            for key, value in extracted_data.items():
                curr_agent = "crossing" if "crossing" in key.split('_')[0] else key.split('_')[0]
                
                if curr_agent not in ["vehicle1", "vehicle2", "crossing"]:
                    continue
                if key.endswith('camera'): 
                    topic = key
                    sensor_root_path = os.path.join(self.target_sample_folder, topic)
                    if 'vehicle' in curr_agent:
                        camera_capture_time = int(this_step_info[key].split('.')[0])
                        with open(os.path.join(self.root_folder, self.sequence, curr_agent + '_state',
                                        str((camera_capture_time // 10) * 10) + '.json'), 'r') as f:
                            vehicle_state_camera_capture_time = json.load(f)

                        agent_cctTg = np.linalg.inv(np.asarray(vehicle_state_camera_capture_time['gTv']))  # cct = camera capture time
                        agent_cctTagent = agent_cctTg @ gTagent_dict[curr_agent]

                        sTagent = self.calib_data[key]['extrinsics']['cTv'] @ agent_cctTagent 
                    else:
                        sTagent = value['cTg']
                    iTc = self.calib_data[key]['intrinsics']['IntrinsicMatrixNew']
                    filename = f"{sensor_root_path}/{self.sequence}_{frame_timestamp_ms:06d}.jpg" 
                    fileformat = "jpg"
                    image_distorted = value['data']
                    image = self.calib_data[key]['intrinsics']['undistort_function'](image_distorted)
                    hw = image.shape[:2]
                    if self.store_raw_data:
                        cv2.imwrite(filename, image)
                    
                elif key.endswith('lidar') and curr_agent == self.ego:
                    if curr_agent == "crossing" and crossing_lidar_done:
                        continue
                    else:
                        crossing_lidar_done = True
                    topic = self.fused_lidar_topic
                    sensor_root_path = os.path.join(self.target_sample_folder, topic)
                    filename = f"{sensor_root_path}/{self.ego}_{self.sequence}_{frame_timestamp_ms:06d}.pcd.bin"
                    fileformat = "pcd"
                    hw = None
                    iTc = None
                    pcd_global = np.concatenate([extracted_data[lidar_source]['data'] for lidar_source in self.lidar_sources])[:, [0, 1, 2, 3, 4, 6]] # xyz, homogoenous_ones, intensity, lidar agent index (og Nuscenes uses channel index instead"
                    egoTg = np.linalg.inv(gTagent_dict[self.ego])
                    sTagent = np.linalg.inv(self.calib_data[key]['extrinsics']['vTl']) if key.startswith('vehicle') else np.eye(4) 
                    pcd_ego_lidar = pcd_global.copy()
                    pcd_ego_lidar[:, :4] = (sTagent @ egoTg @ pcd_ego_lidar[:, :4].T).T 
                    
                    pcd_ego_vis = np.hstack((pcd_ego_lidar[:, :3], pcd_ego_lidar[:, -2:])) 
                    pcd_ego_vis.tofile(filename)
                else:
                    continue
                
                if curr_agent != self.ego:
                    sTego = sTagent @ np.linalg.inv(gTagent_dict[curr_agent]) @ gTagent_dict[self.ego] 
                    egoTs = np.linalg.inv(sTego)
                else:
                    egoTs = np.linalg.inv(sTagent)
                
                self.calibrated_sensor_table.append(self.create_calib_entry(frame_timestamp_ms, topic, egoTs, iTc)) 

                self.sample_data_table.append(self.create_sample_data_entry(frame_timestamp_ms, topic, '/'.join(filename.split('/')[-3:]), fileformat, hw))

            self.create_ann_entries_for_current_ts(frame_timestamp_ms, pcd_global = pcd_global) 
    
    def create_ann_entries_for_current_ts(self, timestamp_ms, pcd_global):
        obj_list = self.labels_tswise[float(timestamp_ms) / 1000]
        for obj in obj_list:
            l, w, h = obj['dimension']
            obj_wlh = [w, l, h]
            object_xyz_gc = obj['position']
            gRobj = Rotation.from_euler('z', obj['orientation']).as_matrix() 
            obj_rot_quaternion = rotation_matrix2nuscenes_quaternion(gRobj) # get quaternion
            track_id = obj['track_id']
            pcd_xyz_object = (pcd_global[:, :3] - object_xyz_gc) @ gRobj
            obj_dimensions = np.array(obj['dimension'])
            min_corner = - obj_dimensions/2
            max_corner =  obj_dimensions/2
            filter_mask = np.all(np.logical_and(pcd_xyz_object >= min_corner, pcd_xyz_object <= max_corner), axis=1)
            pcd_xyz_filtered = pcd_xyz_object[filter_mask]
            self.sample_annotation_table.append(self.create_sample_annotations_entry(object_xyz_gc, obj_wlh, obj_rot_quaternion, track_id, timestamp_ms, num_lidar_pts = len(pcd_xyz_filtered)))

    def create_pose_entry(self, timestamp_ms, gTego):
        return {
            "token": self.pose_tokens[timestamp_ms],
            "timestamp": timestamp_ms,
            "translation": gTego[:-1, -1].tolist(),
            "rotation": rotation_matrix2nuscenes_quaternion(gTego[:-1, :-1]).tolist(),
        }
    def create_calib_entry(self, timestamp_ms, key, sTego, iTc):
        return {
            "token": self.calib_tokens[key][timestamp_ms],
            "sensor_token": self.sensor_tokens[key],
            "translation": sTego[:-1, -1].tolist(),  # Replace with real extrinsics
            "rotation": rotation_matrix2nuscenes_quaternion(sTego[:-1, :-1]).tolist(),  # Quaternion [w, x, y, z]
            "camera_intrinsic": iTc.tolist() if iTc is not None else []
        }      
       
    def create_instance_entry(self, track_id, category):
        sorted_anns = sorted(list(self.sample_annotation_tokens[track_id].items()), key= lambda x: x[0])
        return {
            "token": self.instance_tokens[track_id],
            "category_token": self.category_tokens[category],
            "nbr_annotations": len(self.sample_annotation_tokens[track_id]),
            "first_annotation_token": sorted_anns[0][1],
            "last_annotation_token": sorted_anns[-1][1], # ego dependent because ann is sample_token dependent
        }
    
    def create_sample_annotations_entry(self, obj_xyz, obj_wlh, obj_rot_quaternion, track_id, timestamp_ms, num_lidar_pts):
        
        return { 
            "token": self.sample_annotation_tokens[track_id][timestamp_ms],
            "sample_token": self.sample_tokens[timestamp_ms],
            "instance_token": self.instance_tokens[track_id],
            "visibility_token": self.visibility_tokens["1"], 
            "attribute_tokens": [],
            "translation": obj_xyz, 
            "size": obj_wlh, # nuscenes uses width, length, height
            "rotation": obj_rot_quaternion.tolist(), # Quaternion [w, x, y, z]
            "prev": self.sample_annotation_tokens[track_id].get(timestamp_ms - 100, ''), 
            "next": self.sample_annotation_tokens[track_id].get(timestamp_ms + 100, ''),  # nuscenes: Track is a new instance if it temporarily disappears => next is null if it does not appear in the next Frame
            "num_lidar_pts": num_lidar_pts,
            "num_radar_pts": 0,
        }

    def create_sample_entry(self, timestamp_ms):
        return {
            "token": self.sample_tokens[timestamp_ms],
            "agent_token": self.agent_token, 
            "timestamp": timestamp_ms,
            "prev": self.sample_tokens.get(timestamp_ms - 100, ''),
            "next":  self.sample_tokens.get(timestamp_ms + 100, ''),  
            "scene_token": self.scene_token
        }
    
    def create_sample_data_entry(self, timestamp_ms, key, filename, fileformat, hw):
        return {
            "token": self.sample_data_tokens[key][timestamp_ms],
            "sample_token": self.sample_tokens[timestamp_ms],
            "ego_pose_token": self.pose_tokens[timestamp_ms],
            "calibrated_sensor_token": self.calib_tokens[key][timestamp_ms],
            "filename": filename,
            "fileformat": fileformat, 
            "timestamp": timestamp_ms, 
            "is_key_frame": True, # uncertain about meaning
            "height": hw[0] if hw is not None else None,
            "width": hw[1] if hw is not None else None,
            "next": self.sample_data_tokens[key].get(timestamp_ms + 100, ''),
            "prev": self.sample_data_tokens[key].get(timestamp_ms - 100, '')
        }
    
def create_token():
    return str(uuid.uuid4())

def rotation_matrix2nuscenes_quaternion(R):
    R = Rotation.from_matrix(R).as_quat(scalar_first=True) 
    return R

def main():
    root_folder = r'/path/to/dataset'
    labels_folder = r'/path/to/labels'

    target_nuscenes_folder = os.path.join(os.path.dirname(root_folder), 'nuscenes_format/') 

    os.makedirs(target_nuscenes_folder, exist_ok=True)

    if not os.path.exists(root_folder):
        raise "root_folder does not exist"

    if not os.path.exists(labels_folder):
        raise "labels_folder does not exist"

    #sequences_labels_folder = [l_folder[:-5] for l_folder in os.listdir(labels_folder)]
    #sequences_root_folder = os.listdir(root_folder)
    #sequences = set(sequences_labels_folder) & set(sequences_root_folder)
    # sequence of ESC 1 SPLITS
    TRAIN = ['20241126_0024_crossing1_08', '20241126_0022_crossing1_08', '20241126_0022_crossing1_09', '20241127_0003_crossing1_09', '20241127_0000_crossing1_00', '20241126_0024_crossing1_18', '20241126_0008_crossing1_01', '20241126_0025_crossing1_01', '20241127_0008_crossing1_00', '20241126_0018_crossing1_00', '20241126_0025_crossing1_00', '20241126_0008_crossing1_00', '20241126_0010_crossing2_00', '20241127_0014_crossing2_00', '20241127_0029_crossing2_00', '20241126_0019_crossing2_00', '20241126_0001_crossing2_00', '20241126_0013_crossing2_00', '20241127_0011_crossing3_00', '20241127_0010_crossing3_09', '20241127_0012_crossing3_00']
    VAL = ['20241126_0024_crossing1_19', '20241126_0024_crossing1_09', '20241127_0026_crossing2_08', '20241127_0026_crossing2_09', '20241127_0010_crossing3_08', '20241127_0024_crossing3_08']
    TEST = ['20241126_0014_crossing1_00', '20241126_0017_crossing1_00', '20241127_0003_crossing1_08', '20241126_0004_crossing2_00', '20241127_0025_crossing2_00', '20241127_0009_crossing3_00', '20241127_0024_crossing3_09']
    trainval_sequences = TRAIN + VAL
    CustomNuscenesObject = ExtractCustomNuscenes(root_folder, labels_folder, trainval_sequences, target_nuscenes_folder, version='v1.0-trainval', train_seq_names=TRAIN, use_multiprocessing=False)
    CustomNuscenesObject.write_tables()    
if __name__ == "__main__":
    freeze_support()
    main()
