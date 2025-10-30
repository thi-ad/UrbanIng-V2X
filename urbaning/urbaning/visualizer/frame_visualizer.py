from typing import Tuple, Dict, Optional, Union

import cv2
import numpy as np

from urbaning.data import Frame, LLMap
from urbaning.data.info import frame_level_pos_info, vehicle_cameras, crossing_cameras, vehicle_colors

from .bev_visualizer import BEVVisualizer
from .camera_data_visualizer import CameraDataVisualizer


class FrameVisualizer:
    def __init__(
            self,
            image_size: Tuple[int, int]=(1920, 1080),
            bev_extent: float = 100,
            bev_offset=None,
            bev_rotation: float = 0,
    ):
        self.width, self.height = image_size

        w = (self.width // 6)
        h = (self.height // 5)
        position_information = {k: {'x': x * w, 'y': y * h, 'w': w, 'h': h} for k, (x, y) in frame_level_pos_info.items()}
        position_information['bev'] = {'x': 2 * w, 'y': 2 * h, 'w': 4 * w, 'h': 3 * h}

        self.position_information = position_information

        self.bev_visualizer = BEVVisualizer(
            image_size=(4 * w, 3 * h),
            extent=bev_extent,
            offset=bev_offset,
            rotation=bev_rotation,
        )
        self.camera_visualizers: Dict[str, CameraDataVisualizer] = {
            **{k: CameraDataVisualizer() for this_vehicle_cameras in vehicle_cameras.values() for k in this_vehicle_cameras},
            **{k: CameraDataVisualizer() for this_crossing_cameras in crossing_cameras.values() for k in this_crossing_cameras},
        }
        self.current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.frame: Optional[None, Frame] = None

    def reset(self):
        self.current_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.bev_visualizer.reset()
        for camera_vis in self.camera_visualizers.values():
            camera_vis.reset()
        self.frame = None

    def update_frame(self, frame: Frame):
        self.reset()
        self.frame = frame

    def plot_vehicle_camera_data(self):
        for vehicle in self.frame.vehicles.values():
            for camera_name, camera_data in vehicle.cameras.items():
                self.camera_visualizers[camera_name].plot_camera_data(camera_data)

    def plot_infrastructure_camera_data(self):
        for infrastructure in self.frame.infrastructures.values():
            for camera_name, infrastructure_data in infrastructure.cameras.items():
                self.camera_visualizers[camera_name].plot_camera_data(infrastructure_data)

    def plot_camera_data(self):
        self.plot_vehicle_camera_data()
        self.plot_infrastructure_camera_data()

    def plot_lanelet_map_in_vehicle_images(self, **kwargs):
        for vehicle in self.frame.vehicles.values():
            for camera_name in vehicle.cameras:
                self.camera_visualizers[camera_name].plot_lanelet_map(**kwargs)

    def plot_lanelet_map_in_infrastructure_images(self, **kwargs):
        for infrastructure in self.frame.infrastructures.values():
            for camera_name in infrastructure.cameras:
                self.camera_visualizers[camera_name].plot_lanelet_map(**kwargs)

    def plot_lanelet_map_in_bev(self, **kwargs):
        self.bev_visualizer.plot_lanelet_map(**kwargs)

    def plot_lanelet_map(self, **kwargs):
        self.plot_lanelet_map_in_vehicle_images(**kwargs)
        self.plot_lanelet_map_in_infrastructure_images(**kwargs)
        self.plot_lanelet_map_in_bev(**kwargs)

    def plot_point_cloud_in_vehicle_images(self, **kwargs):
        for vehicle in self.frame.vehicles.values():
            for camera_name in vehicle.cameras:
                self.camera_visualizers[camera_name].plot_point_cloud(source=self.frame, **kwargs)

    def plot_point_cloud_in_infrastructure_images(self, **kwargs):
        for infrastructure in self.frame.infrastructures.values():
            for camera_name in infrastructure.cameras:
                self.camera_visualizers[camera_name].plot_point_cloud(source=self.frame, **kwargs)

    def plot_point_cloud_in_bev(self, **kwargs):
        self.bev_visualizer.plot_point_cloud(source=self.frame, **kwargs)

    def plot_point_cloud(self, **kwargs):
        self.plot_point_cloud_in_vehicle_images(**kwargs)
        self.plot_point_cloud_in_infrastructure_images(**kwargs)
        self.plot_point_cloud_in_bev(**kwargs)

    def plot_labels_in_vehicle_images(self, **kwargs):
        for vehicle in self.frame.vehicles.values():
            for camera_name in vehicle.cameras:
                self.camera_visualizers[camera_name].plot_labels(
                    object_labels=self.frame.labels.objects_in_global_coordinates(remove_av_labels=True),
                    **kwargs)

    def plot_labels_in_infrastructure_images(self, **kwargs):
        for infrastructure in self.frame.infrastructures.values():
            for camera_name in infrastructure.cameras:
                self.camera_visualizers[camera_name].plot_labels(object_labels=self.frame.labels, **kwargs)

    def plot_labels_in_bev(self, **kwargs):
        self.bev_visualizer.plot_labels(
            object_labels=self.frame.labels.objects_in_global_coordinates(remove_av_labels=True), **kwargs)

    def plot_labels(self, **kwargs):
        self.plot_labels_in_vehicle_images(**kwargs)
        self.plot_labels_in_infrastructure_images(**kwargs)
        self.plot_labels_in_bev(**kwargs)

    def plot_vehicles_in_vehicle_images(self, box_color=None, text=None, **kwargs):
        for vehicle in self.frame.vehicles.values():
            for camera_name in vehicle.cameras:
                for vehicle_name, plot_vehicle in self.frame.vehicles.items():
                    if vehicle_name == vehicle.vehicle_name:
                        continue
                    self.camera_visualizers[camera_name].plot_vehicle(
                        vehicle=plot_vehicle,
                        box_color=box_color or vehicle_colors[vehicle_name],
                        text=text or vehicle_name,
                        **kwargs)

    def plot_vehicles_in_infrastructure_images(self, box_color=None, text=None, **kwargs):
        for infrastructure in self.frame.infrastructures.values():
            for camera_name in infrastructure.cameras:
                for vehicle_name, plot_vehicle in self.frame.vehicles.items():
                    self.camera_visualizers[camera_name].plot_vehicle(
                        vehicle=plot_vehicle,
                        box_color=box_color or vehicle_colors[vehicle_name],
                        text=text or vehicle_name,
                        **kwargs
                    )

    def plot_vehicles_in_bev(self, box_color=None, text=None, **kwargs):
        for vehicle_name, plot_vehicle in self.frame.vehicles.items():
            self.bev_visualizer.plot_vehicle(
                vehicle=plot_vehicle,
                box_color=box_color or vehicle_colors[vehicle_name],
                text=text or vehicle_name,
                **kwargs
            )

    def plot_vehicles(self, **kwargs):
        self.plot_vehicles_in_vehicle_images(**kwargs)
        self.plot_vehicles_in_infrastructure_images(**kwargs)
        self.plot_vehicles_in_bev(**kwargs)


    def result(self):
        for vehicle in self.frame.vehicles.values():
            for camera_name in vehicle.cameras:
                this_image = self.camera_visualizers[camera_name].result()
                pos_dict = self.position_information[camera_name]
                x, y, w, h = pos_dict['x'], pos_dict['y'], pos_dict['w'], pos_dict['h']
                self.current_image[y:y + h, x:x + w] = cv2.resize(this_image, (w, h))

        for infrastructure in self.frame.infrastructures.values():
            for camera_name in infrastructure.cameras:
                this_image = self.camera_visualizers[camera_name].result()
                pos_dict = self.position_information[camera_name]
                x, y, w, h = pos_dict['x'], pos_dict['y'], pos_dict['w'], pos_dict['h']
                self.current_image[y:y + h, x:x + w] = cv2.resize(this_image, (w, h))

        this_image = self.bev_visualizer.result()
        pos_dict = self.position_information['bev']
        x, y, w, h = pos_dict['x'], pos_dict['y'], pos_dict['w'], pos_dict['h']
        self.current_image[y:y + h, x:x + w] = this_image

        return self.current_image