"""
Utility functions related to rgb camera
"""
import os
import concurrent.futures
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict

from opencood.utils import box_utils


def generate_key(scenario_folder, cav_id, timestamp, camera_name):
    return f"{scenario_folder}:{cav_id}:{timestamp}:{camera_name}"


def load_rgb_from_files(camera_list, camera_container=None):
    """
    Given the path to the four cameras file, load them into a dictionary.

    Parameters
    ----------
    camera_list : list
        The list contains all camera file locations.

    Returns
    -------
    camera_dict : dict
        The dictionary containing all rgb images.
    """
    if camera_container is not None:
        # get scenario folder name
        scenario_folder_split = camera_list[0].split('/')
        timestamp = scenario_folder_split[-1].split('_')[0]
        cav_id = scenario_folder_split[-2]
        scenario_folder = scenario_folder_split[-3]

    camera_dict = OrderedDict()
    for (i, camera_file) in enumerate(camera_list):
        camera_name = 'camera%d' % i
        if camera_container is not None:
            key = generate_key(scenario_folder, cav_id, timestamp, camera_name)
            if key in camera_container:
                camera_dict[camera_name] = camera_container[key]
            else:
                image = cv2.imread(camera_file)
                camera_dict[camera_name] = image
                camera_container[key] = image
        else:
            camera_dict[camera_name] = cv2.imread(camera_file)

    return camera_dict


def project_3d_to_camera(objects, intrinsic, extrinsic):
    """
    Project objects under LiDAR coordinate to camera space.

    Parameters
    ----------
    objects : np.ndarray
         Objects 3D coordinates under LiDAR frame: (N, 8, 3).
    intrinsic : np.ndarray
        Camera intrinsics.
    extrinsic : np.ndarray
        LiDAR to camera extrinsic.
    Returns
    -------
    bbx_camera_3d : np.ndarray
        The object position under camera coordinate frame, (N, 8, 3)
    """
    bbx_camera_3d = np.zeros_like(objects)

    for i in range(objects.shape[0]):
        # shape: (3, 8)
        object_ = objects[i].T
        # Add an extra 1.0 at the end of each corner so it becomes of
        # shape (4, 8) and it can be multiplied by a (4, 4) extrinsic matrix.
        object_ = np.r_[
            object_, [np.ones(object_.shape[1])]]

        object_in_camera = np.dot(extrinsic, object_)

        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y

        # (x, y ,z) -> (y, -z, x)
        point_in_camera_coords = np.array([
            object_in_camera[1],
            object_in_camera[2] * -1,
            object_in_camera[0]])
        point_in_camera_coords = np.dot(intrinsic, point_in_camera_coords)

        # normalize x, y, z
        point_in_camera_coords = np.array([
            point_in_camera_coords[0, :] / point_in_camera_coords[2, :],
            point_in_camera_coords[1, :] / point_in_camera_coords[2, :],
            point_in_camera_coords[2, :]])

        bbx_camera_3d[i] = point_in_camera_coords.T

    return bbx_camera_3d


def project_3d_to_camera_torch(objects, intrinsic, extrinsic):
    """
    Project objects under LiDAR coordinate to camera space.

    Parameters
    ----------
    objects : torch.Tensor
         Objects 3D coordinates under LiDAR frame: (N, 8, 3).
    intrinsic : torch.Tensor
        Camera intrinsics.
    extrinsic : torch.Tensor
        LiDAR to camera extrinsic.
    Returns
    -------
    bbx_camera_3d : torch.Tensor
        The object position under camera coordinate frame, (N, 8, 3)
    """
    device = objects.device
    if not isinstance(intrinsic, torch.Tensor):
        intrinsic = torch.tensor(intrinsic, device=device, dtype=torch.float32)
    if not isinstance(extrinsic, torch.Tensor):
        extrinsic = torch.tensor(extrinsic, device=device, dtype=torch.float32)

    bbx_camera_3d = torch.zeros_like(objects)

    for i in range(objects.shape[0]):
        # shape: (3, 8)
        object_ = objects[i].T
        # Add an extra 1.0 at the end of each corner so it becomes of
        # shape (4, 8) and it can be multiplied by a (4, 4) extrinsic matrix.
        object_ = torch.cat(
            [object_, torch.ones(object_.shape[1], device=device).unsqueeze(0)])

        object_in_camera = torch.mm(extrinsic, object_)

        # New we must change from UE4's coordinate system to an "standard"
        # camera coordinate system (the same used by OpenCV):

        # ^ z                       . z
        # |                        /
        # |              to:      +-------> x
        # | . x                   |
        # |/                      |
        # +-------> y             v y

        # (x, y ,z) -> (y, -z, x)
        point_in_camera_coords = torch.stack([
            object_in_camera[1],
            object_in_camera[2] * -1,
            object_in_camera[0]])
        point_in_camera_coords = torch.mm(intrinsic, point_in_camera_coords)

        # normalize x, y, z
        point_in_camera_coords = torch.stack([
            point_in_camera_coords[0, :] / point_in_camera_coords[2, :],
            point_in_camera_coords[1, :] / point_in_camera_coords[2, :],
            point_in_camera_coords[2, :]])

        bbx_camera_3d[i] = point_in_camera_coords.T

    return bbx_camera_3d


def p3d_to_p2d_bb(p3d_bb):
    """
    Draw 2d bounding box(4 vertices) from 3d bounding box(8 vertices). 2D
    bounding box is represented by two corner points.

    Parameters
    ----------
    p3d_bb : np.ndarray
        The 3d bounding box is going to project to 2d.

    Returns
    -------
    p2d_bb : np.ndarray
        Projected 2d bounding box.

    """
    min_x = np.amin(p3d_bb[:, 0])
    min_y = np.amin(p3d_bb[:, 1])
    max_x = np.amax(p3d_bb[:, 0])
    max_y = np.amax(p3d_bb[:, 1])
    p2d_bb = np.array([[min_x, min_y], [max_x, max_y]])
    return p2d_bb


def filter_bbx_out_scope(objects, image_w, image_h):
    """
    Filter out the objects whose coordinates are out of the image scope.

    Parameters
    ----------
    objects : np.ndarray
        The object coordinates under camera coordinate frame. (N, 8, 3)
    image_w : int
        Image width.
    image_h : int
        Image height.

    Returns
    -------
    Remaining bounding boxes.
    """

    # remove the objects that is out of the camera scope.
    points_in_canvas_mask = \
        (objects[:, :, 0] > 0.0) & (objects[:, :, 0] < image_w) & \
        (objects[:, :, 1] > 0.0) & (objects[:, :, 1] < image_h) & \
        (objects[:, :, 2] > 0.0)
    points_in_canvas_mask = np.any(points_in_canvas_mask, axis=1)
    filtered_objects = objects[points_in_canvas_mask]

    return filtered_objects, points_in_canvas_mask


def filter_bbx_out_scope_torch(objects, image_w, image_h):
    """
    Filter out the objects whose coordinates are out of the image scope.

    Parameters
    ----------
    objects : torch.Tensor
        The object coordinates under camera coordinate frame. (N, 8, 3)
    image_w : int
        Image width.
    image_h : int
        Image height.

    Returns
    -------
    Remaining bounding boxes.
    """

    # remove the objects that is out of the camera scope.
    points_in_canvas_mask = \
        (objects[:, :, 0] > 0.0) & (objects[:, :, 0] < image_w) & \
        (objects[:, :, 1] > 0.0) & (objects[:, :, 1] < image_h) & \
        (objects[:, :, 2] > 0.0)
    points_in_canvas_mask = torch.any(points_in_canvas_mask, dim=1)
    filtered_objects = objects[points_in_canvas_mask]

    return filtered_objects, points_in_canvas_mask


def draw_2d_bbx(image, objects, color=(255, 0, 0), thickness=2):
    """
    Given a rgb image and its corresponding camera parameters, draw 2D
    bounding boxes on it from the corrdinates of the 3d objects.

    Parameters
    ----------
    image : np.ndarray
        The camera image that is to be drawn.

    objects : np.ndarray
        Objects 3D coordinates under camera frame: (N, 8, 3).

    color : tuple
        Bbx draw color.

    thickness : int
        Draw thickness.

    Returns
    -------
    The output image drawn with 2d bbx and the 2d bbx.
    """
    image_w = image.shape[1]
    image_h = image.shape[0]
    output_image = image.copy()

    # object_2d_coords = np.zeros((objects.shape[0], 2, 3))
    filtered_objects, mask = filter_bbx_out_scope(objects, image_w, image_h)

    for i in range(filtered_objects.shape[0]):
        object_3d_coords = filtered_objects[i]
        object_2d_coord = p3d_to_p2d_bb(object_3d_coords)
        # object_2d_coords[i] = object_2d_coord

        cv2.rectangle(output_image,
                      (int(object_2d_coord[0, 0]), int(object_2d_coord[0, 1])),
                      (int(object_2d_coord[1, 0]), int(object_2d_coord[1, 1])),
                      color, thickness)

    return output_image, filtered_objects, mask


def draw_3d_bbx(image, objects, color=(0, 255, 0), thickness=2):
    """
    Project the 3D bbox on 2D plane and draw on input image.

    Parameters
    ----------
    image : np.ndarray
        The camera image that is to be drawn.

    objects : np.ndarray
        Objects 3D coordinates under camera frame: (N, 8, 3).

    color : tuple
        Bbx draw color.

    thickness : int
        Draw thickness.

    Returns
    -------
    The output image drawn with 3d bbx.
    """
    if objects.shape[0] == 0:
        return image

    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                    (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

    output_image = image.copy()
    image_w = image.shape[1]
    image_h = image.shape[0]
    objects_filtered, _ = filter_bbx_out_scope(objects, image_w, image_h)

    rect_corners = np.array(objects_filtered[:, :, :2], dtype=np.int)

    for i in range(rect_corners.shape[0]):
        corners = rect_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(output_image, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    return output_image


def plot_agent(draw_image_list):
    """
    Use matplotlib to plot all camera images from a certain agent.

    Parameters
    ----------
    draw_image_list : list
        The images with bbx drawn.
    """
    f, axarr = plt.subplots(1, len(draw_image_list), figsize=(20, 20))

    for i in range(len(draw_image_list)):
        axarr[i].imshow(cv2.cvtColor(draw_image_list[i], cv2.COLOR_RGB2BGR))

    plt.show()


def plot_all_agents(draw_image_list, cav_id, save_dir=None):
    """
    Draw all gents camera images with bbx.

    Parameters
    ----------
    draw_image_list : list
        Each element is another list containing a certain agent's camera.

    cav_id : list

    """
    max_subplots = max([len(draw_image_list[i]) for i in range(len(draw_image_list))])
    fig, axarr = plt.subplots(len(draw_image_list), max_subplots)
    fig.set_size_inches(36, 12)

    for i in range(len(draw_image_list)):
        for j in range(len(draw_image_list[i])):
            if len(draw_image_list) == 1:
                axarr[j].imshow(cv2.cvtColor(draw_image_list[i][j],
                                             cv2.COLOR_RGB2BGR))
                axarr[j].set_title('%s, cam%d' % (cav_id[i], j))
                axarr[j].axis('off')
            else:
                axarr[i, j].imshow(cv2.cvtColor(draw_image_list[i][j],
                                                cv2.COLOR_RGB2BGR))
                axarr[i, j].set_title('%s, cam%d' % (cav_id[i], j))
                axarr[i, j].axis('off')
    
    # save to disk
    if save_dir:
        plt.savefig(save_dir)
    plt.show()


# Function to load images from paths
def load_images_from_path(cam_paths):
    images = {}
    for cam_pth, name in cam_paths:
        image = cv2.imread(cam_pth)
        if image is not None:
            key = generate_image_key_from_path(cam_pth)
            images[key] = image
        else:
            print(f"Warning: {cam_pth} could not be loaded.")
    return images


# Helper function to generate key from path
def generate_image_key_from_path(path):
    parts = path.split('/')
    folder = parts[-3]
    cav_id = parts[-2]
    timestamp = parts[-1].split('_')[0]
    name = parts[-1].split('_')[-1].replace('.png', '')
    return generate_image_key(folder, cav_id, timestamp, name)


# Main function to load images into a container
def load_images_into_container(root_dir, all_yamls):
    all_images = {}

    def prepare_paths(folder, cav_id, timestamp):
        paths = []
        base_path = os.path.join(root_dir, folder, str(cav_id))
        for cam_idx in range(4):
            cam_pth = os.path.join(base_path, f'{timestamp}_camera{cam_idx}.png')
            paths.append((cam_pth, f'camera{cam_idx}'))
        return paths

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for folder in all_yamls:
            full_path = os.path.join(root_dir, folder)
            for cav_id in all_yamls[folder]:
                for timestamp in all_yamls[folder][cav_id]:
                    cam_paths = prepare_paths(folder, cav_id, timestamp)
                    futures.append(executor.submit(load_images_from_path, cam_paths))

        for future in concurrent.futures.as_completed(futures):
            images = future.result()
            all_images.update(images)

    return all_images


def generate_image_key(scenario_folder, cav_id, timestamp, camera_name):
    return f"{scenario_folder}:{cav_id}:{timestamp}:{camera_name}"
