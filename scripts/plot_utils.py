import math

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from utils import generate_colors, get_vPv_numpy


def cv2_plot_3d_bounding_box(corners_orig, image, color, text_scale, thickness, text=None):
    # corners --> 2 x 8
    #  color, thickness --> color, 2

    # corners = corners_orig[:2].astype(int)
    # flt, frt, rrt, rlt, rlb, rrb, frb, flb = corners.T
    # flt, frt, rrt, rlt, rlb, rrb, frb, flb = np.sign(corners_orig[2])

    corners = corners_orig.astype(int)

    flt_x, flt_y, flt_z, \
        frt_x, frt_y, frt_z, \
        rrt_x, rrt_y, rrt_z, \
        rlt_x, rlt_y, rlt_z, \
        rlb_x, rlb_y, rlb_z, \
        rrb_x, rrb_y, rrb_z, \
        frb_x, frb_y, frb_z, \
        flb_x, flb_y, flb_z = corners.T.reshape(-1)

    height, width, _ = image.shape

    def plot_internally(p1_x, p1_y, p1_z, p2_x, p2_y, p2_z):

        point1_is_in = (p1_z > 0) and (0 < p1_x < width) and (0 < p1_y < height)
        point2_is_in = (p2_z > 0) and (0 < p2_x < width) and (0 < p2_y < height)

        if point1_is_in or point2_is_in:
            cv2.line(image, (p1_x, p1_y), (p2_x, p2_y), color, thickness)

    plot_internally(flt_x, flt_y, flt_z, frt_x, frt_y, frt_z)
    plot_internally(frt_x, frt_y, frt_z, rrt_x, rrt_y, rrt_z)
    plot_internally(rrt_x, rrt_y, rrt_z, rlt_x, rlt_y, rlt_z)
    plot_internally(rlt_x, rlt_y, rlt_z, flt_x, flt_y, flt_z)
    plot_internally(flb_x, flb_y, flb_z, frb_x, frb_y, frb_z)
    plot_internally(frb_x, frb_y, frb_z, rrb_x, rrb_y, rrb_z)
    plot_internally(rrb_x, rrb_y, rrb_z, rlb_x, rlb_y, rlb_z)
    plot_internally(rlb_x, rlb_y, rlb_z, flb_x, flb_y, flb_z)
    plot_internally(flt_x, flt_y, flt_z, flb_x, flb_y, flb_z)
    plot_internally(frt_x, frt_y, frt_z, frb_x, frb_y, frb_z)
    plot_internally(rlt_x, rlt_y, rlt_z, rlb_x, rlb_y, rlb_z)
    plot_internally(rrt_x, rrt_y, rrt_z, rrb_x, rrb_y, rrb_z)

    if text is not None:
        pos_x = min(max(math.floor(min(corners[0])), -500), width)
        pos_y = min(max(math.floor(min(corners[1])), -500), height)
        cv2.putText(image, text, (pos_x, pos_y), cv2.FONT_HERSHEY_SIMPLEX, text_scale, color, thickness)


def plot_pcd_in_image_pd(image, pcd_for_image_orig, cTg, iTc):
    height, width, _ = image.shape
    pcd_for_image = pcd_for_image_orig.copy()
    pcd_for_image[['ix', 'iy', 'iz']] = iTc.dot(cTg.dot(pcd_for_image[['x', 'y', 'z', 'ones']].T)[:3]).T

    pcd_for_image = pcd_for_image[pcd_for_image['iz'] > 0]
    pcd_for_image['ix'] = pcd_for_image['ix'] / pcd_for_image['iz']
    pcd_for_image['iy'] = pcd_for_image['iy'] / pcd_for_image['iz']

    pcd_for_image = pcd_for_image[np.logical_and(pcd_for_image['ix'] > 2, pcd_for_image['ix'] < width - 2)]
    pcd_for_image = pcd_for_image[np.logical_and(pcd_for_image['iy'] > 2, pcd_for_image['iy'] < height - 2)]

    image[pcd_for_image['iy'].astype(int), pcd_for_image['ix'].astype(int)] = pcd_for_image[['b', 'g', 'r']]


def plot_pcd_in_image_np(image, pcd_for_image_orig, cTg, iTc):
    height, width, _ = image.shape

    number_of_additional_pixels = round(height / 480)

    pcd_for_image = pcd_for_image_orig.copy()
    pcd_for_image[:, 11:14] = iTc.dot(cTg.dot(pcd_for_image[:, :4].T)[:3]).T

    pcd_for_image = pcd_for_image[pcd_for_image[:, 13] > 0]
    pcd_for_image[:, 11] = pcd_for_image[:, 11] / pcd_for_image[:, 13]
    pcd_for_image[:, 12] = pcd_for_image[:, 12] / pcd_for_image[:, 13]

    pcd_for_image = pcd_for_image[np.logical_and(pcd_for_image[:, 11] > number_of_additional_pixels,
                                                 pcd_for_image[:, 11] < width - number_of_additional_pixels)]
    pcd_for_image = pcd_for_image[np.logical_and(pcd_for_image[:, 12] > number_of_additional_pixels,
                                                 pcd_for_image[:, 12] < height - number_of_additional_pixels)]

    for i in range(number_of_additional_pixels):
        image[pcd_for_image[:, 12].astype(int), pcd_for_image[:, 11].astype(int)] = pcd_for_image[:, [9, 8, 7]]
        if i == 1:
            pass
            # image[pcd_for_image[:, 12].astype(int)-1, pcd_for_image[:, 11].astype(int)-1] = pcd_for_image[:, [9, 8, 7]]
            # image[pcd_for_image[:, 12].astype(int)-1, pcd_for_image[:, 11].astype(int)+1] = pcd_for_image[:, [9, 8, 7]]
            # image[pcd_for_image[:, 12].astype(int)+1, pcd_for_image[:, 11].astype(int)-1] = pcd_for_image[:, [9, 8, 7]]
            # image[pcd_for_image[:, 12].astype(int)+1, pcd_for_image[:, 11].astype(int)+1] = pcd_for_image[:, [9, 8, 7]]


def plot_labels_in_image(frame, object_labels, cTg, K, text_scale, thickness):
    for object in object_labels:
        length, width, height = object['dimension']
        x_gc, y_gc, z_gc = object['position']
        psi = object['orientation']
        obj_type = object['object_type']
        track_id = object['track_id']

        vPv = get_vPv_numpy(length, width, height)

        gRv = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

        gtv = np.array([[x_gc], [y_gc], [z_gc]])

        gPv = np.matmul(gRv, vPv) + gtv
        cPv = np.dot(cTg[:-1, :-1], gPv) + cTg[:-1, -1:]
        iPv = np.dot(K, cPv)

        if np.sum(iPv[-1] > 0) > 1:
            image_points = iPv / np.abs(iPv[-1])
            cv2_plot_3d_bounding_box(image_points, frame, generate_colors(track_id),
                                     text_scale=text_scale, thickness=thickness, text=obj_type + ' ' + str(track_id))


def plot_states_in_image(frame, state_informations, cTg, K, text_scale, thickness):
    for k, v in state_informations.items():
        length, width, height = v['dimension']
        gTv = v['gTv']
        id = v['id']
        name = k.split('_')[0]

        vPv = get_vPv_numpy(length, width, height)
        gRv = gTv[:-1, :-1]
        gtv = gTv[:-1, -1:]
        gPv = np.matmul(gRv, vPv) + gtv
        cPv = np.dot(cTg[:-1, :-1], gPv) + cTg[:-1, -1:]
        iPv = np.dot(K, cPv)

        if np.sum(iPv[-1] > 0) > 1:
            image_points = iPv / np.abs(iPv[-1])
            cv2_plot_3d_bounding_box(image_points, frame, generate_colors(id), text_scale=text_scale,
                                     thickness=thickness, text=name)


def plot_pcd_in_lidar_frame_pd(lidar_frame, pcd_for_lidar, pixels_per_meter, offset):
    height, width, _ = lidar_frame.shape
    pcd_for_lidar[['lx', 'ly']] = pcd_for_lidar[['x', 'y']]
    pcd_for_lidar['ly'] = -pcd_for_lidar['ly']
    pcd_for_lidar[['lx', 'ly']] = (pcd_for_lidar[['lx', 'ly']] * pixels_per_meter + offset).astype(int)

    pcd_for_lidar = pcd_for_lidar[np.logical_and(pcd_for_lidar['lx'] > 0, pcd_for_lidar['ly'] > 0)]
    pcd_for_lidar = pcd_for_lidar[np.logical_and(pcd_for_lidar['lx'] < width, pcd_for_lidar['ly'] < height)]

    lidar_frame[pcd_for_lidar['ly'], pcd_for_lidar['lx']] = pcd_for_lidar[['b', 'g', 'r']]


def plot_pcd_in_lidar_frame_np(lidar_frame, pcd_for_lidar, pixels_per_meter, offset, vTg=None):
    height, width, _ = lidar_frame.shape

    if vTg is None:
        vTg = np.eye(4)

    pcd_for_lidar[..., :4] = pcd_for_lidar[..., :4].dot(vTg.T)

    pcd_for_lidar[:, 14:16] = pcd_for_lidar[:, :2]
    pcd_for_lidar[:, 15] = -pcd_for_lidar[:, 15]
    pcd_for_lidar[:, 14:16] = (pcd_for_lidar[:, 14:16] * pixels_per_meter + offset)

    pcd_for_lidar = pcd_for_lidar[np.logical_and(pcd_for_lidar[:, 14] > 1, pcd_for_lidar[:, 15] > 1)]
    pcd_for_lidar = pcd_for_lidar[np.logical_and(pcd_for_lidar[:, 14] < width - 1, pcd_for_lidar[:, 15] < height - 1)]

    lidar_frame[pcd_for_lidar[:, 15].astype(int), pcd_for_lidar[:, 14].astype(int)] = pcd_for_lidar[:, [9, 8, 7]]


def cv2_plot_2d_bounding_box(lidar_frame, x_gc, y_gc, psi, pixels_per_meter, offset, l, w,
                             box_filled, box_thickness, box_color, text=None, text_scale=None, text_thickness=None, text_color=None):
    # all in global coordinates
    xy, yaw = (np.asarray([[x_gc, -y_gc]]) * pixels_per_meter + offset).astype(int).tolist()[0], -psi
    length, width = l * pixels_per_meter, w * pixels_per_meter

    rotation_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw)],
        [np.sin(yaw), np.cos(yaw)]
    ])

    rect_points = np.array([
        [-length / 2, -width / 2],
        [length / 2, -width / 2],
        [length / 2, width / 2],
        [-length / 2, width / 2]
    ])

    rotated_points = np.dot(rect_points, rotation_matrix.T) + xy
    rotated_points = rotated_points.astype(int)
    pts = rotated_points.reshape((-1, 1, 2))
    if box_filled:
        lidar_frame = cv2.fillPoly(lidar_frame, [pts], color=box_color)
    else:
        lidar_frame = cv2.polylines(lidar_frame, [pts], isClosed=True, color=box_color, thickness=box_thickness)

    arrow_length = int(length * 0.5) + 15  # Arrow length scaled to box size
    arrow_tip = (
        int(xy[0] + arrow_length * np.cos(yaw)),
        int(xy[1] + arrow_length * np.sin(yaw))
    )
    lidar_frame = cv2.line(
        lidar_frame,
        tuple(xy),
        arrow_tip,
        color=box_color,  # Red arrow
        thickness=box_thickness,
        # tipLength=box_thickness / 2.0
    )
    if not text is None:
        cv2.putText(lidar_frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, fontScale=text_scale, color=text_color,
                    thickness=text_thickness, )


def plot_labels_in_lidar_frame(lidar_frame, object_labels, pixels_per_meter, offset, text_scale=None, thickness=None, filled=False, vTg=None, title_picture=False):
    if vTg is None:
        vTg = np.eye(4)

    for object in object_labels:
        gTco = np.eye(4)
        gTco[:-1, :-1] = Rotation.from_euler('xyz', [0, 0, object['orientation']], degrees=False).as_matrix()
        gTco[:-1, -1] = np.array(object['position'])

        vTco = vTg.dot(gTco)

        # x_gc, y_gc, z_gc = object['position']
        # psi = object['orientation']

        x_gc, y_gc, z_gc = vTco[:-1, -1].tolist()
        psi = Rotation.from_matrix(vTco[:-1, :-1]).as_euler('xyz', degrees=False)[2]

        l, w, h = object['dimension']
        obj_type = object['object_type']
        track_id = object['track_id']
        bbox_color = (255, 255, 0) if not title_picture else (0, 100, 0) 
        if text_scale is not None:
            text = obj_type[0:1] + ' ' + str(track_id)
            cv2_plot_2d_bounding_box(
                lidar_frame, x_gc, y_gc, psi, pixels_per_meter, offset, l, w, filled, thickness, bbox_color, text,
                text_scale, thickness, (255, 255, 255))
        else:
            cv2_plot_2d_bounding_box(
                lidar_frame, x_gc, y_gc, psi, pixels_per_meter, offset, l, w, filled, thickness, bbox_color)

def plot_states_in_lidar_frame(lidar_frame, state_informations, pixels_per_meter, offset, text_scale=None, thickness=None, filled=True, vTg=None, title_picture=False):
    if vTg is None:
        vTg = np.eye(4)
    pole_positions = []
    for k, v in state_informations.items():
        if "vehicle" in k:
            length, width, height = v['dimension']
            if title_picture:
                length *= 2.5
                width *= 2.5
            gTv = v['gTv']
            id = v['id']

            # psi = math.atan2(gTv[1, 0], gTv[0, 0])
            # x_gc, y_gc = gTv[0, -1], gTv[1, -1]
            
            vTv = vTg.dot(gTv)

            psi = math.atan2(vTv[1, 0], vTv[0, 0])
            x_gc, y_gc = vTv[0, -1], vTv[1, -1]

            if title_picture:
                color = (0, 0, 139) if 'vehicle1' in k else (139, 0, 0) # RGB dark red and dark blue
            else:
                color =  (0, 255, 0) if 'vehicle1' in k else (0, 255, 255)

            if text_scale is not None:
                name = k.split('_')[0]
                cv2_plot_2d_bounding_box(
                    lidar_frame, x_gc, y_gc, psi, pixels_per_meter, offset, length, width, filled, thickness,
                    color, name, text_scale, thickness, (255, 255, 255))
            else:
                if title_picture:
                    cv2_plot_2d_bounding_box(
                        lidar_frame, x_gc, y_gc, psi, pixels_per_meter, offset, length, width, box_filled=False, box_thickness=2,
                        box_color=(0, 0 ,0))
                cv2_plot_2d_bounding_box(
                    lidar_frame, x_gc, y_gc, psi, pixels_per_meter, offset, length, width, filled, thickness,
                    color)

        else:
            color = (255, 165, 0) # RGB
            radius = 15
            if 'lidar' in k and k.split('_')[1][-1] == "2":
                x_gc, y_gc = v['extrinsics']['gTl'][:-2, -1]
                xy = (np.asarray([[x_gc, -y_gc]]) * pixels_per_meter + offset).astype(int).tolist()[0]
                cv2.circle(lidar_frame, xy, radius, (0, 0, 0), thickness=2)  # Black outline
                cv2.circle(lidar_frame, xy, radius, color, thickness=-1)

            elif '53' in k:
                x_gc, y_gc = - v['extrinsics']['cTg'][:-2, -1]
                xy = (np.asarray([[x_gc, -y_gc]]) * pixels_per_meter + offset).astype(int).tolist()[0]
                cv2.circle(lidar_frame, xy, radius, (0, 0, 0), thickness=2)  # Black outline
                cv2.circle(lidar_frame, xy, radius, color, thickness=-1)
