# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib


"""
3D Anchor Generator for Voxel
"""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from opencood.data_utils.post_processor.base_postprocessor \
    import BasePostprocessor
from opencood.utils import box_utils
from opencood.utils.box_overlaps import bbox_overlaps
from opencood.visualization import vis_utils


class VoxelPostprocessor(BasePostprocessor):
    def __init__(self, anchor_params, class_names, train):
        super(VoxelPostprocessor, self).__init__(anchor_params, class_names, train)
        anchor_generator_config = anchor_params['anchor_args'][
            'anchor_generator_config']
        self.order = anchor_params['order']
        self.anchor_generator_config = anchor_generator_config
        # The order in the list will correspond to the order of class names in anchor_generator_config
        self.anchor_sizes = [config['anchor_sizes'] for config in
                             anchor_generator_config]
        self.anchor_rotations = [config['anchor_rotations'] for config in
                                 anchor_generator_config]
        self.anchor_heights = [config['anchor_bottom_heights'] for config in
                               anchor_generator_config]
        self.align_center = [config.get('align_center', False) for config in
                             anchor_generator_config]
        self.anchor_class_names = [config['class_name'] for config in
                                   anchor_generator_config]
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_config:
            self.matched_thresholds[config['class_name']] = config[
                'matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config[
                'unmatched_threshold']


        assert len(self.anchor_sizes) == len(self.anchor_rotations) == len(
            self.anchor_heights)
        # Need to update all the anchor number
        self.num_of_anchor_sets = len(self.anchor_sizes)

        W = anchor_params['anchor_args']['W']  # x-axis
        H = anchor_params['anchor_args']['H']  # y-axis
        self.grid_size = np.array([W, H])
        # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.cav_lidar_range = anchor_params['anchor_args']['cav_lidar_range']


    def generate_anchor_box(self):
        grid_sizes = [self.grid_size[:2] // config['feature_map_stride'] for
                      config in self.anchor_generator_config]
        all_anchors = []
        num_anchors_per_location = []
        for grid_size, anchor_size, anchor_rotation, anchor_height, align_center in zip(
                grid_sizes, self.anchor_sizes, self.anchor_rotations,
                self.anchor_heights, self.align_center):
            num_anchors_per_location.append(
                len(anchor_rotation) * len(anchor_size) * len(anchor_height))
            if align_center:
                x_stride = (self.cav_lidar_range[3] - self.cav_lidar_range[
                    0]) / grid_size[0]
                y_stride = (self.cav_lidar_range[4] - self.cav_lidar_range[
                    1]) / grid_size[1]
                x_offset, y_offset = x_stride / 2, y_stride / 2
            else:
                x_stride = (self.cav_lidar_range[3] - self.cav_lidar_range[
                    0]) / (grid_size[0] - 1)
                y_stride = (self.cav_lidar_range[4] - self.cav_lidar_range[
                    1]) / (grid_size[1] - 1)
                x_offset, y_offset = 0, 0

            # (grid_sizes[1], )
            x_shifts = np.arange(self.cav_lidar_range[0] + x_offset,
                                 self.cav_lidar_range[3] + 1e-5, step=x_stride)
            # (grid_sizes[0], )
            y_shifts = np.arange(self.cav_lidar_range[1] + y_offset,
                                 self.cav_lidar_range[4] + 1e-5, step=y_stride)
            z_shifts = np.array(anchor_height)

            num_anchor_size = len(anchor_size)
            num_anchor_rotation = len(anchor_rotation)

            anchor_rotation = np.array(anchor_rotation)
            anchor_size = np.array(anchor_size)
            # [x_grid, y_grid, z_grid] indexing may need double check
            x_shifts, y_shifts, z_shifts = np.meshgrid(x_shifts, y_shifts,
                                                       z_shifts)
            anchors = np.concatenate([x_shifts, y_shifts, z_shifts], axis=-1)

            # (x_grid,y_grid, 3) -- l w h
            anchor_size = np.tile(anchor_size.reshape(1, -1, 3),
                                  (*anchors.shape[0:2], 1))

            if self.order == 'hwl':
                anchor_size = anchor_size[..., [2, 1, 0]]
            elif self.order == 'lhw':
                anchor_size = anchor_size[..., [0, 2, 1]]
            else:
                sys.exit('Unknown bbx order.')
            # (x_grid, y_grid, 6) -- [x, y, z, h, w, l]
            anchors = np.concatenate((anchors, anchor_size), axis=-1)
            # (x_grid, y_grid, 2, 6)
            anchors = np.tile(anchors[:, :, None, :],
                              (1, 1, num_anchor_rotation, 1))
            anchor_rotation = np.tile(anchor_rotation.reshape(1, 1, -1, 1), (
            *anchors.shape[0:2], num_anchor_size, 1))
            # (x_grid, y_grid, 2, 7) -- [x, y, z, h, w, l, yaw]
            anchors = np.concatenate([anchors, anchor_rotation], axis=-1)
            # # If shifted to box centers; disabled now
            # anchors[..., 2] += anchors[..., 5] / 2  # shift to box centers
            all_anchors.append(anchors)

        return all_anchors, num_anchors_per_location


    def generate_label(self, **kwargs):
        """
        Generate targets for training.

        Parameters
        ----------
        argv : list
            gt_box_center:(max_num, 7), anchor:(H, W, anchor_num, 7)

        Returns
        -------
        label_dict : dict
            Dictionary that contains all target related info.
        """
        assert self.params['order'] == 'hwl', 'Currently Voxel only support' \
                                              'hwl bbx order.'
        # (max_num, 8) -- x, y, z, dx, dy, dz, yaw, class
        gt_box_center_all = kwargs['gt_box_center']
        # (H, W, anchor_num, 7)
        anchors_list = kwargs['anchors']

        num_anchors_per_location = kwargs['num_anchors_per_location']
        # (max_num)
        masks = kwargs['mask']

        gt_box_center_all = gt_box_center_all[masks == 1]
        box_cls_labels = []
        box_reg_targets = []
        for i, (anchor_class_name, anchors, anchor_num) in enumerate(zip(self.anchor_class_names, anchors_list, num_anchors_per_location)):
            # Assume the class name order in gt is the same as the one in anchor definition
            gt_box_center = gt_box_center_all[gt_box_center_all[:, -1] -1 == i]
            # (H, W)
            feature_map_shape = anchors.shape[:2]
            # (H*W*anchor_num, 7)
            anchors = anchors.reshape(-1, 7)
            # normalization factor, (H * W * anchor_num)
            anchors_d = np.sqrt(anchors[:, 4] ** 2 + anchors[:, 5] ** 2)

            # (H, W, 2)
            labels = np.ones((*feature_map_shape, anchor_num)) * -1

            pos_equal_one = np.zeros((*feature_map_shape, anchor_num))
            neg_equal_one = np.zeros((*feature_map_shape, anchor_num))
            # # (H, W, self.anchor_num * 7)
            # targets = np.zeros((*feature_map_shape, anchor_num * 7))
            # (n, 8)
            # gt_box_center_valid = gt_box_center[masks == 1]
            gt_box_center_valid = gt_box_center
            # (n, 1)
            gt_box_class_valid = gt_box_center_valid[:, -1:]
            # (n, 8, 3)
            gt_box_corner_valid = \
                box_utils.boxes_to_corners_3d(gt_box_center_valid[:, :7],
                                              self.params['order'])
            # (H*W*anchor_num, 8, 3)
            anchors_corner = \
                box_utils.boxes_to_corners_3d(anchors,
                                              order=self.params['order'])
            # (H*W*anchor_num, 4)
            anchors_standup_2d = \
                box_utils.corner2d_to_standup_box(anchors_corner)
            # (n, 4)
            gt_standup_2d = \
                box_utils.corner2d_to_standup_box(gt_box_corner_valid)

            # (H*W*anchor_n)
            iou = bbox_overlaps(
                np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
                np.ascontiguousarray(gt_standup_2d).astype(np.float32),
            )

            # the anchor boxes has the largest iou across
            # shape: (n)
            id_highest = np.argmax(iou.T, axis=1)
            # [0, 1, 2, ..., n-1]
            id_highest_gt = np.arange(iou.T.shape[0])
            # make sure all highest iou is larger than 0
            mask = iou.T[id_highest_gt, id_highest] > 0
            id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

            # find anchors iou > params['pos_iou']
            id_pos, id_pos_gt = \
                np.where(iou > self.matched_thresholds[anchor_class_name])
            #  find anchors iou < params['neg_iou']
            id_neg = np.where(np.sum(iou <
                                     self.unmatched_thresholds[anchor_class_name],
                                     axis=1) == iou.shape[1])[0]
            id_pos = np.concatenate([id_pos, id_highest])
            id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])
            id_pos, index = np.unique(id_pos, return_index=True)
            id_pos_gt = id_pos_gt[index]
            id_neg.sort()

            # cal the target and set the equal one
            index_x, index_y, index_z = np.unravel_index(
                id_pos, (*feature_map_shape, anchor_num))
            pos_equal_one[index_x, index_y, index_z] = 1
            labels[index_x, index_y, index_z] = gt_box_center[id_pos_gt, -1]
            # (H, W, anchor_num, 7)
            targets = np.zeros((*feature_map_shape, anchor_num, 7))
            # calculate the targets
            targets[index_x, index_y, np.array(index_z), 0] = \
                (gt_box_center[id_pos_gt, 0] - anchors[id_pos, 0]) / anchors_d[
                    id_pos]
            targets[index_x, index_y, np.array(index_z), 1] = \
                (gt_box_center[id_pos_gt, 1] - anchors[id_pos, 1]) / anchors_d[
                    id_pos]
            targets[index_x, index_y, np.array(index_z), 2] = \
                (gt_box_center[id_pos_gt, 2] - anchors[id_pos, 2]) / anchors[
                    id_pos, 3]
            targets[index_x, index_y, np.array(index_z), 3] = np.log(
                gt_box_center[id_pos_gt, 3] / anchors[id_pos, 3])
            targets[index_x, index_y, np.array(index_z), 4] = np.log(
                gt_box_center[id_pos_gt, 4] / anchors[id_pos, 4])
            targets[index_x, index_y, np.array(index_z), 5] = np.log(
                gt_box_center[id_pos_gt, 5] / anchors[id_pos, 5])
            targets[index_x, index_y, np.array(index_z), 6] = (
                    gt_box_center[id_pos_gt, 6] - anchors[id_pos, 6])


            index_x, index_y, index_z = np.unravel_index(
                id_neg, (*feature_map_shape, anchor_num))
            neg_equal_one[index_x, index_y, index_z] = 1
            labels[index_x, index_y, index_z] = 0

            # to avoid a box be pos/neg in the same time
            index_x, index_y, index_z = np.unravel_index(
                id_highest, (*feature_map_shape, anchor_num))
            neg_equal_one[index_x, index_y, index_z] = 0

            index_x, index_y, index_z = np.unravel_index(
                id_pos, (*feature_map_shape, anchor_num))
            labels[index_x, index_y, index_z] = gt_box_center[id_pos_gt, -1]
            # seems like neg_equal_one is never used throught the optimization

            box_cls_labels.append(labels)
            box_reg_targets.append(targets)

        label_dict = {
            # [(H,W,anchor_num), (H,W,anchor_num)] -> (H, W, anchor_num * num_class)
            'pos_equal_one': np.concatenate(box_cls_labels, axis=-1),
            # [(H,W,anchor_num, 7), (H,W,anchor_num, 7)] -> (H, W, anchor_num * num_class, 7)
            'targets': np.concatenate(box_reg_targets, axis=-2),
            'neg_equal_one': neg_equal_one
        }

        return label_dict

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]['pos_equal_one'])
            neg_equal_one.append(label_batch_list[i]['neg_equal_one'])
            targets.append(label_batch_list[i]['targets'])

        pos_equal_one = \
            torch.from_numpy(np.array(pos_equal_one))
        neg_equal_one = \
            torch.from_numpy(np.array(neg_equal_one))
        targets = \
            torch.from_numpy(np.array(targets))

        return {'targets': targets,
                'pos_equal_one': pos_equal_one,
                'neg_equal_one': neg_equal_one}

    def post_process(self, data_dict, output_dict, gt_range, projection=True):
        """
        Process the outputs of the model to 2D/3D bounding box.
        Step1: convert each cav's output to bounding box format
        Step2: project the bounding boxes to ego space.
        Step:3 NMS

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box3d_tensor : torch.Tensor
            The prediction bounding box tensor after NMS.
        gt_box3d_tensor : torch.Tensor
            The groundtruth bounding box tensor.
        """
        # the final bounding box list
        pred_box3d_list = []
        pred_box2d_list = []
        pred_label_list = []
        unprojected_box3d_list = []

        for cav_id, cav_content in data_dict.items():
            if cav_id not in output_dict:
                continue
            # the transformation matrix to ego space
            transformation_matrix = cav_content['transformation_matrix']

            # (num_class, H, W, anchor_num, 7)
            all_anchors = cav_content['all_anchors']
            # (H, W, num_class, anchor_num, 7)
            all_anchors = all_anchors.permute(1,2,0,3,4).contiguous()
            # (H*W*num_class*anchor_num, 7)
            all_anchors = all_anchors.view(-1, all_anchors.shape[-1])
            num_anchors = all_anchors.shape[0]

            num_anchors_per_location = cav_content['num_anchors_per_location']

            # classification probability
            # (B, num_anchor*num_class*num_class, H, W)
            prob = output_dict[cav_id]['psm']
            batch_size = prob.shape[0]
            # (B, H, W, num_anchor*num_class*num_class)
            prob = F.sigmoid(prob.permute(0, 2, 3, 1))
            # (B, H*W*num_anchor*num_class, num_class)
            prob = prob.reshape(batch_size, num_anchors, -1)
            # (B, H*W*num_anchor*num_class)
            cls_pred, label_preds = torch.max(prob, dim=-1)
            # class is 1-indexed; 0 is background
            label_preds += 1

            # regression map
            reg = output_dict[cav_id]['rm']
            # (B, H, W, num_anchor*num_class*7)
            reg = reg.permute(0, 2, 3, 1).contiguous()
            # (B, H, W, num_anchor*num_class, 7)
            # reg = reg.reshape(*reg.shape[:3], sum(num_anchors_per_location), -1)
            reg = reg.view(batch_size, num_anchors, -1)

            # convert regression map back to bounding box
            # (N, H*W*num_anchor*num_class, 7)
            batch_box3d = self.delta_to_boxes3d(reg, all_anchors, channel_swap=False)
            mask = \
                torch.gt(cls_pred, self.params['target_args']['score_threshold'])
            mask = mask.view(1, -1)
            mask_reg = mask.unsqueeze(2).repeat(1, 1, 7)

            # during validation/testing, the batch size should be 1
            assert batch_box3d.shape[0] == 1
            boxes3d = torch.masked_select(batch_box3d[0],
                                          mask_reg[0]).view(-1, 7)
            # (num_filtered_predicted_box)
            scores = torch.masked_select(cls_pred[0], mask[0])
            # (num_filtered_predicted_box)
            label_preds = torch.masked_select(label_preds[0], mask[0])

            # convert output to bounding box
            if len(boxes3d) != 0:
                # (N, 8, 3)
                boxes3d_corner = \
                    box_utils.boxes_to_corners_3d(boxes3d,
                                                  order=self.params['order'])
                unprojected_box3d_list.append(boxes3d_corner.clone())

                # (N, 8, 3)
                projected_boxes3d = \
                    box_utils.project_box3d(boxes3d_corner,
                                            transformation_matrix)

                # convert 3d bbx to 2d, (N,4)
                projected_boxes2d = \
                    box_utils.corner_to_standup_box_torch(projected_boxes3d)
                # (N, 5)
                boxes2d_score = \
                    torch.cat((projected_boxes2d, scores.unsqueeze(1)), dim=1)

                pred_box2d_list.append(boxes2d_score)
                pred_box3d_list.append(projected_boxes3d)
                pred_label_list.append(label_preds)

        if len(pred_box2d_list) == 0 or len(pred_box3d_list) == 0:
            return None, None
        # shape: (N, 5)
        pred_box2d_list = torch.vstack(pred_box2d_list)
        # scores
        scores = pred_box2d_list[:, -1]
        # predicted 3d bbx
        pred_box3d_tensor = torch.vstack(pred_box3d_list)
        # predicted labels
        pred_label_tensor = torch.cat(pred_label_list)
        # remove large bbx
        keep_index_1 = box_utils.remove_large_pred_bbx(pred_box3d_tensor)
        keep_index_2 = box_utils.remove_bbx_abnormal_z(pred_box3d_tensor)
        keep_index = torch.logical_and(keep_index_1, keep_index_2)
        assert keep_index.sum().cpu() == pred_box3d_tensor.shape[0]

        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        scores = scores[keep_index]
        pred_label_tensor = pred_label_tensor[keep_index]

        unprojected_box3d_tensor = torch.vstack(unprojected_box3d_list)
        unprojected_box3d_tensor = unprojected_box3d_tensor[keep_index]
        # nms
        keep_index = box_utils.nms_rotated(pred_box3d_tensor,
                                           scores,
                                           self.params['nms_thresh']
                                           )
        pred_box3d_tensor = pred_box3d_tensor[keep_index]
        unprojected_box3d_tensor = unprojected_box3d_tensor[keep_index]

        # select cooresponding score
        scores = scores[keep_index]

        # Select corresponding labels
        pred_label_tensor = pred_label_tensor[keep_index]

        # filter out the prediction out of the range.
        mask = \
            box_utils.get_mask_for_boxes_within_range_torch(pred_box3d_tensor, gt_range=gt_range)
        pred_box3d_tensor = pred_box3d_tensor[mask, :, :]
        unprojected_box3d_tensor = unprojected_box3d_tensor[mask, :, :]
        scores = scores[mask]
        pred_label_tensor = pred_label_tensor[mask]
        if not projection:
            pred_box3d_tensor = unprojected_box3d_tensor
        assert scores.shape[0] == pred_box3d_tensor.shape[0]

        score_labels = torch.cat([scores.unsqueeze(1), pred_label_tensor.unsqueeze(1)], dim=1)

        return pred_box3d_tensor, score_labels

    @staticmethod
    def delta_to_boxes3d(deltas, anchors, channel_swap=True):
        """
        Convert the output delta to 3d bbx.

        Parameters
        ----------
        deltas : torch.Tensor
            (N, W, L, 14)
        anchors : torch.Tensor
            (W, L, 2, 7) -> xyzhwlr
        channel_swap : bool
            Whether to swap the channel of deltas. It is only false when using
            FPV-RCNN

        Returns
        -------
        box3d : torch.Tensor
            (N, W*L*2, 7)
        """
        # batch size
        N = deltas.shape[0]
        if channel_swap:
            deltas = deltas.permute(0, 2, 3, 1).contiguous().view(N, -1, 7)
        else:
            deltas = deltas.contiguous().view(N, -1, 7)

        boxes3d = torch.zeros_like(deltas)
        if deltas.is_cuda:
            device = deltas.device
            anchors = anchors.to(device)
            boxes3d = boxes3d.to(device)

        # (W*L*2, 7)
        anchors_reshaped = anchors.view(-1, 7).float()
        # the diagonal of the anchor 2d box, (W*L*2)
        anchors_d = torch.sqrt(
            anchors_reshaped[:, 4] ** 2 + anchors_reshaped[:, 5] ** 2)
        anchors_d = anchors_d.repeat(N, 2, 1).transpose(1, 2)
        anchors_reshaped = anchors_reshaped.repeat(N, 1, 1)

        # Inv-normalize to get xyz
        boxes3d[..., [0, 1]] = torch.mul(deltas[..., [0, 1]], anchors_d) + \
                               anchors_reshaped[..., [0, 1]]
        boxes3d[..., [2]] = torch.mul(deltas[..., [2]],
                                      anchors_reshaped[..., [3]]) + \
                            anchors_reshaped[..., [2]]
        # hwl
        boxes3d[..., [3, 4, 5]] = torch.exp(
            deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
        # yaw angle
        boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]

        return boxes3d

    @staticmethod
    def visualize(pred_box_tensor, gt_tensor, pcd, map_lidar, show_vis, save_path, dataset=None):
        """
        Visualize the prediction, ground truth with point cloud together.

        Parameters
        ----------
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        pcd : torch.Tensor
            PointCloud, (N, 4).

        show_vis : bool
            Whether to show visualization.

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        """
        vis_utils.visualize_single_sample_output_gt(pred_box_tensor,
                                                    gt_tensor,
                                                    pcd,
                                                    map_lidar,
                                                    show_vis,
                                                    save_path)
