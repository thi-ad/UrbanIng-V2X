# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils
from opencood.utils import box_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, min_distance=None, max_distance=None):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    min_distance : float, optional
        The minimum distance between ego and object. The default is None.
    max_distance : float, optional
        The maximum distance between ego and object. The default is None.
    """
    if min_distance is not None and max_distance is not None:
        # filter out the objects that are too close or too far
        gt_boxes, _ = box_utils.filter_boxes_within_min_max_radius(gt_boxes, min_distance, max_distance)
        det_boxes, det_mask = box_utils.filter_boxes_within_min_max_radius(det_boxes, min_distance, max_distance)
        det_score = det_score[det_mask]

    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        if min_distance is not None and max_distance is not None:
            distance_key = f'{min_distance}-{max_distance}'
            result_stat[distance_key][iou_thresh]['score'] += det_score.tolist()
        else:
            result_stat[iou_thresh]['score'] += det_score.tolist()

    if min_distance is not None and max_distance is not None:
        distance_key = f'{min_distance}-{max_distance}'
        result_stat[distance_key][iou_thresh]['fp'] += fp
        result_stat[distance_key][iou_thresh]['tp'] += tp
        result_stat[distance_key][iou_thresh]['gt'] += gt
    else:
        result_stat[iou_thresh]['fp'] += fp
        result_stat[iou_thresh]['tp'] += tp
        result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou, global_sort_detections):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
        
    iou : float
        The threshold of iou.

    global_sort_detections : bool
        Whether to sort the detection results globally.
    """
    iou_5 = result_stat[iou]

    if global_sort_detections:
        fp = np.array(iou_5['fp'])
        tp = np.array(iou_5['tp'])
        score = np.array(iou_5['score'])

        assert len(fp) == len(tp) and len(tp) == len(score)
        sorted_index = np.argsort(-score)
        fp = fp[sorted_index].tolist()
        tp = tp[sorted_index].tolist()
        
    else:
        fp = iou_5['fp']
        tp = iou_5['tp']
        assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / (gt_total + 1e-9)

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx] + 1e-9)

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, global_sort_detections, save_yaml=True, include_intermediate_data=False):
    dump_dict = {}
    for class_name in result_stat.keys():
        dump_dict[class_name] = dict()
        for distance in result_stat[class_name].keys():
            dump_dict[class_name].update({distance: dict()})
            for iou in result_stat[class_name][distance].keys():
                ap, mrec, mpre = calculate_ap(result_stat[class_name][distance], iou, global_sort_detections)
                # gt_len, tp_len, fp_len = result_stat[class_name][distance][iou]['gt'], \
                #     len(result_stat[class_name][distance][iou]['tp']), len(result_stat[class_name][distance][iou]['fp'])
                dump_dict[class_name][distance].update(
                    {
                        'ap_' + str(iou): ap,
                        # 'gt_' + str(iou): gt_len,
                        # 'tp_' + str(iou): tp_len,
                        # 'fp_' + str(iou): fp_len
                    })
                if include_intermediate_data:
                    dump_dict[class_name][distance].update(
                        {
                            'mrec_' + str(iou): mrec,
                            'mpre_' + str(iou): mpre
                        })

                print('The average precision for class {} and distance {} and iou {} is: {}'.format(class_name, distance, iou, ap))

    # sort keys such that all aps are first, then mrec and mpre

    for key in dump_dict.keys():
        # sort by distance key and sort by iou
        distance_keys = list(dump_dict[key].keys())
        distance_keys.sort(key=lambda x: (x.split('-')[0], x.split('-')[1]))
        dump_dict[key] = {k: dump_dict[key][k] for k in distance_keys}
        # sort by iou key
        for distance_key in dump_dict[key].keys():
            iou_keys = list(dump_dict[key][distance_key].keys())
            iou_keys.sort(key=lambda x: (x.split('_')[0], x.split('_')[1]))
            dump_dict[key][distance_key] = {k: dump_dict[key][distance_key][k] for k in iou_keys}
            # add ground truth count of each distance (and class)
            # same iou key to get the gt count
            first_iou_key = list(result_stat[key][distance_key].keys())[0]
            dump_dict[key][distance_key]['gt_entries'] = result_stat[key][distance_key][first_iou_key]['gt']
            if len(result_stat[key][distance_key][first_iou_key]['tp']) > 0:
                dump_dict[key][distance_key]['tp_entries'] = result_stat[key][distance_key][first_iou_key]['tp'][-1]
            else:
                dump_dict[key][distance_key]['tp_entries'] = 0
            if len(result_stat[key][distance_key][first_iou_key]['fp']) > 0:
                dump_dict[key][distance_key]['fp_entries'] = result_stat[key][distance_key][first_iou_key]['fp'][-1]
            else:
                dump_dict[key][distance_key]['fp_entries'] = 0

        if '0-all' in dump_dict[key].keys():
            dump_dict[key]['0-all'] = dump_dict[key].pop('0-all')

    # add mAP acrross all classes
    APs = {}
    for class_name in dump_dict.keys():
        for distance in dump_dict[class_name].keys():
            if distance not in APs.keys():
                APs[distance] = {}
            for iou in dump_dict[class_name][distance].keys():
                if 'ap_' in iou:
                    if iou.split('_')[1] not in APs[distance].keys():
                        APs[distance][iou.split('_')[1]] = []
                    APs[distance][iou.split('_')[1]].append(dump_dict[class_name][distance][iou])

    mAPs = {}
    for distance in APs.keys():
        mAPs[distance] = {}
        for iou in APs[distance].keys():
            mAPs[distance][iou] = float(np.mean(APs[distance][iou]))
            dump_dict['mAP_' + distance + '_' + iou] = mAPs[distance][iou]
            print('The average precision for distance {} and iou {} is: {}'.format(distance, iou, mAPs[distance][iou]))

    if save_yaml:
        output_file = 'eval_3d_det.yaml' if not global_sort_detections else 'eval_3d_det_global_sort.yaml'
        yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))

    return dump_dict
