# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import time
import statistics
from tqdm import tqdm
import torch
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str,
                        required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method',
                        #required=True,
                        type=str,
                        default='intermediate',
                        help='late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis',action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--gpu', type=str, default='cuda:1',
                        help='gpu number to use')
    parser.add_argument('--load_best', action='store_true',
                        help='whether to load the best model')

    opt = parser.parse_args()
    return opt


def run_point_cloud_inference(save_dir, model, data_loader, device, fusion_method, save_results=True,
                              show_vis=False, show_sequence=False, save_vis=False,
                              save_npy=False, global_sort_detections=False):
    dataset = data_loader.dataset

    # create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # returns eval statistics
    distances_to_check = [[0, 100], [0, 30], [30, 50], [50, 100]]

    object_types = dataset.object_types

    ious_to_check = [0.1, 0.2, 0.3, 0.5, 0.7]
    result_stat = {}
    for object_type in object_types:
        result_stat[object_type] = {}
        for distances in distances_to_check:
            distance_key = f'{distances[0]}-{distances[1]}'
            result_stat[object_type][distance_key] = {}
            for iou in ious_to_check:
                result_stat[object_type][distance_key][iou] = {
                    'tp': [],
                    'fp': [],
                    'gt': 0,
                    'score': []
                }

    if show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(100):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    for i, batch_data in tqdm(enumerate(data_loader), total=len(data_loader)):
        with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            if fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_late_fusion(
                        batch_data,
                        model,
                        dataset)
            elif fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_early_fusion(
                        batch_data,
                        model,
                        dataset)
            elif fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                    inference_utils.inference_intermediate_fusion(
                        batch_data,
                        model,
                        dataset)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                        'fusion is supported.')
            
            # visualization purpose
            for class_id, class_name in enumerate(result_stat.keys()):
                class_id += 1
                keep_index_pred = pred_score[:, -1] == class_id
                keep_index_gt = gt_label_tensor == class_id

                preds_class_tensor = pred_box_tensor[keep_index_pred, ...]
                preds_class_score = pred_score[keep_index_pred, 0]
                gt_class_box_tensor = gt_box_tensor[keep_index_gt, ...]
                for distances in result_stat[class_name].keys():
                    min_distance, max_distance = \
                        [d_k for d_k in distances.split('-')]
                    for iou in result_stat[class_name][distances].keys():
                        eval_utils.caluclate_tp_fp(
                            preds_class_tensor,
                            preds_class_score,
                            gt_class_box_tensor,
                            result_stat[class_name],
                            iou,
                            min_distance,
                            max_distance
                        )

            if save_npy:
                npy_save_path = os.path.join(save_dir, 'npy')
                if not os.path.exists(npy_save_path):
                    os.makedirs(npy_save_path)
                inference_utils.save_prediction_gt(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'][0],
                    i,
                    npy_save_path)

            if show_vis or save_vis:
                vis_save_path = ''
                if save_vis:
                    vis_save_path = os.path.join(save_dir, 'vis')
                    if not os.path.exists(vis_save_path):
                        os.makedirs(vis_save_path)
                    vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                dataset.visualize_result(
                    pred_box_tensor,
                    gt_box_tensor,
                    batch_data['ego']['origin_lidar'],
                    None,
                    show_vis,
                    vis_save_path,
                    dataset=dataset)

            if show_sequence:
                pcd, pred_o3d_box, gt_o3d_box = \
                    vis_utils.visualize_inference_sample_dataloader_with_map(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        None,
                        vis_pcd,
                        mode='constant'
                        )
                if i == 0:
                    vis.add_geometry(pcd)
                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_pred,
                                                pred_o3d_box,
                                                update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_gt,
                                                gt_o3d_box,
                                                update_mode='add')

                vis_utils.linset_assign_list(vis,
                                            vis_aabbs_pred,
                                            pred_o3d_box)
                vis_utils.linset_assign_list(vis,
                                            vis_aabbs_gt,
                                            gt_o3d_box)
                vis.update_geometry(pcd)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.001)

    eval_results = eval_utils.eval_final_results(
        result_stat,
        save_dir,
        global_sort_detections,
        save_results
    )

    if show_sequence:
        vis.destroy_window()

    return eval_results


def prepare_inference_dataloader(hypes, specific_scenario = None, **dataloader_kwargs):
    standard_dl_kwargs = {
        'num_workers': 16,
        'shuffle': False,
        'pin_memory': False,
        'drop_last': False
    }
    standard_dl_kwargs.update(dataloader_kwargs)
    standard_dl_kwargs['batch_size'] = 1  # always 1 for inference

    opencood_dataset = build_dataset(hypes, visualize=True, train=False, specific_scenario=specific_scenario)

    is_camera_dataset = True if 'camera' in opencood_dataset.__class__.__name__.lower() else False

    data_loader = DataLoader(
        opencood_dataset,
        collate_fn=opencood_dataset.collate_batch if is_camera_dataset else opencood_dataset.collate_batch_test,
        **standard_dl_kwargs)
    
    return data_loader, is_camera_dataset


def prepare_inference_device(gpu='cuda:0'):
    device = 'cpu' if gpu is None else gpu
    if gpu is not None and torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')
        
    return device


def prepare_inference_model(hypes, device):
    model = train_utils.create_model(hypes)
    model.to(device)

    return model


def load_inference_model_weights(model, saved_path, load_best: bool = False, epoch=None):
    if epoch:
        model = train_utils.load_specific_epoch_model(saved_path, model, epoch)
    else:
        _, model = train_utils.load_saved_model(saved_path, model, load_best=load_best)
    
    model.eval()

    return model


def run_inference(
        model_dir, fusion_method='intermediate', show_vis=False,
        show_sequence=False, save_vis=False,
        save_npy=False, global_sort_detections=False,
        gpu='cuda:0', load_best=False):
    
    assert os.path.exists(model_dir), '%s not found' % model_dir

    assert fusion_method in ['late', 'early', 'intermediate']
    assert not (show_vis and show_sequence), \
        'you can only visualize the results in single image mode or video mode'

    hypes = yaml_utils.load_yaml(None, model_dir)

    data_loader, is_camera_dataset = prepare_inference_dataloader(hypes)

    device = prepare_inference_device(gpu)

    model = prepare_inference_model(hypes, device)

    model = load_inference_model_weights(model, model_dir, load_best=load_best)
    model.to(device)
    model.eval()

    save_dir = os.path.join(model_dir, os.path.basename(hypes['validate_dir']))
    if 'specific_ego_id' in hypes['train_params'] and hypes['train_params']['max_cav'] > 1:
        save_dir = os.path.join(save_dir, str(hypes['train_params']['specific_ego_id']))

    run_point_cloud_inference(
        save_dir, model, data_loader, device, save_results=True,
        fusion_method=fusion_method, show_vis=show_vis,
        show_sequence=show_sequence, save_vis=save_vis,
        save_npy=save_npy, global_sort_detections=global_sort_detections)


def main():
    opt = test_parser()
    model_dir = opt.model_dir
    fusion_method = opt.fusion_method
    show_vis = opt.show_vis
    show_sequence = opt.show_sequence
    save_vis = opt.save_vis
    save_npy = opt.save_npy
    global_sort_detections = opt.global_sort_detections
    gpu = opt.gpu
    load_best = opt.load_best

    run_inference(
        model_dir, fusion_method, show_vis=show_vis,
        show_sequence=show_sequence, save_vis=save_vis,
        save_npy=save_npy, global_sort_detections=global_sort_detections,
        gpu=gpu, load_best=load_best
    )


if __name__ == '__main__':
    main()
