# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.data_utils.post_processor.bev_postprocessor import BevPostprocessor
from opencood.data_utils.post_processor.ciassd_postprocessor import CiassdPostprocessor
from opencood.data_utils.post_processor.fpvrcnn_postprocessor import FpvrcnnPostprocessor
from opencood.data_utils.post_processor.camera_bev_postprocessor import CameraBevPostprocessor

__all__ = {
    'VoxelPostprocessor': VoxelPostprocessor,
    'BevPostprocessor': BevPostprocessor,
    'CiassdPostprocessor': CiassdPostprocessor,
    'FpvrcnnPostprocessor': FpvrcnnPostprocessor,
    'CameraBevPostprocessor': CameraBevPostprocessor
}


def build_postprocessor(anchor_cfg, class_names, train):
    process_method_name = anchor_cfg['core_method']
    error_message = f"{process_method_name} is not found. " \
                     f"Please add your post-processor file's name in opencood/" \
                     f"data_utils/post_processor/init.py"
    assert process_method_name in __all__.keys(), \
        error_message
    
    anchor_generator = __all__[process_method_name](
        anchor_params=anchor_cfg,
        class_names=class_names,
        train=train
    )

    return anchor_generator
