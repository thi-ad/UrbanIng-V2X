# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from opencood.data_utils.datasets.lidar.late_fusion_dataset import LidarLateFusionDataset
from opencood.data_utils.datasets.lidar.early_fusion_dataset import LidarEarlyFusionDataset
from opencood.data_utils.datasets.lidar.intermediate_fusion_dataset import LidarIntermediateFusionDataset

__all__ = {
    'LidarLateFusionDataset': LidarLateFusionDataset,
    'LidarEarlyFusionDataset': LidarEarlyFusionDataset,
    'LidarIntermediateFusionDataset': LidarIntermediateFusionDataset
}

# The communication range for cavs
COM_RANGE = 70


def build_dataset(dataset_cfg, visualize=False, train=True, **kwargs):
    dataset_name = dataset_cfg['fusion']['core_method']
    error_message = f"{dataset_name} is not found. " \
                    f"Please add your processor file's name in opencood/" \
                    f"data_utils/datasets/init.py"
    assert dataset_name in __all__.keys(), error_message

    dataset = __all__[dataset_name](
        params=dataset_cfg,
        visualize=visualize,
        train=train,
        **kwargs
    )

    return dataset
