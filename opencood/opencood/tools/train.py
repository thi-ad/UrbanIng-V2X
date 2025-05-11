# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os
import statistics

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
import torch.distributed as dist
from opencood.utils.seg_utils import cal_iou_training


os.environ['MASTER_PORT'] = '29600'
os.environ['WORLD_SIZE'] = "2"


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str,
                        required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--distributed', action='store_true',
                        help='whether to use distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--gpu', default=0, type=int,
                        help='ID of gpu to use (if available)')
    opt = parser.parse_args()
    return opt


def run_training(hypes_yaml, model_dir=None, seed=0, distributed=False, half=False, gpu=0):
    """
    Run training for the model.

    Parameters
    ----------
    hypes_yaml : str
        yaml file path.
    model_dir : str
        continued training path.
    seed : int
        random seed.
    distributed : bool
        whether to use distributed training.
    half : bool
        whether to use half precision training.
    gpu : int
        gpu id.

    Returns
    -------
    None
    """
    hypes = yaml_utils.load_yaml(hypes_yaml, model_dir)

    if distributed:
        distributed = multi_gpu_utils.init_distributed_mode()

    print('-----------------Seed Setting----------------------')
    seed = train_utils.init_random_seed(None if seed == 0 else seed)
    hypes['train_params']['seed'] = seed
    print('Set seed to %d' % seed)
    train_utils.set_random_seed(seed)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validation_dataset = build_dataset(hypes, visualize=False, train=False)


    if distributed:
        sampler_train = DistributedSampler(opencood_train_dataset, shuffle=True)
        sampler_val = DistributedSampler(opencood_validation_dataset,
                                         shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=16,
                                  collate_fn=opencood_train_dataset.collate_batch)
        val_loader = DataLoader(opencood_validation_dataset,
                                sampler=sampler_val,
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                    batch_size=hypes['train_params']['batch_size'],
                                    num_workers=16,
                                    collate_fn=opencood_train_dataset.collate_batch,
                                    shuffle=True,
                                    pin_memory=True)
        val_loader = DataLoader(opencood_validation_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch,
                                shuffle=False,
                                pin_memory=True)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    if not distributed and gpu:
        device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if model_dir:
        saved_path = model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    # we assume gpu is necessary
    model.to(device)
    
    model_without_ddp = model

    if distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[gpu],
                find_unused_parameters=True
            )
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_scheduler(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    best_val_loss = 1e10

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)

        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        pbar_train = tqdm.tqdm(total=len(train_loader), leave=True)

        # Run training for one epoch
        for i, batch_data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not half:
                output_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(output_dict, batch_data['ego']['label_dict'])
            else:
                with torch.cuda.amp.autocast():
                    output_dict = model(batch_data['ego'])
                    final_loss = criterion(output_dict, batch_data['ego']['label_dict'])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar_train)
            pbar_train.update(1)

            if not half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

        # Run evaluation
        if epoch % hypes['train_params']['eval_freq'] == 0 or epoch == epoches - 1:
            valid_avg_loss = []

            pbar_val = tqdm.tqdm(total=len(val_loader), leave=True)
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])

                    final_loss = criterion(output_dict, batch_data['ego']['label_dict'])

                    valid_avg_loss.append(final_loss.item())

                    pbar_val.update(1)

            valid_avg_loss = statistics.mean(valid_avg_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, valid_avg_loss))
            writer.add_scalar('Validate_Loss', valid_avg_loss, epoch)
        
            if valid_avg_loss < best_val_loss:
                best_val_loss = valid_avg_loss
                torch.save(model_without_ddp.state_dict(),
                           os.path.join(saved_path, 'best_model.pth'))
    
        # Save the model
        if epoch % hypes['train_params']['save_freq'] == 0 or epoch == epoches - 1:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))
    
        # reinitialize the dataset
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)


def main():
    # python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 opencood/tools/train.py --distributed --hypes_yaml /home/dominik/Git_Repos/Private/Ingolstadt_Crossing_Dataset_DL/opencood/hypes_yaml/lidar/thi_dataset/point_pillar_cobevt.yaml
    opt = train_parser()
    hypes_yaml = opt.hypes_yaml
    model_dir = opt.model_dir
    seed = opt.seed
    distributed = opt.distributed
    half = opt.half
    gpu = opt.gpu

    if distributed:
        gpu = opt.gpu
    
    run_training(hypes_yaml, model_dir, seed, distributed, half, gpu)

    

if __name__ == '__main__':
    main()
