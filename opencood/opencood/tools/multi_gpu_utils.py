import os
import torch
import torch.distributed as dist


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return False

    dist_url = 'env://'

    torch.cuda.set_device(gpu)
    dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, GPU: {}'.format(
        rank, dist_url, gpu), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method='env://',
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)

    return True


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

