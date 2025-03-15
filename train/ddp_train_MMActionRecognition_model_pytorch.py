import os
import random
import torch
import numpy as np
from torch import nn
import torch.multiprocessing as mp

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from mmcv import load
from functools import partial

from actionModels.rgb_pose_action_recognition_model import RGBPoseAttentionActionRecognizer

from core.evalMetrics import EvalMetrics
from core.log_buffer import LogBuffer
from ddp.trainer import RGBPoseTrainer
import core.utils as utils
from dataPrep.dataPrepMultiModality import CustomPoseDataset
 


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes

    Returns:
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


num_classes_dict = {'ntu60_xsub': 60, 'ntu60_xview': 60, 'ntu120_xsub': 120, 'ntu120_xset': 120,
                    'toyota_xsub': 31, 'toyota_xview2': 19, 'ucla_xview': 10, 'pku_xsub': 51, 'pku_xview': 51}


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def main(rank, world_size, total_epoches, bsz, learning_rate, save_every, work_dir, dataset, evaluation_protocol):
    ddp_setup(rank, world_size)

    batch_size = bsz

    print(f"{evaluation_protocol}: {ann_file_val}")
    train_ntu_annos = load(ann_file_train)
    validation_ntu_annos = load(ann_file_val)

    training_data = CustomPoseDataset(dataset, train_ntu_annos, mode='train', pose_input='joint')
    test_data = CustomPoseDataset(dataset, validation_ntu_annos, mode='test', pose_input='joint')

    # =====increase num of workers=============================#
    seed = 100
    num_workers = 0
    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    train_dataLoader = DataLoader(training_data, batch_size, shuffle=False, pin_memory=True, num_workers=num_workers,
                                  sampler=DistributedSampler(training_data), collate_fn=default_collate,
                                  worker_init_fn=init_fn)
    test_dataLoader = DataLoader(test_data, batch_size, shuffle=False, num_workers=num_workers,
                                 sampler=DistributedSampler(test_data), collate_fn=default_collate,
                                 worker_init_fn=init_fn)

    utils.set_random_seed(seed=seed, deterministic=True)
    # Constructing the DDP model
    model = RGBPoseAttentionActionRecognizer(attention='CBAM_spatial_efficient_temporal',
                                             backbone_type='poseX3dTShiftSE', num_classes=num_classes)
    # Converting your model to use torch.nn.SyncBatchNorm
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.init_weights()
    # optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = None
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                          weight_decay=0.0003)

    steps = len(train_dataLoader)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps * total_epoches)

    log_buffer = LogBuffer()
    test_log_buffer = LogBuffer()
    evalMetric = EvalMetrics()
    logger = utils.getLogger(work_dir)
    print("Start Trainer===")
    trainer = RGBPoseTrainer(model, train_dataLoader, test_dataLoader, optimizer, scheduler, loss_fn, evalMetric, rank,
                             log_buffer, test_log_buffer, save_every, 1, logger, work_dir, dataset, evaluation_protocol,
                             mode='multi-gpus', pretrained=True)
    trainer.train(max_epochs=total_epoches)
    # trainer.do_inference()
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    batch_size = 12
    num_gpus = 2
    initial_lr = (num_gpus * batch_size * 0.001) / 16
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--save_every', type=int, default=1, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=batch_size, type=int,
                        help='Input batch size on each device (default: 32)')
    parser.add_argument('--total_epochs', default=10, type=int, help='Total number of epoches')
    parser.add_argument('--learning_rate', default=round(initial_lr, 4), type=float)
    parser.add_argument('--work_dir',
                        default='/ProposedFramework/X3dTShift_RGB_X3dTShiftPose_SE_16F_CBAM_spatial_efficient_temporal_skip_connection_alignment_NTU120_XSub')
    parser.add_argument("--dataset", type=str, default='ntu120', help="dataset")
    parser.add_argument("--evaluation_protocol", type=str, default='xsub', help="evaluation_protocol")
    print("round(initial_lr,4)", round(initial_lr, 4))
    args = parser.parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)

    num_classes = num_classes_dict[args.dataset + '_' + args.evaluation_protocol]
    if args.dataset == 'ntu60':
    
        if args.evaluation_protocol == 'xsub':
            ann_file_train = '/skeleton/ntu60_xsub_train.pkl'
            ann_file_val = '/skeleton/ntu60_xsub_val.pkl'
        else:
            ann_file_train = '/skeleton/ntu60_xview_train.pkl'
            ann_file_val = '/skeleton/ntu60_xview_val.pkl'
    elif args.dataset == 'ntu120':
        if args.evaluation_protocol == 'xsub':
            ann_file_train = '/skeleton/ntu120_xsub_train.pkl'
            ann_file_val = '/skeleton/ntu120_xsub_val.pkl'
        else:
            ann_file_train = '/skeletonntu120_xset_train.pkl'
            ann_file_val = '/skeleton/ntu120_xset_val.pkl'
    elif args.dataset == 'toyota':
        if args.evaluation_protocol == 'xsub':
            ann_file_train = '/SmartHomeDataset/train_CS.pkl'
            ann_file_val = '/SmartHomeDataset/test_CS.pkl'
        else:
            ann_file_train = '/SmartHomeDataset/train_CV2.pkl'
            ann_file_val = '/SmartHomeDataset/test_CV2.pkl'
    
    elif args.dataset == 'pku':
        if args.evaluation_protocol == 'xsub':
            ann_file_train = '/PKU-MMD/pku_xsub_train.pkl'
            ann_file_val = '/PKU-MMD/pku_xsub_test.pkl'
        else:
            ann_file_train = '/PKU-MMD/pku_xview_train.pkl'
            ann_file_val = '/PKU-MMD/pku_xview_test.pkl'
    
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(
    world_size, args.total_epochs, args.batch_size, args.learning_rate, args.save_every, args.work_dir, args.dataset,
    args.evaluation_protocol), nprocs=world_size)
