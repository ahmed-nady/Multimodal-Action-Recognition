import copy

data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_val'
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'

file_client_args = dict(io_backend='disk')
train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='DecordDecode'),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSampleFrames',
        clip_len=48,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')]
dataset_type = 'VideoDataset'


from mmaction.datasets.pipelines import Compose
vid_pipeline = Compose(train_pipeline)
vid_valid_pipeline = Compose(val_pipeline)

import torch
from torch.utils.data import Dataset
import numpy as np
num_classes =60


import os.path as osp
from typing import Callable, List, Optional, Union

def list_from_file(annot_file):
    with open(annot_file,'r') as f:
        annotations = f.readlines()
    return annotations
def load_data_list(ann_file,data_prefix) -> List[dict]:
    """Load annotation file to get video information."""
    #exists(ann_file)
    data_list = []
    fin = list_from_file(ann_file)
    for line in fin:
        line_split = line.strip().split(' ')
        filename, label = line_split
        label = int(label)
        if data_prefix is not None:
            filename = osp.join(data_prefix, filename)
        data_list.append(dict(filename=filename, label=label))
    return data_list
class CustomPoseDataset(Dataset):
    def __init__(self,data_list,mode):
        self.data_list = copy.deepcopy(data_list)
        self.mode =mode
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        fake_anno = copy.deepcopy(self.data_list[idx])
        fake_anno['start_index'] = 0
        #fake_anno['modality'] = 'Pose'
        if self.mode == 'train':
            network_input = vid_pipeline(fake_anno)['imgs']

        else:
            network_input = vid_valid_pipeline(fake_anno)['imgs']

        action_label = fake_anno['label']
        onehot_encoding = np.zeros((num_classes,))
        onehot_encoding[action_label]=1.

        return network_input,onehot_encoding



