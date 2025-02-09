import copy


clip_len = 16

# dataset settings
# dataset_type = 'VideoDataset'
data_root = '/media/hd1/NADY/ActionRecognitionDatasets/NTU60TSMSpatialAlignment224'
data_root_val = '/media/hd1/NADY/ActionRecognitionDatasets/NTU60TSMSpatialAlignment224'
num_classes =120
data_root = '/media/hd1/NADY/ActionRecognitionDatasets/NTU120TSMSpatialAlignment224'
data_root_val = '/media/hd1/NADY/ActionRecognitionDatasets/NTU120TSMSpatialAlignment224'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='UniformSampleFrames', clip_len=clip_len,num_clips=1),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(256, 256)),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW',collapse=True),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSampleFrames',
        clip_len=clip_len,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW',collapse=True),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='UniformSampleFrames',
        clip_len=clip_len,
        num_clips=1,
        test_mode=True),
    dict(type='DecordDecode'),
    #dict(type='Resize', scale=(256, 256)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW',collapse=True),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]


from mmaction.datasets.pipelines import Compose
pipeline = Compose(train_pipeline)
valid_pipeline = Compose(val_pipeline)
import os.path as osp
import torch
from torch.utils.data import Dataset
import numpy as np

class CustomPoseDataset(Dataset):
    def __init__(self,ann_file,mode):
        self.ann_file = ann_file
        self.mode =mode
        self.multi_class = False
        self.video_infos = self.load_annotations()
    def load_annotations(self):
        """Load annotation file to get video information."""
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if data_root is not None:
                    filename = osp.join(data_root, filename)
                video_infos.append(dict(filename=filename, label=label))
        return video_infos
    def __len__(self):
        return len(self.video_infos)
    def __getitem__(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = 'RGB'
        results['start_index'] = 0

        if self.mode == 'train':
            imgs = pipeline(results)['imgs']

        else:
            imgs = valid_pipeline(results)['imgs']

        action_label = results['label']
        onehot_encoding = np.zeros((num_classes,))
        onehot_encoding[action_label]=1.

        return imgs,onehot_encoding



