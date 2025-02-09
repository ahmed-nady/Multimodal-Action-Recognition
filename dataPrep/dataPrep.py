import copy

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW',collapse=True),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

train_aug_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, 64)),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GroupWiseTranslation'),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW',collapse=True),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    # dict(type='Resize', scale=(-1, 64)),
    # dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False),
    dict(type='FormatShape', input_format='NCTHW',collapse=True),
    dict(type='ToTensor', keys=['imgs'])
]
test_pipeline = [
    dict(
        type='UniformSampleFrames', clip_len=48, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(56, 56), keep_ratio=False),
    # dict(type='Resize', scale=(-1, 64)),
    # dict(type='CenterCrop', crop_size=64),
    dict(
        type='GeneratePoseTarget',
        sigma=0.6,
        use_score=True,
        with_kp=True,
        with_limb=False,
        double=True,
        left_kp=left_kp,
        right_kp=right_kp),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])]
dataset_type = 'PoseDataset'


from mmaction.datasets.pipelines import Compose
skeleton_pipeline = Compose(train_pipeline)
skeleton_aug_pipeline = Compose(train_aug_pipeline)
skeleton_valid_pipeline = Compose(val_pipeline)

import torch
from torch.utils.data import Dataset
import numpy as np
num_classes =60
class CustomPoseDataset(Dataset):
    def __init__(self,annotation_lst,mode,SupCon=False):
        self.annotation_lst = copy.deepcopy(annotation_lst)
        self.mode =mode
        self.SupCon =SupCon
    def __len__(self):
        return len(self.annotation_lst)
    def __getitem__(self, idx):
        fake_anno = copy.deepcopy(self.annotation_lst[idx])
        fake_anno['start_index'] = 0
        fake_anno['modality'] = 'Pose'
        if self.mode == 'train':
            heatmaps_volumes = skeleton_pipeline(fake_anno)['imgs']
            inputs = heatmaps_volumes
            if self.SupCon:
                fake_anno = self.annotation_lst[idx].copy()
                fake_anno['start_index'] = 0
                fake_anno['modality'] = 'Pose'
                aug_heatmaps_volumes = skeleton_aug_pipeline(fake_anno)['imgs']
                inputs = [heatmaps_volumes,aug_heatmaps_volumes]
        else:
            heatmaps_volumes = skeleton_valid_pipeline(fake_anno)['imgs']
            inputs = heatmaps_volumes

        action_label = fake_anno['label']
        onehot_encoding = np.zeros((num_classes,))
        onehot_encoding[action_label]=1.

        if self.SupCon:
            return inputs,action_label
        return inputs,onehot_encoding



