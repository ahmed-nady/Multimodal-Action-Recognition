import copy

rgb_frms = 16
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

from mmaction.datasets.pipelines import Compose
import torch
from torch.utils.data import Dataset
import numpy as np


class CustomPoseDataset(Dataset):
    def __init__(self, dataset_name,annotation_lst, mode,pose_input='joint',class_prob=None):
        self.video_infos = copy.deepcopy(annotation_lst)
        self.mode = mode
        self.class_prob = class_prob
        self.dataset_name=dataset_name
        self.setDatasetInfo(dataset_name)
        if pose_input=='joint':
            self.j_flag=True
            self.l_flag = False
        elif pose_input=='limb':
            self.j_flag = False
            self.l_flag = True
        else:
            self.j_flag = True
            self.l_flag = True

        self.set_dataset_pipeline(self.dataset_name, self.data_root)

        self.rgb_skeleton_pipeline = Compose(self.train_pipeline)
        self.rgb_skeleton_val_pipeline = Compose(self.val_pipeline)

    def set_dataset_pipeline(self,dataset_name,data_root):
        self.train_pipeline = [
            dict(type='MMUniformSampleFrames', clip_len=dict(Pose=48, RGB=rgb_frms), num_clips=1),
            dict(type='MMDecode', data_root=data_root, dataset=dataset_name),
            dict(type='MMCompact', padding=0.25, hw_ratio=1., allow_imgpad=False),
            # dict(type='Resize', scale=(-1, 64), keep_ratio=False),
            # dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
            dict(type='Resize', scale=(56, 56), keep_ratio=False),
            dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=self.j_flag,
                with_limb=self.l_flag, num_joints=17),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW', collapse=True),
            dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
        ]

        self.val_pipeline = [
            dict(type='MMUniformSampleFrames', clip_len=dict(Pose=48, RGB=rgb_frms), num_clips=1, test_mode=True),
            dict(type='MMDecode', data_root=data_root, dataset=dataset_name),
            dict(type='MMCompact', padding=0.25, hw_ratio=1., allow_imgpad=False),
            dict(type='Resize', scale=(56, 56), keep_ratio=False),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=self.j_flag,
                with_limb=self.l_flag, num_joints=17),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW', collapse=False),
            dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
        ]
        self.test_pipeline = [
            dict(type='MMUniformSampleFrames', clip_len=dict(Pose=48, RGB=rgb_frms), num_clips=1, test_mode=True),
            dict(type='MMDecode', data_root=data_root, dataset=dataset_name),
            dict(type='MMCompact', padding=0.25, hw_ratio=1., allow_imgpad=False),
            dict(type='Resize', scale=(56, 56), keep_ratio=False),
            dict(
                type='GeneratePoseTarget',
                sigma=0.6,
                use_score=True,
                with_kp=self.j_flag,
                with_limb=self.l_flag, num_joints=17),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='FormatShape', input_format='NCTHW', collapse=False),
            dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
            dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
        ]
    def setDatasetInfo(self,dataset_name):
        if dataset_name == 'ntu60':
            self.num_classes = 60
            self.data_root = 'ActionRecognitionDatasets/NTU60SpatialAlignment224/'

        elif dataset_name == 'ntu120':
            self.num_classes = 120
            self.data_root = 'ActionRecognitionDatasets/NTU120SpatialAlignment224/'
        elif dataset_name == 'toyota':
            self.num_classes = 31
            #self.num_classes = 19
            self.data_root = 'ActionRecognitionDatasets/SmarthomeSpatialAlignmentGTPoseHRNet224/'
               
        elif dataset_name =='pku':
            self.num_classes = 51
            self.data_root = 'ActionRecognitionDatasets/PKUSpatialAlignment224/'
    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, idx):
        fake_anno = copy.deepcopy(self.video_infos[idx])
        fake_anno['start_index'] = 0
        fake_anno['modality'] = 'Pose'

        if self.mode == 'train':
            fake_anno['test_mode'] = False
            pipeline_result = self.rgb_skeleton_pipeline(fake_anno)
            imgs, heatmaps_volumes = pipeline_result['imgs'], pipeline_result['heatmap_imgs']

        else:
            fake_anno['test_mode'] = True
            pipeline_result = self.rgb_skeleton_val_pipeline(fake_anno)
            imgs,heatmaps_volumes = pipeline_result['imgs'],pipeline_result['heatmap_imgs']
            # inputs = heatmaps_volumes

        action_label = fake_anno['label']
        onehot_encoding = np.zeros((self.num_classes,))
        onehot_encoding[action_label] = 1.

        return imgs, heatmaps_volumes, onehot_encoding



