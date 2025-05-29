import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init,constant_init
from mmcv.utils import _BatchNorm
from mmcv.runner import load_checkpoint
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmaction.models.backbones import X3D,C3D,X3DPose,ResNet3dSlowOnly,\
    X3DTemporalShift,X3DTemporalShiftPose
from mmaction.models.heads import X3DHead,I3DHead
from collections import OrderedDict
import re
from .attention_module import SpatialTemporalAttention,CBAMSpatialEfficientTemporalAttention
from .NonLocalBlock import NonLocal3d


class PoseActionRecognition(nn.Module):
    def __init__(self,backbone_type='X3DPose',num_classes=60,num_stages=4,conv1_stride=2):
        super(PoseActionRecognition,self).__init__()
        in_channels = 256
        
        if backbone_type=='X3DPose':
            self.backbone = X3DPose(gamma_d=1,in_channels=17,base_channels=24,num_stages=3,se_ratio=None,use_swish=False,
                            stage_blocks=(5, 11, 7),spatial_strides=(2, 2, 2),conv1_stride=conv1_stride)
            in_channels = 216
        elif backbone_type=='X3DPoseSE':
            self.backbone = X3DPose(gamma_d=1,in_channels=17,base_channels=24,num_stages=3,
                            stage_blocks=(5, 11, 7),spatial_strides=(2, 2, 2),conv1_stride=conv1_stride)
            in_channels = 216
        elif backbone_type=='poseX3dTShiftSE':
            self.backbone = X3DTemporalShiftPose(gamma_d=1, in_channels=17, base_channels=24, num_stages=3,
                                                      stage_blocks=(5, 11, 7), spatial_strides=(2, 2, 2),
                                                      conv1_stride=1)
            in_channels = 216
        elif backbone_type=='SlowOnly':
            self.backbone = ResNet3dSlowOnly(depth=50,
                            pretrained=None,in_channels=17,base_channels=32,num_stages=3,out_indices=(2, ),
                            stage_blocks=(4, 6, 3),conv1_stride_s=1,pool1_stride_s=1,inflate=(0, 1, 1),spatial_strides=(2, 2, 2),
                            temporal_strides=(1, 1, 1),dilations=(1, 1, 1))
            in_channels = 512

        else:
            self.backbone = C3D(in_channels=17,base_channels=32, num_stages=num_stages, temporal_downsample=False)
        print("backbone_type: ",backbone_type)
        self.cls_head = I3DHead(num_classes=num_classes, in_channels=in_channels)

    def forward(self, heatmap_imgs):

        pose_feats = self.backbone(heatmap_imgs)
        pose_logits = self.cls_head(pose_feats)
        return pose_logits
class RGBActionRecognition(nn.Module):
    def __init__(self,num_classes=60,pretrained=None):
        super().__init__()
        self.pretrained = pretrained
        self.rgb_backbone = X3D(gamma_w=1, gamma_b=2.25, gamma_d=2.2,use_sta=False)
        self.rgb_cls_head = X3DHead(num_classes=num_classes, in_channels=432)
    def init_weights(self):
        if isinstance(self.pretrained,str) :
            print(f"Looad checkpoint from {self.pretrained}")
            loc = f"cuda:{0}"
            state_dict = torch.load(self.pretrained,map_location=loc)
            # strip prefix of state_dict
            metadata = getattr(state_dict, '_metadata', OrderedDict())
            state_dict_mod = OrderedDict()
            for k, v in state_dict['state_dict'].items():
                if 'backbone' in k:
                    state_dict_mod[re.sub('backbone.', '', k)] = v
                elif 'cls_head' in k:
                    state_dict_mod[re.sub('cls_head.', '', k)] = v
            # # Keep metadata in state_dict
            state_dict_mod._metadata = metadata
            self.rgb_backbone.load_state_dict(state_dict_mod, strict=False)
            self.rgb_cls_head.load_state_dict(state_dict_mod, strict=False)

            print('Loaded Successfully...!')
        self.rgb_backbone.init_weights()
        self.rgb_cls_head.init_weights()
    def forward(self,imgs):
        rgb_feats = self.rgb_backbone(imgs)
        rgb_logits = self.rgb_cls_head(rgb_feats)
        return rgb_logits

class RGBPoseAttentionActionRecognizer(nn.Module):
    def __init__(self, RGBPretrained=None, PosePretrained=None, attention='temporal',backbone_type='poseX3dTShiftSE',num_classes=120):
        super().__init__()
        self.RGBPretrained = RGBPretrained
        self.PosePretrained = PosePretrained

        self.rgb_backbone = X3DTemporalShift(gamma_w=1, gamma_b=2.25, gamma_d=2.2,
                                             use_sta=False,se_style='half')#,se_style='all')
        self.rgb_cls_head = I3DHead(num_classes=num_classes, in_channels=432)

        self.pose_backbone = X3DTemporalShiftPose(gamma_d=1, in_channels=17, base_channels=24, num_stages=3,
                                                  stage_blocks=(5, 11, 7), spatial_strides=(2, 2, 2),
                                                  conv1_stride=1)
        self.pose_cls_head = I3DHead(num_classes=num_classes, in_channels=216)

        self.attention_module = None
        self.attention = attention
        if  attention =='spatial_temporal':
             self.attention_module = SpatialTemporalAttention(channels=216)
        elif attention=='CBAM_spatial_efficient_temporal':
            self.attention_module = CBAMSpatialEfficientTemporalAttention(attention_type='nested')
        elif attention=='self_attention':
            self.attention_module = NonLocal3d(in_channels=432,mode='dot_product')
        if self.attention_module is None:
            print("Attention should be selected .........!")
      
    def init_weights(self):
        # initialize RGBPoseModel: time-strided conv and rgb_pose interaction
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.Conv1d):
                kaiming_init(m)
            elif isinstance(m, _BatchNorm):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                """Initiate the parameters from scratch."""
                normal_init(m, std=0.01)
        if self.attention == 'self_attention':
            self.attention_module.init_weights()
        self.rgb_backbone.init_weights()
        self.rgb_cls_head.init_weights()
        self.pose_backbone.init_weights()
        self.pose_cls_head.init_weights()

    def forward(self, imgs, heatmap_imgs):

        rgb_feats = self.rgb_backbone(imgs)
        pose_feats = self.pose_backbone(heatmap_imgs)
        # ========implement guided block=======#
        # ===get crossponding pose frames to those RGB frames === or use time-strided convolution
        time_strided_inds = [i for i in range(0, 48, 3)]
        # pose_feats shape is N x  C x T x H x W
        time_strided_pose_feats = torch.index_select(pose_feats, 2,
                                                     torch.tensor(time_strided_inds, device=pose_feats.device))

        if self.attention == 'self_attention':
            rgb_attended_feats = self.attention_module(rgb_feats,time_strided_pose_feats)
            guided_rgb_logits = self.rgb_cls_head(rgb_attended_feats)
        else:

            attention_maps = self.attention_module(time_strided_pose_feats)
            rgb_attended_feats = rgb_feats * attention_maps
            guided_rgb_logits = self.rgb_cls_head(rgb_feats+rgb_attended_feats)

        pose_logits = self.pose_cls_head(pose_feats)
        return guided_rgb_logits, pose_logits 

