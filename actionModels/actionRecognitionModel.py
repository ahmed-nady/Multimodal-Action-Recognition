# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import load_checkpoint
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.cnn import normal_init
from mmaction.utils.logger import  get_root_logger
class C3D(nn.Module):
    """C3D backbone, without flatten and mlp.

    Args:
        pretrained (str | None): Name of pretrained model.
    """

    def __init__(self,
                 in_channels=17,
                 base_channels=32,
                 num_stages=4,
                 temporal_downsample=False,
                 pretrained=None):
        super().__init__()
        conv_cfg = dict(type='Conv3d')
        norm_cfg = dict(type='BN3d')
        act_cfg = dict(type='ReLU')
        self.pretrained = pretrained
        self.in_channels = in_channels
        self.base_channels = base_channels
        assert num_stages in [3, 4]
        self.num_stages = num_stages
        self.temporal_downsample = temporal_downsample
        pool_kernel, pool_stride = 2, 2
        if not self.temporal_downsample:
            pool_kernel, pool_stride = (1, 2, 2), (1, 2, 2)

        c3d_conv_param = dict(kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.conv1a = ConvModule(self.in_channels, self.base_channels, **c3d_conv_param)
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = ConvModule(self.base_channels, self.base_channels * 2, **c3d_conv_param)
        self.pool2 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv3a = ConvModule(self.base_channels * 2, self.base_channels * 4, **c3d_conv_param)
        self.conv3b = ConvModule(self.base_channels * 4, self.base_channels * 4, **c3d_conv_param)
        self.pool3 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        self.conv4a = ConvModule(self.base_channels * 4, self.base_channels * 8, **c3d_conv_param)
        self.conv4b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

        if self.num_stages == 4:
            #self.pool4 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
            self.conv5a = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
            self.conv5b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')
            self.pretrained = None#cache_checkpoint(self.pretrained)
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data. The size of x is (num_batches, 3, 16, 112, 112).

        Returns:
            torch.Tensor: The feature of the input samples extracted by the backbone.
        """
        x = self.conv1a(x)

        x = self.pool1(x)
        x = self.conv2a(x)

        x = self.pool2(x)
        x = self.conv3a(x)
        x = self.conv3b(x)

        x = self.pool3(x)
        x = self.conv4a(x)
        x = self.conv4b(x)

        if self.num_stages == 3:
            return x

        #x = self.pool4(x)
        x = self.conv5a(x)
        x = self.conv5b(x)

        return x
class ClsHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """
    def __init__(self,
                 num_classes,
                 in_channels,
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__()
        self.in_channels =in_channels
        self.num_classes = num_classes
        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None
        #self.embedding = nn.Linear(256, 256)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        # x = self.embedding(x)
        # x = F.normalize(x, p=2, dim=1)
        return cls_score

class ActionRecognizer(nn.Module):
    def __init__(self,pretrained=None):
        super().__init__()
        self.pretrained = pretrained
        self.backbone = C3D()
        self.clsHead = ClsHead(num_classes=60,in_channels=256)
    def init_weights(self):
        if isinstance(self.pretrained,str):
            print(f"Looad checkpoint from {self.pretrained}")
            state_dict = torch.load(self.pretrained)
            from collections import OrderedDict
            import re
            # state_dict =OrderedDict()
            # for k,v in pretrainedModel_stat_dict.items:
            #     key = pretrainedModel_stat_dict[k][7:]
            #     state_dict[key] = v
            """
            revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].
            """
            # strip prefix of state_dict
            metadata = getattr(state_dict, '_metadata', OrderedDict())
            revise_keys: list = [(r'^module\.', '')]
            for p, r in revise_keys:
                state_dict = OrderedDict(
                    {re.sub(p, r, k): v
                     for k, v in state_dict.items()})
            # Keep metadata in state_dict
            state_dict._metadata = metadata
            self.backbone.load_state_dict(state_dict, strict=False)

            #====initialize weights of head classification=====
            self.clsHead.init_weights()
        else:
            self.backbone.init_weights()
            self.clsHead.init_weights()
    def forward(self,x):
        feats = self.backbone(x)
        logits= self.clsHead(feats)
        return logits

if __name__== "__main__":

    model = ActionRecognizer(pretrained='checkpoint_0.pth')
    model.init_weights()
    #====freeze the backbone to extract feats without retraining===
    for param in model.backbone.parameters():
        param.requires_grad=False
    #model.backbone.requires_grad_(False)
    for name,param in model.named_parameters():
        print(name,param.requires_grad)
        #param.requires_grad = False