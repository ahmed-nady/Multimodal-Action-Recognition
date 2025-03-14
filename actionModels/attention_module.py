import torch.nn as nn
import torch
from mmcv.cnn import ConvModule

class SpatialTemporalAttention(nn.Module):
    def __init__(self,channels=256,TemporalAttentionType ='global'):
        super().__init__()
        print(f"SpatialTemporalAttention temporal_attention_type: {TemporalAttentionType}")
        self.inter_channels =  16
        self.TemporalAttentionType = TemporalAttentionType
        self.conv_ch_compress= torch.nn.Conv3d(channels, 1, (1, 3, 3),padding='same')
        self.conv_sptial_attention = torch.nn.Conv3d(1, 1, (1, 7, 7),padding='same')
        self.gap = nn.AdaptiveAvgPool3d((None, 1,1 ))
        self.fc1 = nn.Linear(in_features=16,out_features=self.inter_channels,bias=False)
        self.fc2 = nn.Linear(in_features=self.inter_channels, out_features=16, bias=False)
        self.sigmoid = nn.Sigmoid()
     
    def forward(self,x):
        input_feats =x
        bs,c,t,h,w = x.shape
        x_ch_compressed = torch.relu(self.conv_ch_compress(x))
        x = self.conv_sptial_attention(x_ch_compressed)
        spatial_attention = self.sigmoid(x).view(bs,1,t,h,w)
        pooled_spatial_attention = self.gap(spatial_attention)
        x = torch.relu(self.fc1(pooled_spatial_attention.view(bs,t)))
        x = self.fc2(x)

        temporal_attention = self.sigmoid(x).view(bs, 1, t, 1, 1)  #
        return spatial_attention *temporal_attention

class TemporalAttention(nn.Module):
    def __init__(self,channels=256):
        super().__init__()
        conv_cfg = dict(type='Conv3d')
        norm_cfg = dict(type='BN3d')
        act_cfg = dict(type='ReLU')
        fusion_conv_param = dict(kernel_size=(1,3,3), padding='same', conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        #self.conv_att = torch.nn.Conv3d(256, 1, (1, 3, 3),padding='same')
        self.conv_att = ConvModule(channels, 1 , **fusion_conv_param)
        self.gap = nn.AdaptiveAvgPool3d((None, 1,1 ))
        self.fc1 = nn.Linear(in_features=16,out_features=16,bias=False)
        self.fc2 = nn.Linear(in_features=16, out_features=16, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        bs,c,t,_,_ = x.shape
        x = self.conv_att(x)
        x = self.gap(x)
        x = torch.relu(self.fc1(x.view(bs,t)))
        x = self.fc2(x)
        x = self.sigmoid(x).view(bs,1,t,1,1)#.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class CBAMSpatialEfficientTemporalAttention(nn.Module):
    def __init__(self,attention_type='serial'):
        super().__init__()
        self.attention_type = attention_type
        k = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, (1, k, k), stride=1, padding='same', relu=False)
     
        self.gap_nested = nn.AdaptiveAvgPool3d((None, 1,1 ))
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv_temporal_attention = torch.nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x,x_rgb=None):
        bs,c,t,h,w = x.shape
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        
        spatial_attention = self.sigmoid(x_out).view(bs,1,t,h,w)

        if self.attention_type=='nested':
            x = self.gap_nested(spatial_attention)
            x = self.conv_temporal_attention(x.view(bs, 1, t))
            temporal_attention = self.sigmoid(x).view(bs,1,t,1,1)#.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            return spatial_attention *temporal_attention
        else:
            x_rgb_scaled = x_rgb*spatial_attention
            x_rgb = x_rgb_scaled.transpose(1, 2)
            x_rgb = self.gap(x_rgb)
            x_rgb = self.conv_temporal_attention(x_rgb.view(bs, 1, t))
            x_rgb = self.sigmoid(x_rgb.view(bs, t))
            return x_rgb.view(bs, 1, t, 1, 1)


if __name__ == '__main__':
    att = CBAMSpatialEfficientTemporalAttention(attention_type='nested')
    input = torch.randn(1, 256, 16, 7, 7)
    out = att(input)
    print(out.shape, out)
