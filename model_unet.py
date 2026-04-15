"""ResUNet with Squeeze-and-Excitation blocks for flood super-resolution.

Architecture overview:
    - Encoder: SEResBlock-based encoder with SE channel attention at each level
    - Decoder: bilinear upsampling with skip connections and 1x1 skip projections
    - Models: ResUNet (baseline), ResUNet_aux (with auxiliary static inputs),
      ResUNet_aux_MTL (multi-task: depth + binary flood mask)
    - Also includes a standard UNet baseline with SyncBatchNorm.

Input channels are formed by concatenating dynamic inputs (rainfall, previous flood)
with static auxiliary inputs (DEM, roughness, runoff coefficient, LR flood).
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# ****************** ResUNet with multiple levels ******************
class SEResBlock(nn.Module):
    def __init__(self, in_plane, num_plane, stride=1, reduction=16, downsample=None):
        super(SEResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_plane, out_channels=num_plane, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_plane)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=num_plane, out_channels=num_plane, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_plane)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.SEfc = nn.Sequential(
            nn.Linear(num_plane, int(num_plane/reduction), bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(int(num_plane/reduction), num_plane, bias=False),
            nn.Sigmoid()
        )
        self.downsample = downsample

    def forward(self, x):
        # [H, W] -> [H/s, W/s]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # [H/s, W/s] -> [H/s, W/s]
        out = self.conv2(out)
        out = self.bn2(out)
        channel_att = self.avg_pool(out)
        channel_att = torch.flatten(channel_att, start_dim=1)
        channel_att = self.SEfc(channel_att)
        channel_att = channel_att.view([channel_att.size(0), channel_att.size(1), 1, 1])
        out = out * channel_att.expand_as(out)
        # The residual and the output must be of the same dimensions
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x

        out = out + identity
        out = self.relu(out)

        return out


class DownConv(nn.Module):
    def __init__(self, in_plane, num_plane):
        super(DownConv, self).__init__()
        # use Instance Normalization for shallow layers
        self.dconv = nn.Sequential(
            nn.Conv2d(in_plane, num_plane, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_plane),
            nn.ReLU(inplace=False),
            nn.Conv2d(num_plane, num_plane, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_plane),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.dconv(x)


class UpConv(nn.Module):
    def __init__(self, in_plane, skip_plane, num_plane):
        super(UpConv, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.skip_connect = nn.Conv2d(in_channels=skip_plane, out_channels=in_plane, kernel_size=1, bias=False)
        self.conv = DownConv(2*in_plane, num_plane)

    def forward(self, x, skip_x):
        out = self.upsample(x)
        out = torch.cat([out, self.skip_connect(skip_x)], dim=1)
        out = self.conv(out)
        return out


class Encoder_SENet(nn.Module):
    def __init__(self, num_input_channels=2, num_target_channels=512, num_levels=4):
        super(Encoder_SENet, self).__init__()
        self.inplane = int(num_target_channels / (2**num_levels))

        self.conv_ini = DownConv(num_input_channels, self.inplane)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.num_downsample_level = num_levels
        self.downsample_layer_list = nn.ModuleList()
        for lid in range(num_levels):
            power_tmp = int(num_levels - lid - 1)
            num_ch = int(num_target_channels / (2**power_tmp))

            if lid == 0:
                stride = 1
            else:
                stride = 2

            self.downsample_layer_list.append(self._make_layer(num_ch, 2, stride=stride))

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, num_plane, blocks, stride=1):
        num_plane = int(num_plane)
        downsample = None
        if (stride != 1) or (self.inplane != num_plane):
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplane, out_channels=num_plane, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_plane)
            )
        layers = [SEResBlock(self.inplane, num_plane, stride, 16, downsample)]
        self.inplane = num_plane
        for _ in range(1, blocks):
            layers.append(SEResBlock(self.inplane, num_plane))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.conv_ini(x)
        x_in = self.maxpool(x0)

        x_tmp = self.downsample_layer_list[0](x_in)
        x_tmp_list = [x0, x_tmp]
        for i in range(1, self.num_downsample_level):
            x_tmp = self.downsample_layer_list[i](x_tmp)
            x_tmp_list.append(x_tmp)

        return x_tmp_list


class ResUNet(nn.Module):
    def __init__(self, num_input_channels=2, num_target_channels=512, num_levels=4):
        super(ResUNet, self).__init__()
        self.encoder = Encoder_SENet(num_input_channels=num_input_channels, num_target_channels=num_target_channels, num_levels=num_levels)

        self.num_levels = num_levels

        self.up_conv_list = nn.ModuleList()
        for lid in range(num_levels):
            power_tmp = int(lid)
            num_ch_in = int(num_target_channels / (2**power_tmp))
            num_ch_out = int(num_target_channels / (2**(power_tmp+1)))

            self.up_conv_list.append(UpConv(in_plane=num_ch_in, skip_plane=num_ch_out, num_plane=num_ch_out))

        self.out_conv = nn.Conv2d(num_ch_out, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x_var_list = self.encoder(x)

        out = self.up_conv_list[0](x_var_list[-1], x_var_list[-2])
        for i in range(1, self.num_levels):
            out = self.up_conv_list[i](out, x_var_list[-(i+2)])
        # x0, x1, x2, x3, x_out = self.encoder(x)
        # out = self.up_conv4(x_out, x3)
        # out = self.up_conv3(out, x2)
        # out = self.up_conv2(out, x1)
        # out = self.up_conv1(out, x0)

        out = self.out_conv(out)
        out = self.relu(out)
        
        return out

    def get_tot_prm(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResUNet_aux(nn.Module):
    def __init__(self, num_input_channels=2, num_target_channels=512, num_levels=4, num_aux_target_channels=[512], num_aux_levels_list=[4]):
        super(ResUNet_aux, self).__init__()

        self.num_levels = num_levels

        # self.encoder = Encoder_SENet(num_input_channels=num_input_channels, num_target_channels=num_target_channels, num_levels=num_levels)
        # self.encoder_aux = nn.ModuleList()
        # for i in range(len(num_aux_levels_list)):
        #     self.encoder_aux.append(Encoder_SENet(num_input_channels=1, num_target_channels=num_aux_target_channels[i], num_levels=num_aux_levels_list[i]))
        # num_features_channels = num_target_channels + sum(num_aux_target_channels)

        self.encoder = Encoder_SENet(num_input_channels=num_input_channels + len(num_aux_levels_list), 
                                     num_target_channels=num_target_channels, 
                                     num_levels=num_levels)

        num_features_channels = num_target_channels
        self.up_conv_list = nn.ModuleList()
        for lid in range(num_levels):
            power_tmp = int(lid)
            if lid == 0:
                num_ch_in = int(num_features_channels)
            else:
                num_ch_in = int(num_target_channels / (2**power_tmp))
            num_ch_out = int(num_target_channels / (2**(power_tmp+1)))

            self.up_conv_list.append(UpConv(in_plane=num_ch_in, skip_plane=num_ch_out, num_plane=num_ch_out))

        self.out_conv = nn.Conv2d(num_ch_out, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_aux):
        # x_var_list = self.encoder(x)

        # x_aux_var_list = []
        # for i in range(len(self.encoder_aux)):
        #     out_aux = self.encoder_aux[i](x_aux[i])[-1]
        #     x_aux_var_list.append(out_aux)
        
        # x_var_list[-1] = torch.cat([x_var_list[-1]] + x_aux_var_list, dim=1)

        x = torch.cat([x, x_aux], dim=1)
        x_var_list = self.encoder(x)
        
        out = self.up_conv_list[0](x_var_list[-1], x_var_list[-2])
        for i in range(1, self.num_levels):
            out = self.up_conv_list[i](out, x_var_list[-(i+2)])
        # x0, x1, x2, x3, x_out = self.encoder(x)
        # out = self.up_conv4(x_out, x3)
        # out = self.up_conv3(out, x2)
        # out = self.up_conv2(out, x1)
        # out = self.up_conv1(out, x0)

        out = self.out_conv(out)
        out = self.relu(out)
        
        return out

    def get_tot_prm(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def load_pretrained_model(self, trained_record: str, map_location=None):
        """Load pretrained models.

        Parameters
        ----------

        trained_record : str
            Path to the pretrained model file.
        
        """
        pretrained_dict = torch.load(trained_record, map_location=map_location)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if k in state_dict and (v.size() == state_dict[k].size()):
                model_dict[k] = v
                print(f"Loading pretrained model parameter: {k}, size: {v.size()}")
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)
    

class ResUNet_aux_MTL(nn.Module):
    def __init__(self, num_input_channels=2, num_target_channels=512, num_levels=4, num_aux_target_channels=[512], num_aux_levels_list=[4]):
        super(ResUNet_aux_MTL, self).__init__()
        self.num_levels = num_levels
        self.encoder = Encoder_SENet(num_input_channels=num_input_channels + len(num_aux_levels_list), 
                                     num_target_channels=num_target_channels, 
                                     num_levels=num_levels)

        num_features_channels = num_target_channels
        
        # ------depth output------
        self.up_conv_list_depth = nn.ModuleList()
        for lid in range(num_levels):
            power_tmp = int(lid)
            if lid == 0:
                num_ch_in = int(num_features_channels)
            else:
                num_ch_in = int(num_target_channels / (2**power_tmp))
            num_ch_out = int(num_target_channels / (2**(power_tmp+1)))

            self.up_conv_list_depth.append(UpConv(in_plane=num_ch_in, skip_plane=num_ch_out, num_plane=num_ch_out))

        self.out_conv_depth = nn.Conv2d(num_ch_out, 1, kernel_size=1)
        # self.fuse_conv_depth = nn.Conv2d(2, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

        # ------mask output------
        self.up_conv_list_mask = nn.ModuleList()
        for lid in range(num_levels):
            power_tmp = int(lid)
            if lid == 0:
                num_ch_in = int(num_features_channels)
            else:
                num_ch_in = int(num_target_channels / (2**power_tmp))
            num_ch_out = int(num_target_channels / (2**(power_tmp+1)))

            self.up_conv_list_mask.append(UpConv(in_plane=num_ch_in, skip_plane=num_ch_out, num_plane=num_ch_out))

        self.out_conv_mask = nn.Conv2d(num_ch_out, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, x_aux):
        x = torch.cat([x, x_aux], dim=1)
        x_var_list = self.encoder(x)
        
        # ------depth output------
        out_depth = self.up_conv_list_depth[0](x_var_list[-1], x_var_list[-2])
        for i in range(1, self.num_levels):
            out_depth = self.up_conv_list_depth[i](out_depth, x_var_list[-(i+2)])
        
        out_depth = self.out_conv_depth(out_depth)
        out_depth = self.relu(out_depth)

        # ------mask output------
        out_mask = self.up_conv_list_mask[0](x_var_list[-1], x_var_list[-2])
        for i in range(1, self.num_levels):
            out_mask = self.up_conv_list_mask[i](out_mask, x_var_list[-(i+2)])
        
        out_mask = self.out_conv_mask(out_mask)
        out_mask = self.sigmoid(out_mask)

        # # ------fuse output------
        # out_depth = torch.cat([out_depth, out_mask], dim=1)
        # out_depth = self.fuse_conv_depth(out_depth)
        # out_depth = self.relu(out_depth)

        return out_depth, out_mask
    
    def load_pretrained_model(self, trained_record: str, map_location=None):
        """Load pretrained models.

        Parameters
        ----------

        trained_record : str
            Path to the pretrained model file.
        
        """
        pretrained_dict = torch.load(trained_record, map_location=map_location)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if k in state_dict and (v.size() == state_dict[k].size()):
                model_dict[k] = v
                print(f"Loading pretrained model parameter: {k}, size: {v.size()}")
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def get_tot_prm(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# ****************** ResUNet with multiple levels ******************


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.SyncBatchNorm(mid_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.SyncBatchNorm(out_channels),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        #super().__init__()
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256 // (2 if bilinear else 1))
        self.up1 = Up(256, 128 // (2 if bilinear else 1), bilinear)
        self.up2 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.relu = nn.ReLU(inplace=False)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.SyncBatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        # Apply ReLU to ensure non-negative outputs
        return self.relu(logits)


if __name__ == "__main__":
    pass