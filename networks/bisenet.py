# import os
# import torch
# import numpy as np
# from torch import nn
# from PIL import Image
# from glob import glob
# from torchvision import models
# import torch.nn.functional as F
# # Build data loader
# from tqdm import tqdm
# from torchvision import transforms
# from collections import OrderedDict
# from torch.utils.data import DataLoader, Dataset
# from torchsummary import summary

from .libs import *

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        # Head block is a convolution layer
        self.conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm2d(out_channels)

        # ReLU functions
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x

class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# spatial_path = SpatialPath()
# spatial_path = spatial_path.cpu()
#
# summary(spatial_path, (3, 256, 256))

# Build context path
class ContextPath(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.max_pool = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x_input):
        # Get feature from lightweight backbone network
        x = self.conv1(x_input)
        x = self.relu(self.bn1(x))
        x = self.max_pool(x)

        # Downsample 1/4
        feature1 = self.layer1(x)

        # Downsample 1/8
        feature2 = self.layer2(feature1)

        # Downsample 1/16
        feature3 = self.layer3(feature2)

        # Downsample 1/32
        feature4 = self.layer4(feature3)

        # Build tail with global averange pooling
        tail = self.avg_pool(feature4)
        return feature3, feature4, tail


# Attention Refinement Module (ARM)
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels

    def forward(self, x_input):
        # Apply Global Average Pooling
        x = self.avg_pool(x_input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)

        # Channel of x_input and x must be same
        return torch.mul(x_input, x) # Nhan ma tran x_input va x theo bao bao

# Define Feature Fusion Module
class FeaturefusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

    def forward(self, x_input1, x_input2):
        x = torch.concat((x_input1, x_input2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.conv_block(x)

        # Apply above branch in feature
        x = self.avg_pool(feature)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        # Multipy feature and x
        x = torch.mul(x, feature)

        # combine feature and x by add matrix
        x = torch.add(x, feature)

        return x

# Define BiSeNet
class BiSeNet(nn.Module):
    def __init__(self, num_classes, training=True):
        super().__init__()
        self.training = training
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        self.arm1 = AttentionRefinementModule(in_channels=256, out_channels=256)
        self.arm2 = AttentionRefinementModule(in_channels=512, out_channels=512)

        # Supervision for calculate loss
        self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)

        # Feature Fusion Module
        self.ffm = FeaturefusionModule(num_classes=num_classes, in_channels=1024)

        # Final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, x_input):
        # Spatial path output
        sp_out = self.spatial_path(x_input)

        # Context path output
        feature1, feature2, tail = self.context_path(x_input)

        # Apply Attension Refinement Module
        feature1, feature2 = self.arm1(feature1), self.arm2(feature2)

        # Combine output of lightweight mode with tail
        feature2 = torch.mul(feature2, tail)

        # Upsampling
        size2d_out = sp_out.size()[-2:]
        feature1 = F.interpolate(feature1, size=size2d_out, mode='bilinear')
        feature2 = F.interpolate(feature2, size=size2d_out, mode='bilinear')
        context_out = torch.cat((feature1, feature2), dim=1)

        # Apply Freature Fusion Module
        combine_feature = self.ffm(sp_out, context_out)

        # Upsampling
        bisinet_out = F.interpolate(combine_feature, scale_factor=8, mode='bilinear')
        bisinet_out = self.conv(bisinet_out)

        # when training model
        if self.training:
            feature1_sup = self.supervision1(feature1)
            feature2_sup = self.supervision2(feature2)

            feature1_sup = F.interpolate(feature1_sup, size=x_input.size()[-2:], mode='bilinear')
            feature2_sup = F.interpolate(feature2_sup, size=x_input.size()[-2:], mode='bilinear')

            return bisinet_out, feature1_sup, feature2_sup

        return bisinet_out
