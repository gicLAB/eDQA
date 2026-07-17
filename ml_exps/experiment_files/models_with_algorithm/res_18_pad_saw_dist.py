from functools import partial
from collections import Counter, namedtuple
from typing import Any, Callable, List, Optional, Type, Union
import heapq

import torch
import torch.nn as nn
from torch import Tensor
from models_with_algorithm.util import *

import math
import pickle
import zlib
import lzma
import zstandard as zstd

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1
    
    class Node(namedtuple('Node', ['freq', 'value', 'left', 'right'])):
        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(
        self,
        N: int,
        m: int,
        imp_ratio: float,
        inplanes: int,
        planes: int,
        layer_index: int = 0,
        seed: int = 0,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.N = N

        self.layer_index = layer_index
        if m == 1:
            self.bits_values = {1: 0.5, 0: 0.0}
        elif m == 2:
            self.bits_values = {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0}
        elif m == 3:
            self.bits_values = {1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 0: 0.0}
        #self.bits_values = {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0}
        self.seed = seed
        self.m = m
        self.dist = {}

    def get_mem_usage(self):
        return self.dist

    def quant_DQA(self, channel, max_ele, N):
        m_to_bit = {1:0b1, 2:0b11, 3:0b111}

        delta = max_ele/(math.pow(2, N-1))
        #z0 = torch.round(channel / delta) / 2**3
        if delta == 0:
            quantized = channel
            diff = torch.zeros(channel.shape).long()
        else:
            lim = 2 ** (N - 1) - 1
            quantized = torch.floor(torch.round(channel / delta).clamp(-lim-1, lim).long() >> self.m).long() # >> 3
            diff = torch.round(channel / delta).clamp(-lim-1, lim).long() & m_to_bit[self.m]
        
        huffman_diff = torch.zeros(diff.shape)
        freq = dict(Counter(diff.flatten().cpu().numpy()))

        return freq

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)


        for cchannel in range(out.shape[1]):
            max_ele = out[:,cchannel,:,:].abs().max().item()
            v_dict = self.quant_DQA(out[:,cchannel,:,:], max_ele, self.N) # +3
            for k in v_dict:
                if k not in self.dist:
                    self.dist[k] = 0

                self.dist[k] = self.dist[k] + v_dict[k]

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        N: int,
        m: int, 
        imp_ratio: float,
        seed: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(N, m, imp_ratio, 1, seed, block, 64, layers[0])
        self.layer2 = self._make_layer(N, m, imp_ratio, 2, seed, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(N, m, imp_ratio, 3, seed, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(N, m, imp_ratio, 4, seed, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.a_dist = {}

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        N: int,
        m: int, 
        imp_ratio: float,
        layer_index: int,
        seed: int,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        count = 0
        layers.append(
            block(
                N, m, imp_ratio, self.inplanes, planes, count + (layer_index - 1) * blocks, seed, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        count+=1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    N,
                    m,
                    imp_ratio,
                    self.inplanes,
                    planes,
                    count + (layer_index - 1) * blocks,
                    seed,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
            count+=1

        return nn.Sequential(*layers)

    def get_mem_usage(self):
        # sizes = []
        for m in self.layer1:
            b_dict = m.get_mem_usage()
            # sizes.append((act, overhead))
            for k in b_dict:
                if k not in self.a_dist:
                    self.a_dist[k] = 0
                self.a_dist[k] = self.a_dist[k] + b_dict[k]
        
        for m in self.layer2:
            b_dict = m.get_mem_usage()
            # sizes.append((act, overhead))
            for k in b_dict:
                if k not in self.a_dist:
                    self.a_dist[k] = 0
                self.a_dist[k] = self.a_dist[k] + b_dict[k]

        for m in self.layer3:
            b_dict = m.get_mem_usage()
            # sizes.append((act, overhead))
            for k in b_dict:
                if k not in self.a_dist:
                    self.a_dist[k] = 0
                self.a_dist[k] = self.a_dist[k] + b_dict[k]
        
        for m in self.layer4:
            b_dict = m.get_mem_usage()
            # sizes.append((act, overhead))
            for k in b_dict:
                if k not in self.a_dist:
                    self.a_dist[k] = 0
                self.a_dist[k] = self.a_dist[k] + b_dict[k]

        return self.a_dist


    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    N: int,
    m: int,
    imp_ratio: float,
    seed: int,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
) -> ResNet:
    
    model = ResNet(N, m, imp_ratio, seed, block, layers)

    return model

def resnet18(bit=3, m=3, imp_ratio=0.4, seed=0, num_class=10) -> ResNet:
    return _resnet(bit, m, imp_ratio, seed, BasicBlock, [2, 2, 2, 2])
