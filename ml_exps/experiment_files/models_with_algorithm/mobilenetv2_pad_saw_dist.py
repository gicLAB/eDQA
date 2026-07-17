'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import Counter, namedtuple
import heapq
from models_with_algorithm.util import *
import pickle

class Block(nn.Module):
    class Node(namedtuple('Node', ['freq', 'value', 'left', 'right'])):
        def __lt__(self, other):
            return self.freq < other.freq

    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, layer_index=0, bit=8, m=3, imp_ratio=0.4, seed=0):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.layer_index = layer_index
        self.bit = bit
        self.act_size_byte = 0
        self.overhead_size_byte = 0
        self.h_coding_byte = 0
        self.h_tree_byte = 0
        self.un_compressed_byte = 0
        self.seed = seed
        self.shortcut = nn.Sequential()
        self.imp_ratio = imp_ratio
        self.m = m
        self.dist = {}
        
        if m == 1:
            self.bits_values = {1: 0.5, 0: 0.0}
        elif m == 2:
            self.bits_values = {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0}
        elif m == 3:
            self.bits_values = {1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 0: 0.0}

        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def get_mem(self):
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


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
    
        for cchannel in range(out.shape[1]):
            max_ele = out[:,cchannel,:,:].abs().max().item()
            v_dict = self.quant_DQA(out[:,cchannel,:,:], max_ele, self.bit) # +3
            for k in v_dict:
                if k not in self.dist:
                    self.dist[k] = 0

                self.dist[k] = self.dist[k] + v_dict[k]

        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, bit=8, m=3, imp_ratio=0.4, data = 'cifar', seed=0):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, bit=bit, m=m, imp_ratio=imp_ratio, seed=seed)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.a_dist = {}
        self.data = data

    def _make_layers(self, in_planes, bit, m, imp_ratio, seed):
        layers = []
        layer_count = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, layer_count, bit, m, imp_ratio, seed))
                in_planes = out_planes
                layer_count+=1
        
        return nn.Sequential(*layers)

    def get_mem_usage(self):
        for b in self.layers:
            b_dict = b.get_mem()
            
            for k in b_dict:
                if k not in self.a_dist:
                    self.a_dist[k] = 0
                self.a_dist[k] = self.a_dist[k] + b_dict[k] 

        return self.a_dist

    def forward(self, x):
        self.act_usage = 0
        self.overhead_usage = 0
        self.h_coding_usage = 0
        self.h_tree_usage = 0
        self.un_compressed_usage = 0
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        if self.data == 'im':
            out = F.adaptive_avg_pool2d(out, 1)
        else:
            out = F.avg_pool2d(out, 4)
        
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
