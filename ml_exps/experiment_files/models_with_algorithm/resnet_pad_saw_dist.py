'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
import math
import numpy as np
import copy
import random
from collections import Counter, namedtuple
import heapq
from models_with_algorithm.util import *
import pickle
import zlib
import lzma
import zstandard as zstd
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    class Node(namedtuple('Node', ['freq', 'value', 'left', 'right'])):
        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(self, in_planes, planes, stride=1, option='A', layer_index=0, N=3, m=3, imp_ratio=0.4, seed=0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.layer_index = layer_index
        self.N=N
        self.avg_q_channel = None
        self.avg_channel = None
        self.count_channel = 0
        #self.channel_map = self.get_channel_map()
        
        if m == 1:
            self.bits_values = {1: 0.5, 0: 0.0}
        elif m == 2:
            self.bits_values = {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0}
        elif m == 3:
            self.bits_values = {1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 0: 0.0}

        self.act_size_byte = 0
        self.overhead_size_byte = 0
        self.h_coding_byte = 0
        self.h_tree_byte = 0
        self.un_compressed_byte = 0
        self.seed = seed
        self.imp_ratio = imp_ratio
        self.m = m
        self.dist = {}

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
    
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
            #diff = torch.round(channel / delta).clamp(-lim-1, lim)/8 - quantized

        huffman_diff = torch.zeros(diff.shape)
        #print(set(quantized.flatten().cpu().numpy()))
        #print(set(diff.flatten().cpu().numpy()), set((torch.round(channel / delta).clamp(-lim-1, lim)/8).flatten().cpu().numpy()))
        freq = dict(Counter(diff.flatten().cpu().numpy()))
        #print(freq.keys())
        return freq
    
    def get_mem(self):
        return self.dist

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #max_ele = out.abs().max().item()
        
        '''
        if self.layer_index / 5 == 0:
            important_ratio = 0.45
        '''
        
        for cchannel in range(out.shape[1]):
            max_ele = out[:,cchannel,:,:].abs().max().item()
            v_dict = self.quant_DQA(out[:,cchannel,:,:], max_ele, self.N) # +3
            for k in v_dict:
                if k not in self.dist:
                    self.dist[k] = 0

                self.dist[k] = self.dist[k] + v_dict[k]
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, N, m, imp_ratio, seed, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_index=1, N=N, m=m, imp_ratio=imp_ratio, seed=seed)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, layer_index=2, N=N, m=m, imp_ratio=imp_ratio, seed=seed)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, layer_index=3, N=N, m=m, imp_ratio=imp_ratio, seed=seed)
        self.linear = nn.Linear(64, num_classes)
        self.a_dist = {}
        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, stride, layer_index, N, m, imp_ratio, seed):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        count = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, layer_index=count + (layer_index - 1) * num_blocks, N=N, m=m, imp_ratio=imp_ratio, seed=seed))
            self.in_planes = planes * block.expansion
            count += 1
        return nn.Sequential(*layers)
    
    def get_mem_usage(self):
        #sizes = []
        for b in self.layer1:
            b_dict = b.get_mem()
            for k in b_dict:
                if k not in self.a_dist:
                    self.a_dist[k] = 0
                self.a_dist[k] = self.a_dist[k] + b_dict[k]
            
        for b in self.layer2:
            b_dict = b.get_mem()
            for k in b_dict:
                if k not in self.a_dist:
                    self.a_dist[k] = 0
                self.a_dist[k] = self.a_dist[k] + b_dict[k]
            
        for b in self.layer3:
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
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet32(bit=3, m=3, imp_ratio=0.4, seed=0, num_class=10):
    return ResNet(BasicBlock, [5, 5, 5], bit, m, imp_ratio, seed, num_classes=num_class)
