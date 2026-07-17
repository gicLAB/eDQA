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

    def __init__(self, in_planes, planes, stride=1, option='A', N=3, calibrating=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        
        self.N=N
        self.scale = 1
        self.calibrating = calibrating

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
    
    def calc_layer_similarity(self, fp_act, q_act) -> float:
        """Compute cosine similarity between full-precision and quantized activations."""
        fp_flat = fp_act.view(fp_act.size(0), -1)  # flatten per sample
        q_flat = q_act.view(q_act.size(0), -1)
        cos_sim = F.cosine_similarity(fp_flat, q_flat, dim=1).mean().item()
        return cos_sim

    def quant_dequant(self, x):
        qmax = 2 ** (self.N - 1) - 1
        x_q = torch.clamp(torch.round(x / self.scale), -qmax, qmax)
        return x_q * self.scale

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        
        
        if self.calibrating:
            search_steps = 100
            #search_range = 0.2
            alpha = 0.5
            beta = 2
            max_ele = out.abs().max().item()
            
            qmax = 2 ** (self.N - 1) - 1
            best_scale = max_ele / qmax
            best_score = -1
            '''
            candidates = torch.linspace(
                best_scale * (1 - search_range),
                best_scale * (1 + search_range),
                steps=search_steps
            )
            '''
            candidates = torch.linspace(
                best_scale * alpha,
                best_scale * beta,
                steps=search_steps
            )

            for scale in candidates:
                q_act = torch.clamp(torch.round(out / scale), -qmax, qmax) * self.scale
                score = self.calc_layer_similarity(out, q_act)
                if score > best_score:
                    best_score = score
                    best_scale = scale
            
            self.scale = best_scale
                    
        else:
            out = self.quant_dequant(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, N, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, N=N)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, N=N)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, N=N)
        self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, stride, N):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        count = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, N=N, calibrating=False))
            self.in_planes = planes * block.expansion
            count += 1
        return nn.Sequential(*layers)
    
    def calibration_on(self):
        for m in self.layer1:
            m.calibrating = True
        
        for m in self.layer2:
            m.calibrating = True
        
        for m in self.layer3:
            m.calibrating = True
    
    def calibration_off(self):
        for m in self.layer1:
            m.calibrating = False
        
        for m in self.layer2:
            m.calibrating = False
        
        for m in self.layer3:
            m.calibrating = False


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet32(bit=3, num_class=10):
    return ResNet(BasicBlock, [5, 5, 5], bit, num_classes=num_class)
