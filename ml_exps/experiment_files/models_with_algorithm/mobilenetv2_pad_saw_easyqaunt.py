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

    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, bit=8, calibrating=False):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.bit = bit
        self.scale = 1
        self.calibrating = calibrating
        self.shortcut = nn.Sequential()
        
        #self.bits_values = {1: 0.5, 0: 0.0}
        #self.bits_values = {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0}

        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def calc_layer_similarity(self, fp_act, q_act) -> float:
        """Compute cosine similarity between full-precision and quantized activations."""
        fp_flat = fp_act.view(fp_act.size(0), -1)  # flatten per sample
        q_flat = q_act.view(q_act.size(0), -1)
        cos_sim = F.cosine_similarity(fp_flat, q_flat, dim=1).mean().item()
        return cos_sim

    def quant_dequant(self, x):
        qmax = 2 ** (self.bit - 1) - 1
        x_q = torch.clamp(torch.round(x / self.scale), -qmax, qmax)
        return x_q * self.scale

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        
        if self.calibrating:
            search_steps = 100
            #search_range = 0.2
            alpha = 0.5
            beta = 2
            max_ele = out.abs().max().item()
            
            qmax = 2 ** (self.bit - 1) - 1
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


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10, bit=8, data='cifar', seed=0):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, bit=bit)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.data = data

    def _make_layers(self, in_planes, bit):
        layers = []
        layer_count = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, bit, False))
                in_planes = out_planes
                layer_count+=1
        
        return nn.Sequential(*layers)

    def calibration_on(self):
        for m in self.layers:
            m.calibrating = True
    
    def calibration_off(self):
        for m in self.layers:
            m.calibrating = False

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
