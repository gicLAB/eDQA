'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_with_algorithm.util import *

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, layer_index=0, bit=8):
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
        self.mem = 0
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def get_mem(self):
        return self.mem
    
    def test_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

    def forward(self, x):
        if self.layer_index > 0:
            self.mem = 0
            act_scale = percentile_search(self.test_forward, x, self.bit)
            noise = search_bias(self.test_forward, x, self.bit, act_scale)
            x = x + noise
            x = quant_activation(x, self.bit, act_scale)
            self.mem = self.mem + get_act_byte(x, self.bit)
            x = de_quant_activation(x, act_scale)
            x = x - noise 

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
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

    def __init__(self, num_classes=10, bit=8):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, bit=bit)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.bit = bit
        self.mem_usage = 0

    def _make_layers(self, in_planes, bit):
        layers = []
        layer_count = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, layer_count, bit))
                in_planes = out_planes
                layer_count+=1
        
        return nn.Sequential(*layers)
    
    def get_mem_usage(self):
        for layer in self.layers:
            self.mem_usage = self.mem_usage + layer.get_mem()
        return self.mem_usage

    def test_forward(self, x):
        out = F.relu(self.bn2(self.conv2(x)))
        out = F.avg_pool2d(out, 4)
        return out

    def forward(self, x):
        self.mem_usage = 0
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)

        act_scale = percentile_search(self.test_forward, out, self.bit)
        noise = search_bias(self.test_forward, out, self.bit, act_scale)
        out = out + noise
        
        out = quant_activation(out, self.bit, act_scale)
        self.mem_usage = self.mem_usage + get_act_byte(out, self.bit)
        out = de_quant_activation(out, act_scale)
        
        out = out - noise

        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

