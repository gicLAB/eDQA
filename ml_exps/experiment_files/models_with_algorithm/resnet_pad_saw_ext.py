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
    class Node(namedtuple('Node', ['freq', 'value', 'left', 'right'])):
        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(self, in_planes, planes, stride=1, option='A', layer_index=0, N=3, seed=0):
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
        
        self.bits_values = {1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 0: 0.0}
        #self.bits_values = {1: 0.5, 0: 0.0}
        #self.bits_values = {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0}

        self.act_size_byte = 0
        self.overhead_size_byte = 0
        self.h_coding_byte = 0
        self.h_tree_byte = 0
        self.un_compressed_byte = 0
        self.seed = seed
        self.channel_map = self.get_channel_map()

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
    
    def get_channel_map(self):
        with open('/home/wenhao/DQA_Exp_Ext/ResNet32_Path_local/experiment_res/path_resnet32_seed_' + str(self.seed) + '_ext.pkl', 'rb') as file:
            data = pickle.load(file)
        return data

    def build_huffman_tree(self, diff):
        huffman_diff = torch.zeros(diff.shape)
        freq = Counter(diff.flatten().cpu().numpy())
        pq = [self.Node(freq, v, None, None) for v, freq in freq.items()]
        heapq.heapify(pq)
        while len(pq) > 1:
            left = heapq.heappop(pq)
            right = heapq.heappop(pq)
            merged = self.Node(left.freq + right.freq, None, left, right)
            heapq.heappush(pq, merged)
        return pq[0]

    def convert_tree_to_map(self, hf_root):
        def visit_hf_tree(path, res_map, root):
            if root.value != None:
                res_map[path] = root.value
                return
            if root.left != None:
                visit_hf_tree(path + '0', res_map, root.left)
            if root.right != None:
                visit_hf_tree(path + '1', res_map, root.right)

        hf_map = {}

        visit_hf_tree('', hf_map, hf_root)

        return hf_map

    def convert_map_to_code(self, hf_map):
        code = ''

        if '' in hf_map:
            return '0'

        pre_fix_length_to_b = {0: '000', 1: '001', 2: '010', 3: '011', 4: '100', 5: '101', 6: '110', 7: '111'}
        #pre_fix_length_to_b = {0: '0', 1: '1'}
        #pre_fix_length_to_b = {0: '00', 1: '01', 2: '10', 3: '11'}
        for k, v in hf_map.items():
            code = code + pre_fix_length_to_b[len(k)]
            code = code + k
            code = code + pre_fix_length_to_b[v]
        return code

    def convert_code_to_map(self, hf_map_code):
        hf_r_map = {}
        if hf_map_code == '0':
            return {'': 0}

        ln_lp = 0
        ln_rp = 3 #3

        pr_lp = 0
        pr_rp = 0

        v_lp = 0
        v_rp = 0

        b_to_pre_fix_length = {'000': 0, '001': 1, '010': 2, '011': 3, '100': 4, '101': 5, '110': 6, '111': 7}
        #b_to_pre_fix_length = {'0': 0, '1': 1}
        #b_to_pre_fix_length = {'00': 0, '01': 1, '10': 2, '11': 3}
        while v_rp < len(hf_map_code):
            length = hf_map_code[ln_lp: ln_rp]
            # print(length)
            pr_lp = ln_rp
            pr_rp = ln_rp + b_to_pre_fix_length[length]

            pre_fix = hf_map_code[pr_lp:pr_rp]

            v_lp = pr_rp
            v_rp = v_lp + 3 #3

            value = b_to_pre_fix_length[hf_map_code[v_lp:v_rp]]

            hf_r_map[pre_fix] = value

            ln_lp = v_rp
            ln_rp = ln_lp + 3 #3

        return hf_r_map

    def huffman_encode(self, diff):
        hf_root = self.build_huffman_tree(diff)
        hf_dict = {}

        def gen_code(node, prefix=''):
            if node.value != None:
                hf_dict[node.value] = prefix if prefix != '' else '0'
            else:
                gen_code(node.left, prefix + '0')
                gen_code(node.right, prefix + '1')

        gen_code(hf_root)

        one_d_diff = diff.flatten().cpu().numpy()
        huffman_diff = ''
        for d in one_d_diff:
            huffman_diff += hf_dict[d]

        hf_map = self.convert_tree_to_map(hf_root)
        hf_map_code = self.convert_map_to_code(hf_map)

        return huffman_diff, hf_map_code

    def huffman_decode(self, huffman_diff, hf_map_code, shape):
        hf_map = self.convert_code_to_map(hf_map_code)
        decoded_diff = []
        curr = ''
        for b in huffman_diff:
            curr = curr + b
            if '' in hf_map:
                decoded_diff.append(self.bits_values[hf_map['']])
                curr = ''
            elif curr in hf_map:
                decoded_diff.append(self.bits_values[hf_map[curr]])
                curr = ''

        new_diff = torch.tensor(decoded_diff)
        return new_diff.view(shape).cuda()

    def quant(self, channel, max_ele, N):
        delta = max_ele/(math.pow(2, N-1))
        if delta == 0:
            return channel.long()
        lim = 2 ** (N - 1) - 1
        return torch.round(channel / delta).clamp(-lim-1, lim).long()
   
    def dequant(self, channel_q, max_ele, N):
        delta = max_ele/(math.pow(2, N-1))
        return delta * channel_q

    def quant_DQA(self, channel, max_ele, N):
        def map_func(value):
            return self.bits_values.get(value, value)
        delta = max_ele/(math.pow(2, N-1))
        
        if delta == 0:
            quantized = channel.long()
            diff = torch.zeros(channel.shape).long()
        else:
            lim = 2 ** (N - 1) - 1
            quantized = torch.floor(torch.round(channel / delta).clamp(-lim-1, lim).long() >> 3).long() # >> 3
            diff = torch.round(channel / delta).clamp(-lim-1, lim).long() & 0b111 
        self.un_compressed_byte += get_act_byte(diff, 3) #3
            #huffman encode diff
        huffman_diff, huffman_tree = self.huffman_encode(diff)
        #print(quantized)
        return quantized, huffman_diff, huffman_tree

    def deQuant_DQA(self, max_ele, N, quantized, huffman_diff, huffman_tree):
        delta_lower = max_ele/(math.pow(2, N-4)) #4 
        #huffman decode diff
        
        diff = self.huffman_decode(huffman_diff, huffman_tree, quantized.shape)
        
        return delta_lower * (quantized + diff)
    
    def get_mem(self):
        return self.act_size_byte, self.overhead_size_byte, self.h_coding_byte, self.h_tree_byte, self.un_compressed_byte

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        #max_ele = out.abs().max().item()
        self.avg_compression = 0
        self.act_size_byte = 0
        self.overhead_size_byte = 0
        self.h_coding_byte = 0
        self.h_tree_byte = 0
        self.un_compressed_byte = 0
        important_ratio = 0.4
        
        '''
        if self.layer_index / 5 == 0:
            important_ratio = 0.45
        '''

        for cchannel in range(out.shape[1]):
            max_ele = out[:,cchannel,:,:].abs().max().item()
            if cchannel in self.channel_map[self.layer_index][-int(len(self.channel_map[self.layer_index]) * important_ratio):]:
                q_out, huffman_diff, huffman_tree = self.quant_DQA(out[:,cchannel,:,:], max_ele, self.N+3)#3
                self.act_size_byte += get_act_byte(q_out, self.N)
                overhead_all_size, hc, ht = get_overhead_byte_hf(huffman_diff, huffman_tree)
                self.overhead_size_byte += overhead_all_size
                self.h_coding_byte += hc
                self.h_tree_byte += ht
                out[:,cchannel,:,:] = self.deQuant_DQA(max_ele, self.N+3, q_out, huffman_diff, huffman_tree)#N + 3 
            else:
                channel_q  = self.quant(out[:,cchannel,:,:], max_ele, self.N)
                self.act_size_byte += get_act_byte(channel_q, self.N)
                out[:,cchannel,:,:] = self.dequant(channel_q, max_ele, self.N)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, N, seed, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, layer_index=1, N=N, seed=seed)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, layer_index=2, N=N, seed=seed)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, layer_index=3, N=N, seed=seed)
        self.linear = nn.Linear(64, num_classes)
        self.act_usage = 0
        self.overhead_usage = 0
        self.h_coding_usage = 0
        self.h_tree_usage = 0
        self.un_compressed_usage = 0
        self.apply(_weights_init)


    def _make_layer(self, block, planes, num_blocks, stride, layer_index, N, seed):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        count = 0
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, layer_index=count + (layer_index - 1) * num_blocks, N=N, seed=seed))
            self.in_planes = planes * block.expansion
            count += 1
        return nn.Sequential(*layers)
    
    def get_mem_usage(self):
        #sizes = []
        for b in self.layer1:
            act, overhead, hc_usage, ht_usage, ucu_usage = b.get_mem()
            #sizes.append((act, overhead))
            self.act_usage += act
            self.overhead_usage += overhead
            self.h_coding_usage += hc_usage
            self.h_tree_usage += ht_usage
            self.un_compressed_usage += ucu_usage
        for b in self.layer2:
            act, overhead, hc_usage, ht_usage, ucu_usage = b.get_mem()
            #sizes.append((act, overhead))
            self.act_usage += act
            self.overhead_usage += overhead
            self.h_coding_usage += hc_usage
            self.h_tree_usage += ht_usage
            self.un_compressed_usage += ucu_usage
        for b in self.layer3:
            act, overhead, hc_usage, ht_usage, ucu_usage = b.get_mem()
            #sizes.append((act, overhead))
            self.act_usage += act
            self.overhead_usage += overhead
            self.h_coding_usage += hc_usage
            self.h_tree_usage += ht_usage
            self.un_compressed_usage += ucu_usage

        return self.act_usage, self.overhead_usage, self.h_coding_usage, self.h_tree_usage, self.un_compressed_usage

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

def resnet32(bit=3, seed=0, num_class=10):
    return ResNet(BasicBlock, [5, 5, 5], bit, seed, num_classes=num_class)
