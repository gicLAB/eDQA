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
    def __init__(self, in_planes, out_planes, expansion, stride, layer_index=0, bit=8, seed=0):
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
        self.channel_map = self.get_channel_map()

        #self.bits_values = {1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 0: 0.0}
        #self.bits_values = {1: 0.5, 0: 0.0}
        self.bits_values = {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0}

        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def get_channel_map(self):
        with open('/home/wenhao/DQA_Exp_fix/classification/ResNet32_Path/experiment_res/path_mobilev2_seed_' + str(self.seed) + '.pkl', 'rb') as file:
            data = pickle.load(file)
        return data

    def get_mem(self):
        return self.act_size_byte, self.overhead_size_byte, self.h_coding_byte, self.h_tree_byte, self.un_compressed_byte

    def quant(self, channel, max_ele, N):
        delta = max_ele/(math.pow(2, N-1))
        if delta == 0:
            return channel.long()
        lim = 2 ** (N - 1) - 1
        return torch.round(channel / delta).clamp(-lim-1, lim).long(), delta
    
    def de_quant(self, channel_q, delta):
        return delta * channel_q

    def deQuant_right_shift(self, channel, max_ele, N):
        delta = max_ele/(math.pow(2, N-1))
        delta_lower = max_ele/(math.pow(2, N-3)) # -4

        return delta_lower * torch.bitwise_right_shift(torch.round(channel / delta), 3)

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
        
        #pre_fix_length_to_b = {0: '000', 1: '001', 2: '010', 3:'011', 4: '100', 5: '101', 6: '110', 7: '111'}
        #pre_fix_length_to_b = {0: '0', 1: '1'}
        pre_fix_length_to_b = {0: '00', 1: '01', 2: '10', 3: '11'}
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
        ln_rp = 2 #3 

        pr_lp = 0
        pr_rp = 0

        v_lp = 0
        v_rp = 0
        
        #b_to_pre_fix_length = {'000':0, '001':1, '010':2, '011':3, '100':4, '101':5, '110':6, '111':7}
        #b_to_pre_fix_length = {'0': 0, '1': 1}
        b_to_pre_fix_length = {'00': 0, '01': 1, '10': 2, '11': 3}
        while v_rp < len(hf_map_code):
            length = hf_map_code[ln_lp: ln_rp]
            #print(length)
            pr_lp = ln_rp
            pr_rp = ln_rp + b_to_pre_fix_length[length]
            
            pre_fix = hf_map_code[pr_lp:pr_rp]

            v_lp = pr_rp
            v_rp = v_lp + 2 # + 3

            value = b_to_pre_fix_length[hf_map_code[v_lp:v_rp]]

            hf_r_map[pre_fix] = value

            ln_lp = v_rp
            ln_rp = ln_lp + 2 # + 3
        
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
        huffman_diff =''
        for d in one_d_diff:
            huffman_diff+=hf_dict[d]

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

    def quant_DQA(self, channel, max_ele, N):
        delta = max_ele/(math.pow(2, N-1))
        #z0 = torch.round(channel / delta) / 2**3
        if delta == 0:
            quantized = channel
            diff = torch.zeros(channel.shape).long()
        else:
            lim = 2 ** (N - 1) - 1
            quantized = torch.floor(torch.round(channel / delta).clamp(-lim-1, lim).long() >> 2).long() # >> 3
            diff = torch.round(channel / delta).clamp(-lim-1, lim).long() & 0b11
        self.un_compressed_byte += get_act_byte(diff, 2) # 3
        #huffman encode diff
        huffman_diff, huffman_tree = self.huffman_encode(diff)
        return quantized, huffman_diff, huffman_tree

    def deQuant_DQA(self, max_ele, N, quantized, huffman_diff, huffman_tree):
        delta_lower = max_ele/(math.pow(2, N-3)) # -4
        #huffman decode diff
        diff = self.huffman_decode(huffman_diff, huffman_tree, quantized.shape)
        return delta_lower * (quantized + diff)

    def power_of_two_quantize(self, x, num_bits):
        eps = 1e-8
        sign = torch.sign(x)
        x_abs = torch.clamp(x.abs(), min=eps)

        # Find dynamic scale
        amax = x_abs.max()

        # Compute exponent range
        e_min = -(2 ** (num_bits - 2))
        e_max = (2 ** (num_bits - 2)) - 1

        # Quantize
        log2 = torch.log2(x_abs)
        log2_rounded = torch.round(log2)
        log2_clamped = torch.clamp(log2_rounded, min=e_min, max=e_max)

        x_quant = sign * (2.0 ** log2_clamped)

        return x_quant


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        #max_ele = torch.max(out)
        self.act_size_byte = 0
        self.overhead_size_byte = 0
        self.h_coding_byte = 0
        self.h_tree_byte = 0
        self.un_compressed_byte = 0
        important_ratio = 0.4
        
        '''
        for cchannel in range(out.shape[1]):
            max_ele = out[:,cchannel,:,:].abs().max().item() 
            if cchannel in self.channel_map[self.layer_index][-int(len(self.channel_map[self.layer_index]) * important_ratio):]:
                q_out, huffman_diff, huffman_tree = self.quant_DQA(out[:,cchannel,:,:], max_ele, self.bit+2) # +3
                self.act_size_byte += get_act_byte(q_out, self.bit)
                overhead_all_size, hc, ht = get_overhead_byte(huffman_diff, huffman_tree)
                self.overhead_size_byte += overhead_all_size
                self.h_coding_byte += hc
                self.h_tree_byte += ht
                out[:,cchannel,:,:] = self.deQuant_DQA(max_ele, self.bit+2, q_out, huffman_diff, huffman_tree) # +3
            else:
                channel_q, delta  = self.quant(out[:,cchannel,:,:], max_ele, self.bit)
                self.act_size_byte += get_act_byte(channel_q, self.bit)
                out[:,cchannel,:,:] = self.de_quant(channel_q, delta)
        '''

        out = self.power_of_two_quantize(out, self.bit)

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

    def __init__(self, num_classes=10, bit=8, seed=0):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32, bit=bit, seed=seed)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        self.act_usage = 0
        self.overhead_usage = 0
        self.h_coding_usage = 0
        self.h_tree_usage = 0
        self.un_compressed_usage = 0

    def _make_layers(self, in_planes, bit, seed):
        layers = []
        layer_count = 0
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, layer_count, bit, seed))
                in_planes = out_planes
                layer_count+=1
        
        return nn.Sequential(*layers)

    def get_mem_usage(self):
        for b in self.layers:
            act, overhead, hc_usage, ht_usage, ucu_usage = b.get_mem()
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
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
