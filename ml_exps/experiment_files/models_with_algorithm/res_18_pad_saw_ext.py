from functools import partial
from collections import Counter, namedtuple
from typing import Any, Callable, List, Optional, Type, Union
import heapq

import torch
import torch.nn as nn
from torch import Tensor

import math
import pickle


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
        self.bits_values = {1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 0: 0.0}
        #self.bits_values = {1: 0.25, 2: 0.5, 3: 0.75, 0: 0.0}
        self.seed = seed
        self.channel_map = self.get_channel_map()


    def get_channel_map(self):
        with open('/home/wenhao/DQA_Exp_TMLR/ResNet32_Path_local/experiment_res/path_resnet18_seed_' + str(self.seed) + '.pkl', 'rb') as file:
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
        # pre_fix_length_to_b = {0: '0', 1: '1'}
        # pre_fix_length_to_b = {0: '00', 1: '01', 2: '10', 3: '11'}
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
        ln_rp = 3  # 3

        pr_lp = 0
        pr_rp = 0

        v_lp = 0
        v_rp = 0

        b_to_pre_fix_length = {'000': 0, '001': 1, '010': 2, '011': 3, '100': 4, '101': 5, '110': 6, '111': 7}
        # b_to_pre_fix_length = {'0': 0, '1': 1}
        #b_to_pre_fix_length = {'00': 0, '01': 1, '10': 2, '11': 3}
        while v_rp < len(hf_map_code):
            length = hf_map_code[ln_lp: ln_rp]
            # print(length)
            pr_lp = ln_rp
            pr_rp = ln_rp + b_to_pre_fix_length[length]

            pre_fix = hf_map_code[pr_lp:pr_rp]

            v_lp = pr_rp
            v_rp = v_lp + 3  # 3

            value = b_to_pre_fix_length[hf_map_code[v_lp:v_rp]]

            hf_r_map[pre_fix] = value

            ln_lp = v_rp
            ln_rp = ln_lp + 3  # 3

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

        delta = max_ele / (math.pow(2, N - 1))

        if delta == 0:
            quantized = channel.long()
            diff = torch.zeros(channel.shape).long()
        else:
            lim = 2 ** (N - 1) - 1
            quantized = torch.floor(torch.round(channel / delta).clamp(-lim - 1, lim).long() >> 3).long()  # >> 3
            diff = torch.round(channel / delta).clamp(-lim - 1, lim).long() & 0b111
        #self.un_compressed_byte += get_act_byte(diff, 2)  # 3
        # huffman encode diff
        huffman_diff, huffman_tree = self.huffman_encode(diff)
        # print(quantized)
        return quantized, huffman_diff, huffman_tree

    def deQuant_DQA(self, max_ele, N, quantized, huffman_diff, huffman_tree):
        delta_lower = max_ele / (math.pow(2, N - 4))  # 4
        # huffman decode diff

        diff = self.huffman_decode(huffman_diff, huffman_tree, quantized.shape)

        return delta_lower * (quantized + diff)

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

        important_ratio = 0.55

        for cchannel in range(out.shape[1]):
            max_ele = out[:, cchannel, :, :].abs().max().item()
            if cchannel in self.channel_map[self.layer_index][
                           -int(len(self.channel_map[self.layer_index]) * important_ratio):]:
                q_out, huffman_diff, huffman_tree = self.quant_DQA(out[:, cchannel, :, :], max_ele, self.N + 3)  # 3
                #self.act_size_byte += get_act_byte(q_out, self.N)
                #overhead_all_size, hc, ht = get_overhead_byte(huffman_diff, huffman_tree)
                #self.overhead_size_byte += overhead_all_size
                #self.h_coding_byte += hc
                #self.h_tree_byte += ht
                out[:, cchannel, :, :] = self.deQuant_DQA(max_ele, self.N + 3, q_out, huffman_diff,
                                                          huffman_tree)  # N + 3
            else:
                channel_q = self.quant(out[:, cchannel, :, :], max_ele, self.N)
                #self.act_size_byte += get_act_byte(channel_q, self.N)
                out[:, cchannel, :, :] = self.dequant(channel_q, max_ele, self.N)

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
        self.layer1 = self._make_layer(N, 1, seed, block, 64, layers[0])
        self.layer2 = self._make_layer(N, 2, seed, block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(N, 3, seed, block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(N, 4, seed, block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                N, self.inplanes, planes, count + (layer_index - 1) * blocks, seed, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        count+=1
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    N,
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
    seed: int,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
) -> ResNet:
    
    model = ResNet(N, seed, block, layers)

    return model

def resnet18(N, seed) -> ResNet:
    return _resnet(N, seed, BasicBlock, [2, 2, 2, 2])
