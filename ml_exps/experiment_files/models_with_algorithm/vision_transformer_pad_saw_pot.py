import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface

import numpy as np
from collections import Counter, namedtuple
import heapq
from models_with_algorithm.util import *
import pickle

class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    class Node(namedtuple('Node', ['freq', 'value', 'left', 'right'])):
        def __lt__(self, other):
            return self.freq < other.freq

    def __init__(
        self,
        N: int,
        seed: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        layer_index: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        self.N = N
        self.layer_index = layer_index

        self.bits_values = {1: 0.125, 2: 0.25, 3: 0.375, 4: 0.5, 5: 0.625, 6: 0.75, 7: 0.875, 0: 0.0}

        self.act_size_byte = 0
        self.overhead_size_byte = 0
        self.h_coding_byte = 0
        self.h_tree_byte = 0
        self.un_compressed_byte = 0
        self.seed = seed
        self.channel_map = self.get_channel_map()

    def get_channel_map(self):
        with open('/home/wenhao/DQA_Exp_Ext/ResNet32_Path_local/experiment_res/path_vit_seed_' + str(
                self.seed) + '.pkl', 'rb') as file:
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
        ln_rp = 3

        pr_lp = 0
        pr_rp = 0

        v_lp = 0
        v_rp = 0

        b_to_pre_fix_length = {'000': 0, '001': 1, '010': 2, '011': 3, '100': 4, '101': 5, '110': 6, '111': 7}
        while v_rp < len(hf_map_code):
            length = hf_map_code[ln_lp: ln_rp]
            # print(length)
            pr_lp = ln_rp
            pr_rp = ln_rp + b_to_pre_fix_length[length]

            pre_fix = hf_map_code[pr_lp:pr_rp]

            v_lp = pr_rp
            v_rp = v_lp + 3

            value = b_to_pre_fix_length[hf_map_code[v_lp:v_rp]]

            hf_r_map[pre_fix] = value

            ln_lp = v_rp
            ln_rp = ln_lp + 3

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
        delta = max_ele / (math.pow(2, N - 1))
        if delta == 0:
            return channel.long()
        lim = 2 ** (N - 1) - 1
        return torch.round(channel / delta).clamp(-lim - 1, lim).long()

    def dequant(self, channel_q, max_ele, N):
        delta = max_ele / (math.pow(2, N - 1))
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
            quantized = torch.floor(torch.round(channel / delta).clamp(-lim - 1, lim).long() >> 3).long()
            diff = torch.round(channel / delta).clamp(-lim - 1, lim).long() & 0b111
        self.un_compressed_byte += get_act_byte(diff, 3)
        # huffman encode diff
        huffman_diff, huffman_tree = self.huffman_encode(diff)
        # print(quantized)
        return quantized, huffman_diff, huffman_tree

    def deQuant_DQA(self, max_ele, N, quantized, huffman_diff, huffman_tree):
        delta_lower = max_ele / (math.pow(2, N - 4))
        # huffman decode diff

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

    def get_mem_usage(self):
        return self.act_size_byte, self.overhead_size_byte, self.h_coding_byte, self.h_tree_byte, self.un_compressed_byte

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)

        out = x + y

        self.avg_compression = 0
        self.act_size_byte = 0
        self.overhead_size_byte = 0
        self.h_coding_byte = 0
        self.h_tree_byte = 0
        self.un_compressed_byte = 0
        important_ratio = 0.4
        
        '''
        for cchannel in range(out.shape[2]):
            max_ele = out[:, :, cchannel].abs().max().item()
            if cchannel in self.channel_map[self.layer_index][
                           -int(len(self.channel_map[self.layer_index]) * important_ratio):]:
                q_out, huffman_diff, huffman_tree = self.quant_DQA(out[:, :, cchannel], max_ele, self.N + 3)
                self.act_size_byte += get_act_byte(q_out, self.N)
                overhead_all_size, hc, ht = get_overhead_byte(huffman_diff, huffman_tree)
                self.overhead_size_byte += overhead_all_size
                self.h_coding_byte += hc
                self.h_tree_byte += ht
                out[:, :, cchannel] = self.deQuant_DQA(max_ele, self.N + 3, q_out, huffman_diff,
                                                                   huffman_tree)
            else:
                channel_q = self.quant(out[:, :, cchannel], max_ele, self.N)
                self.act_size_byte += get_act_byte(channel_q, self.N)
                out[:, :, cchannel] = self.dequant(channel_q, max_ele, self.N)
        '''
        '''
        for cchannel in range(out.shape[2]):
            max_ele = out[:, :, cchannel].abs().max().item()
            channel_q = self.quant(out[:, :, cchannel], max_ele, self.N)
            out[:, :, cchannel] = self.dequant(channel_q, max_ele, self.N)
        '''
        out = self.power_of_two_quantize(out, self.N)
        return out


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        N: int,
        seed: int,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)

        self.act_usage = 0
        self.overhead_usage = 0
        self.h_coding_usage = 0
        self.h_tree_usage = 0
        self.un_compressed_usage = 0
        self.n_l = num_layers

        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                N,
                seed,
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
                i,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def get_mem_usage(self):
        # sizes = []
        for name, m in self.layers.named_children():
            act, overhead, hc_usage, ht_usage, ucu_usage = m.get_mem_usage()
            # sizes.append((act, overhead))
            self.act_usage += act
            self.overhead_usage += overhead
            self.h_coding_usage += hc_usage
            self.h_tree_usage += ht_usage
            self.un_compressed_usage += ucu_usage

        return self.act_usage, self.overhead_usage, self.h_coding_usage, self.h_tree_usage, self.un_compressed_usage

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        N: int,
        seed: int,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        _log_api_usage_once(self)
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            N,
            seed,
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def get_mem_usage(self): 
        return self.encoder.get_mem_usage()
                                                    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def _vision_transformer(
    N: int,
    seed: int,
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,

) -> VisionTransformer:
    image_size = 224

    model = VisionTransformer(
        N=N,
        seed=seed,
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
    )

    return model

def vit_b_16(bit, seed) -> VisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    return _vision_transformer(
        N=bit,
        seed=seed,
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
    )
