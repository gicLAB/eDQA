import math
import torch
from torch import Tensor
import sys

def get_act_byte(tensor, bits):
    s_in_bytes = (bits * tensor.numel())/8
    return s_in_bytes

def get_overhead_byte(huffman_s, huffman_t):
    s_in_bytes = 0
    s_in_bytes += len(huffman_s)/8
    s_in_bytes += sys.getsizeof(huffman_t)
    return s_in_bytes

def percentile_search(test_forward, x, bit, search_space=100):
    absmax = x.abs().max()
    min_loss = None
    best_clip = None

    for ii in range(search_space, 0, -1):
        clip_value = absmax/search_space*ii
        act_scale = clip_value/(2**(bit-1)-1)
        z0 = test_forward(x)
        z = test_forward(all_quant_activation(x.clamp(-clip_value, clip_value), bit, act_scale))
        loss = ((z-z0)**2).mean()
        '''
        if min_loss is None:
            min_loss = loss
            best_clip = clip_value
        '''
        if min_loss is None or loss < min_loss:
            min_loss = loss
            best_clip = clip_value
            best_act_scale = act_scale

    return best_act_scale

def search_mean(test_forward, x, bit, act_scale):
    criterion = torch.nn.MSELoss()
    loss_min = 1e6
    best_candidate = torch.tensor([0.0])
    search_space_mean = 50
    for ii in range(-search_space_mean,search_space_mean):
        candidate = act_scale * ii/search_space_mean
        xq = all_quant_activation(x + candidate, bit=bit, act_scale=act_scale)
        xq -= candidate
        zq = test_forward(xq)
        z = test_forward(x)

        loss = criterion(zq, z)
        if loss < loss_min:
            loss_min = loss
            best_candidate = candidate
    return best_candidate

def search_bias(test_forward, x, bit, act_scale):
    best_noisy_mean = search_mean(test_forward, x, bit, act_scale)
    noisy_bias = best_noisy_mean #(torch.randn_like(x[:1,:1,:])*2-1)*act_scale #best_noisy_mean
    search_size = 100
    best_loss = 1e6

    criterion = torch.nn.MSELoss()
    for ii in range(search_size*2):
        candidate = best_noisy_mean + noisy_bias * ii/search_size
        xq = x + candidate
        xq = all_quant_activation(xq, bit, act_scale)
        xq = xq - candidate

        z = test_forward(x)
        zq = test_forward(xq)

        loss = criterion(z, zq)
        if loss < best_loss:
            best_loss = loss
            best_noise = noisy_bias

    return best_noise

def quant_activation(x, bit, act_scale):
    n = 2 ** (bit - 1) - 1
    aint = (x / act_scale).round().clamp(-n-1,n)
    return aint

def de_quant_activation(x_q, act_scale):
    return x_q * act_scale

def all_quant_activation(x, bit, act_scale):
    n = 2 ** (bit - 1) - 1
    aint = (x / act_scale).round().clamp(-n-1,n)
    x = aint * act_scale
    return x
