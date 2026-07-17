import math

import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import models_with_algorithm.resnet_pad_saw_ablation as resnet
from models_with_algorithm.mobilenetv2_pad_saw_ablation import MobileNetV2
import argparse
import copy
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--step_ft', type=int, default=15)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--model', type=str, default='resnet32')
parser.add_argument('--bit', type=int, default=8)
parser.add_argument('--exp_number', type=str, default='0')
parser.add_argument('--m', type=int, default=3)
parser.add_argument('--imp_ratio', type=float, default=0.4)
parser.add_argument('--seed', type=int, default=0)


global args, iters

args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Seeded everything")

set_seed(args.seed)

args.gpu = [int(i) for i in args.gpu.split(',')]
torch.cuda.set_device(args.gpu[0] if args.gpu else None)
print(torch.cuda.get_device_name())

L_cls_f = nn.CrossEntropyLoss().cuda()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./cifar10', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
        transforms.ToTensor(), normalize]), download=True),
    batch_size=128, shuffle=False,
    num_workers=args.workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./cifar10', train=False, transform=transforms.Compose([
        transforms.ToTensor(), normalize])),
    batch_size=128, shuffle=False,
    num_workers=args.workers, pin_memory=True)

'''
data_dir = '/home/wenhao/.cache/kagglehub/datasets/xiataokang/tinyimagenettorch/versions/1/tiny-imagenet-200'

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

#print('length', len(test_data))
#exit()

train_loader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

val_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
'''

device = 'cuda'

def get_model_weights_size(ckpt):
    size_model = 0
    for param in ckpt.values():
        if param.is_floating_point():
            size_model += param.numel() * torch.finfo(param.dtype).bits
        elif param.dtype == torch.int64:
            size_model += param.numel() * torch.iinfo(param.dtype).bits
        else:
            print('error not supported type')
    return size_model / 8

def validate(val_loader, model, L_cls_f, loader_type=''):
    global args
    history_time = []
    history_mem = []
    history_act_mem = []
    history_overhead_mem = []
    history_hc_mem = []
    history_ht_mem = []
    history_ucu_mem = []
    loss = 0
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            print('cur', i)
            target = target.cuda(non_blocking=True)
            start_time = time.time()
            z = model(input)
            total_time = time.time() - start_time
            L_cls = L_cls_f(z, target)
            loss += L_cls.item()
            _, predicted = torch.max(z.data, 1)
            total += input.size(0)
            correct += (predicted == target).sum().item()
            history_time.append(total_time)

            '''
            act, overhead, hc, ht, ucu = model.module.get_mem_usage()
            history_mem.append(act + overhead)
            history_act_mem.append(act)
            history_overhead_mem.append(overhead)
            history_hc_mem.append(hc)
            history_ht_mem.append(ht)
            history_ucu_mem.append(ucu)
            '''
    #print(history_mem)
    '''
    return loss / len(val_loader), correct / total * 100, sum(history_time)/len(history_time), sum(history_mem)/len(history_mem), sum(history_act_mem)/len(history_act_mem), sum(history_overhead_mem)/len(history_overhead_mem), sum(history_hc_mem)/len(history_hc_mem), sum(history_ht_mem)/len(history_ht_mem), sum(history_ucu_mem)/len(history_ucu_mem)
    '''
    return correct / total * 100, sum(history_time)/len(history_time)
    
print(args.model, args.bit)
if args.model == 'resnet32':
    model = resnet.resnet32(bit=args.bit, m=args.m, imp_ratio=args.imp_ratio, seed=args.seed)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/model_32.th'
elif args.model == 'mobilev2':
    model = MobileNetV2(bit=args.bit, m=args.m, imp_ratio=args.imp_ratio, seed=args.seed)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/mobilev2.th'
'''
elif args.model == 'densenet':
    model = densenet.densenet121(bit=args.bit, seed=args.seed)
    model.classifier = nn.Linear(model.classifier.in_features, 200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/dense_tinyimagenet_10_epochs.pth'
elif args.model == 'vit':
    model = vit.vit_b_16(args.bit, args.seed)
    model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/vit_tinyimagenet_10_epochs.pth'
elif args.model == 'resnet18':
    model = res_18.resnet18(args.bit, args.seed)
    model.fc = nn.Linear(model.fc.in_features, 200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/res_18_tinyimagenet_50_epochs.pth'
'''

checkpoint = torch.load(checkpoint, map_location='cuda:0')
model.load_state_dict(checkpoint['state_dict'])
acc_post_val, avg_time = validate(val_loader, model, L_cls_f, 'test')

res = {}

res['acc'] = acc_post_val
res['time'] = avg_time
'''
res['mem'] = avg_mem
res['act'] = avg_act
res['overhead'] = avg_overhead
res['huffman_coding'] = avg_hc
res['huffman_tree'] = avg_ht
res['ucu'] = avg_ucu
'''

#res['model_size'] = get_model_weights_size(checkpoint['state_dict'])
#res['model_size'] = get_model_weights_size(checkpoint)
print(res)


with open('experiment_res/abl_verify_again/DQA_res_abl_' + str(args.model) +'_m_' + str(args.m) +'_imp_'+str(int(args.imp_ratio * 100))+'_seed_' + str(args.seed) + '_bit_' + str(args.bit) + '.pkl', 'wb') as file:
    pickle.dump(res, file)
