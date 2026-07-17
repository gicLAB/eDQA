import math

import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torchvision.models as models
from torch.utils.data import DataLoader, SubsetRandomSampler
from models_with_algorithm.mobilenetv2_pad_saw_path_ext import MobileNetV2
import models_with_algorithm.resnet_pad_saw_path as resnet
import models_with_algorithm.res_18_baseline as res_18
import models_with_algorithm.densenet_pad_saw_path as densenet
import models_with_algorithm.vision_transformer_path as vit
import argparse
import copy
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--step_ft', type=int, default=15)
parser.add_argument('--ft_lr', type=float, default=0.001)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--model', type=str, default='resnet32')
parser.add_argument('--exp_number', type=str, default='0')
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

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

data_dir = '/data/ILSVRC2012' #'/project/data_im'

train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
#test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

train_data_index = torch.randperm(len(train_data))[:5000]
sampler = SubsetRandomSampler(train_data_index)

train_loader = DataLoader(train_data, batch_size=1024*2, shuffle=False, num_workers=4, sampler=sampler, pin_memory=True)

device = 'cuda'

def validate(val_loader, model, L_cls_f, loader_type=''):
    global args

    loss = 0
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            #print('cur', i)
            target = target.cuda(non_blocking=True)
            z = model(input)
            L_cls = L_cls_f(z, target)
            loss += L_cls.item()
            _, predicted = torch.max(z.data, 1)
            total += input.size(0)
            correct += (predicted == target).sum().item()
            
    return loss / len(val_loader), correct / total * 100

act = {}
data_loader = train_loader
path = []
layer_channel_rank = {}

if args.model == 'resnet32':
    layer_map = {0:16, 1:32, 2:64}
    layers_num = 15
elif args.model == 'mobilev2':
    layer_map = {0:16, 1:24, 2:24, 3:32, 4:32, 5:32, 6:64, 7:64, 8:64, 9:64, 10:96, 11:96, 12:96, 13:160, 14:160, 15:160, 16:320}
    layers_num = 17
elif args.model == 'densenet':
    layer_map = {0:32}
    layers_num = 58
elif args.model == 'vit':
    layer_map = {0:768}
    layers_num = 12
elif args.model == 'resnet18':
    layer_map = {0:64, 1: 128, 2: 256, 3: 512}
    layers_num = 8

for layer in range(0, layers_num):
    best_acc_post = 0
    best_c = 0
    rank = []
    if args.model == 'resnet32':
        layer_index_cur = int(layer/5)
    elif args.model == 'mobilev2':
        layer_index_cur = layer
    elif args.model == 'densenet':
        layer_index_cur = layer * 0
    elif args.model == 'vit':
        layer_index_cur = layer * 0
    elif args.model == 'resnet18':
        layer_index_cur = int(layer/2)

    for i in range(0, layer_map[layer_index_cur]):
        #print('current', i)
        path.append(i)
        
        if args.model == 'resnet32':
            model = resnet.resnet32(path=path)
            model.linear = nn.Linear(model.linear.in_features, 200)
            model = nn.DataParallel(model, device_ids=args.gpu).cuda()
            checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/resnet32_tinyimagenet_50_epochs.pth'
        elif args.model == 'mobilev2':
            model = MobileNetV2(path=path)
            model.linear = nn.Linear(model.linear.in_features, 200)
            model = nn.DataParallel(model, device_ids=args.gpu).cuda()
            checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/mobilev2_tinyimagenet_50_epochs.pth'

        elif args.model == 'vit':
            model = vit.vit_b_16(path=path)
            model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=10)
            model = nn.DataParallel(model, device_ids=args.gpu).cuda()
            checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/vit_cifar_shuffle_10_epochs.pth'

        elif args.model == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT
            tv_model = models.resnet18(weights=weights)
            # Get the state_dict (weights)
            pretrained_state_dict = tv_model.state_dict()

            model = res_18.resnet18(path)
            model.load_state_dict(pretrained_state_dict)
            #model.fc = nn.Linear(model.fc.in_features, 10)
            model = nn.DataParallel(model, device_ids=args.gpu).cuda()
            #checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/resnet18_cifar_shuffle_50_epochs.pth'
        
        #checkpoint = torch.load(checkpoint, map_location='cuda:0')
        #model.load_state_dict(checkpoint['state_dict'])
        #model.load_state_dict(pretrained_state_dict)
        st = time.time()
        loss_train, acc_post = validate(data_loader, model, L_cls_f, '* ')
        #print('used', time.time() - st)
        rank.append((i, acc_post))
        if acc_post > best_acc_post:
            best_acc_post = acc_post
            best_c = i
        path.pop()
    path.append(best_c)
    sorted_rank = sorted(rank, key=lambda x: x[1])
    channel_sorted = [item[0] for item in sorted_rank]
    layer_channel_rank[layer] = channel_sorted
    print(layer_channel_rank.keys())

with open('experiment_res/path_' + str(args.model) + '_seed_' + str(args.seed) + '_im.pkl', 'wb') as file:
    pickle.dump(layer_channel_rank, file)