import math

import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
from models_with_algorithm.mobilenetv2_pad_saw_path import MobileNetV2
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

'''
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
dataset_train = random.sample(list(datasets.CIFAR10(root='./cifar10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                    transforms.ToTensor(), normalize]), download=True)), 5000)


train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)
'''

data_dir = '/home/wenhao/.cache/kagglehub/datasets/xiataokang/tinyimagenettorch/versions/1/tiny-imagenet-200'
'''
transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
'''
transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
train_data_index = torch.randperm(len(train_data))[:2500]
sampler = SubsetRandomSampler(train_data_index)

test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

train_loader = DataLoader(train_data, batch_size=1024*2, shuffle=False, num_workers=4, sampler=sampler, pin_memory=True)

val_loader = DataLoader(test_data, batch_size=1024*2, shuffle=False, num_workers=0, pin_memory=True)

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
            model = nn.DataParallel(model, device_ids=args.gpu).cuda()
            checkpoint = 'experiment_files/models/model_32.th'
        elif args.model == 'mobilev2':
            model = MobileNetV2(path=path)
            model = nn.DataParallel(model, device_ids=args.gpu).cuda()
            checkpoint = 'experiment_files/models/mobilev2.th'
            
        elif args.model == 'densenet':
            model = densenet.densenet121(path=path)
            model.classifier = nn.Linear(model.classifier.in_features, 200)
            model = nn.DataParallel(model, device_ids=args.gpu).cuda()
            checkpoint = 'experiment_files/models/dense_tinyimagenet_10_epochs.pth'

        elif args.model == 'vit':
            model = vit.vit_b_16(path=path)
            model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=200)
            model = nn.DataParallel(model, device_ids=[0, 1]).cuda()
            checkpoint = 'experiment_files/models/vit_tinyimagenet_10_epochs.pth'

        elif args.model == 'resnet18':
            model = res_18.resnet18(path)
            model.fc = nn.Linear(model.fc.in_features, 200)
            model = nn.DataParallel(model, device_ids=[0]).cuda()
            checkpoint = 'experiment_files/models/res_18_tinyimagenet_50_epochs.pth'
        
        checkpoint = torch.load(checkpoint, map_location='cuda:0')
        #model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)
        st = time.time()
        loss_train, acc_post = validate(data_loader, model, L_cls_f, '* ')
        print('used', time.time() - st)
        rank.append((i, acc_post))
        if acc_post > best_acc_post:
            best_acc_post = acc_post
            best_c = i
        path.pop()
    path.append(best_c)
    sorted_rank = sorted(rank, key=lambda x: x[1])
    channel_sorted = [item[0] for item in sorted_rank]
    layer_channel_rank[layer] = channel_sorted
    print(layer_channel_rank)

with open('experiment_res/path_' + str(args.model) + '_seed_' + str(args.seed) + '.pkl', 'wb') as file:
    pickle.dump(layer_channel_rank, file)
