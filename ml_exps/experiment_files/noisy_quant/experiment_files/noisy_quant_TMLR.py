import math

import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision import transforms, datasets
from models_with_algorithm.mobilenetv2_Noisy_ext import MobileNetV2
import models_with_algorithm.resnet_Noisy as resnet
import models_with_algorithm.densenet_noisy as densenet
import models_with_algorithm.res_18_Noisy as res_18
import models_with_algorithm.vision_transformer_noisy as vit
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

test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

val_loader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

device = 'cuda'

def validate(val_loader, model, L_cls_f, loader_type=''):
    global args
    history_time = []
    history_mem = []

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
            #history_mem.append(model.module.get_mem_usage())
    #print(history_mem) 
    return loss / len(val_loader), correct / total * 100, sum(history_time)/len(history_time) #, sum(history_mem)/len(history_mem)


print(args.model, args.bit)
if args.model == 'resnet32':
    model = resnet.resnet32(bit=args.bit)
    model.linear = nn.Linear(model.linear.in_features, 200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/resnet32_tinyimagenet_50_epochs.pth'
elif args.model == 'mobilev2':
    model = MobileNetV2(bit=args.bit)
    model.linear = nn.Linear(model.linear.in_features, 200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/mobilev2_tinyimagenet_50_epochs.pth'
elif args.model == 'densenet':
    model = densenet.densenet121(bit=args.bit, seed=args.seed)
    model.classifier = nn.Linear(model.classifier.in_features, 200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/dense_tinyimagenet_10_epochs.pth'
elif args.model == 'vit':
    model= vit.vit_b_16(args.bit)
    model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=10)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/vit_cifar_shuffle_10_epochs.pth'
elif args.model == 'resnet18':
    weights = models.ResNet18_Weights.DEFAULT
    tv_model = models.resnet18(weights=weights)
    # Get the state_dict (weights)
    pretrained_state_dict = tv_model.state_dict()

    model = res_18.resnet18(args.bit)
    model.load_state_dict(pretrained_state_dict)
    #model.fc = nn.Linear(model.fc.in_features, 10)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()

#checkpoint = torch.load(checkpoint, map_location='cuda:0')
#model.load_state_dict(checkpoint)
loss_val, acc_post_val, avg_time = validate(val_loader, model, L_cls_f, 'test')

res = {}
res['acc'] = acc_post_val
res['time'] = avg_time
#res['mem'] = avg_mem
print(res)


with open('experiment_res/noisy_res_' + str(args.model) + '_seed_' + str(args.seed) + '_bit_' + str(args.bit) + '_tmlr.pkl', 'wb') as file:
    pickle.dump(res, file)