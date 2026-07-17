import math

import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, SubsetRandomSampler
import models_with_algorithm.resnet_pad_saw_easyquant as resnet
import models_with_algorithm.mobilenetv2_pad_saw_easyqaunt as mobilenet 
import models_with_algorithm.res_18_pad_saw_easyquant as res_18
import models_with_algorithm.vision_transformer_pad_saw_easyquant as vit
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
parser.add_argument('--method', type=str, default='eq')
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


if args.model == 'vit_cifar' or args.model == 'resnet18_cifar':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    dataset_train = random.sample(list(datasets.CIFAR10(root='./cifar10', train=True, transform=transforms.Compose([
                    transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.RandomCrop(224, 4),
                        transforms.ToTensor(), normalize]), download=True)), 1000)


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./cifar10', train=False, transform=transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(), normalize])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

elif args.model == 'vit_im' or args.model == 'resnet18_im':
    data_dir = '/home/wenhao/.cache/kagglehub/datasets/xiataokang/tinyimagenettorch/versions/1/tiny-imagenet-200'

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

elif args.model == 'resnet32_im' or args.model == 'mobilev2_im':
    data_dir = '/home/wenhao/.cache/kagglehub/datasets/xiataokang/tinyimagenettorch/versions/1/tiny-imagenet-200'

    mean = [0.4802, 0.4481, 0.3975]
    std  = [0.2302, 0.2265, 0.2262]

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    train_data = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train)

    train_data_index = torch.randperm(len(train_data))[:5000]
    sampler = SubsetRandomSampler(train_data_index)

    train_loader = DataLoader(train_data, batch_size=1024*2, shuffle=False, num_workers=4, sampler=sampler, pin_memory=True)

    test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_train)
    val_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

else:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset_train = random.sample(list(datasets.CIFAR10(root='./cifar10', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                    transforms.ToTensor(), normalize]), download=True)), 5000)


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=128, shuffle=False, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./cifar10', train=False, transform=transforms.Compose([
            transforms.ToTensor(), normalize])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

def calibrate(cal_loader, model):
    print('calibrating...')
    model.module.calibration_on()
    loss = 0
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for i, (input, target) in enumerate(cal_loader):
            target = target.cuda(non_blocking=True)
            z = model(input)
            L_cls = L_cls_f(z, target)
            loss += L_cls.item()
            _, predicted = torch.max(z.data, 1)
            total += input.size(0)
            correct += (predicted == target).sum().item()
    print('calibration done')
    
    model.module.calibration_off()
    return model


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

    return correct / total * 100, sum(history_time)/len(history_time)
    
print(args.model, args.bit)

if args.model == 'resnet32_cifar':
    model = resnet.resnet32(args.bit)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/model_32.th'

elif args.model == 'mobilev2_cifar':
    model = mobilenet.MobileNetV2(bit=args.bit, data='cifar')
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/mobilev2.th'

elif args.model == 'resnet32_im':
    model = resnet.resnet32(args.bit)
    model.linear = nn.Linear(model.linear.in_features, 200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/resnet32_tinyimagenet_50_epochs.pth'

elif args.model == 'mobilev2_im':
    model = mobilenet.MobileNetV2(bit=args.bit, data='im')
    model.linear = nn.Linear(model.linear.in_features, 200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/mobilev2_tinyimagenet_50_epochs.pth'

elif args.model == 'vit_im':
    model = vit.vit_b_16(args.bit)
    model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/vit_tinyimagenet_10_epochs.pth'

elif args.model == 'resnet18_im':
    model = res_18.resnet18(args.bit)
    model.fc = nn.Linear(model.fc.in_features, 200)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = 'experiment_files/models/res_18_tinyimagenet_50_epochs.pth'

elif args.model == 'vit_cifar':
    model = vit.vit_b_16(bit=args.bit)
    model.heads.head = nn.Linear(in_features=model.heads.head.in_features, out_features=10)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/vit_cifar_shuffle_10_epochs.pth'

elif args.model == 'resnet18_cifar':
    model = res_18.resnet18(args.bit)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = nn.DataParallel(model, device_ids=args.gpu).cuda()
    checkpoint = '/home/wenhao/DQA_Exp_Ext/train_model/resnet18_cifar_shuffle_50_epochs.pth'




checkpoint = torch.load(checkpoint, map_location='cuda:0')

if args.model == 'resnet32_cifar' or args.model == 'mobilev2_cifar':
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model = calibrate(train_loader, model)
acc_post_val, avg_time = validate(val_loader, model, L_cls_f, 'test')

res = {}

res['acc'] = acc_post_val
res['time'] = avg_time

print(res)

with open('experiment_res/DQA_res_' + str(args.model) + '_' + str(args.method) + '_seed_' + str(args.seed) + '_bit_' + str(args.bit) + '_easyquant.pkl', 'wb') as file:
    pickle.dump(res, file)
