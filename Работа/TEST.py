import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# from trains import StorageManager
import random
from data import *
import argparse
from transformer import *
import cifar
import mnist 
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.resnet_simclr import ResNetSimCLR
from simclr import SimCLR
from training import *
from noisy_dataset import Noisy_Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--initial_lr', type=float, help='initial learning rate', default=0.001)
parser.add_argument('--momentum', type=float, help='weight_decay for training', default=0.9)
parser.add_argument('--dataset', type=str, help='fashionmnist, cifar10, or cifar100', default='cifar10')
parser.add_argument('--network', type=str, default='fixed fe')
parser.add_argument('--resnet', help='resnet18 or resnet50', type=str, default='resnet18')
parser.add_argument('--step_size', type=int, default=7)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--simcrl_epochs', type=int, default=20)
parser.add_argument('--epochs', type=int, default=5)  # 25!!!
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--noise_rate', type=float, default=0.4)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--workers', type=int, default=16, help='how many subprocesses to use for data loading')  # 4!!!
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--imsize', type=int, default=28)
# parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

# parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')

args = parser.parse_args()
tests = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
if __name__ == "__main__":
    accs = [0, 0, 0, 0, 0, 0]
    recalls = [0, 0, 0, 0, 0, 0]
    precisions = [0, 0, 0, 0, 0, 0]
    Fs = [0, 0, 0, 0, 0, 0]
    for test in range(6):
        args.noise_rate = tests[test]
        dataset = ContrastiveLearningDataset(root_folder='data/')
        train_dataset = dataset.get_dataset(args.dataset, args.n_views)  
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = ResNetSimCLR(args.resnet, args.num_classes)

        optimizer = torch.optim.Adam(model.parameters(), args.initial_lr, weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                               last_epoch=-1)
        simclr = SimCLR(device=device, model=model, optimizer=optimizer, scheduler=scheduler, args=args)
        try:
            simclr.model.backbone.load_state_dict(torch.load('./simcrlnet'))
        except:
          # simclr.train(train_loader)
            torch.save(simclr.model.backbone.state_dict(), './simcrlnet')
        simclr.model.remove_projection_head()

        if args.dataset == 'cifar10':
            args.num_classes = 10
            #train_data = cifar.CIFAR10(root='data/', train=True,
            #                           transform=transform_train(args.dataset),
            #                           download=True,
            #                           noise_type='symmetric', noise_rate=args.noise_rate, random_state=0)
            test_data = torchvision.datasets.CIFAR10(root='data/', train=True, download=True,
                                                     transform=transform_train(args.dataset))

        if args.dataset == 'fashionmnist':
            args.num_classes = 10
            test_data = torchvision.datasets.FashionMNIST(root='data/', train=True, download=True,
                                                          transform=transform_train(args.dataset))
        train_data = Noisy_Dataset(test_data, transform=transform_train(args.dataset),
                                   noise_type='symmetric', noise_rate=args.noise_rate, random_state=0,
                                   num_classes=args.num_classes)
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.workers,
                                  drop_last=False)
        classifier = train_fixed_feature_extractor(simclr.model.backbone, train_loader, device, args)
        # torch.save(classifier.state_dict(), './testnet')
        acc, precision, recall, F1 = check_model(classifier, train_loader, train_data, device)
        accs[test] = acc
        precisions[test] = precision
        recalls[test] = recall
        Fs[test] = F1
    print(accs)
    print(precisions)
    print(recalls)
    print(Fs)

