import torch
import numpy as np
from torchvision.transforms import transforms
import PIL
import utils


def transform_simcrl(dataset_name):
    if dataset_name == 'fashionmnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(0.08, 1.0),
                interpolation=PIL.Image.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            utils.get_color_distortion(s=0.5),
            transforms.ToTensor(),
            utils.Clip(),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return transform


def transform_train(dataset_name):
    if dataset_name == 'fashionmnist':
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(0.08, 1.0),
                interpolation=PIL.Image.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            utils.get_color_distortion(s=0.5),
            transforms.ToTensor(),
            utils.Clip(),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return transform


def transform_test(dataset_name):
    if dataset_name == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                32,
                scale=(0.08, 1.0),
                interpolation=PIL.Image.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            utils.get_color_distortion(s=0.5),
            transforms.ToTensor(),
            utils.Clip(),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    return transform


def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target