import os
import subprocess
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import random_split
import math
import torch.nn.functional as F

def get_datasets(args):

    # CIFAR-10 dataset
    if (args.dataset_name == 'cifar10') :
        # Image preprocessing modules
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(34),
            transforms.ToTensor()])

        dataset = torchvision.datasets.CIFAR10(root=f'{args.dataset_path}/{args.dataset_name}',
                                                    train=True, 
                                                    transform=transform,
                                                    download=True)
        train_size = (len(dataset)*70)//100
        train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset)-train_size])

        test_dataset = torchvision.datasets.CIFAR10(root=f'{args.dataset_path}/{args.dataset_name}',
                                                    train=False, 
                                                    transform=transforms.ToTensor())
    dss = {
        'train' : train_dataset,
        'validation' : val_dataset,
        'test' : test_dataset,
    }

def get_dataloaders(args):
    """
    Return the PyTorch dataloaders.
    """
    # Datasets
    transform = 'default'
    target_transform = 'default'
    if args.backbone in ['vit', 'simsiam']:
        patch_transform = 'default'
    else:
        patch_transform = {'train': None, 'train_aug': None, 'validation': None, 'test': None}
    dss = get_datasets(args)

    # Dataloaders
    dl_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    dls = {
        'train': DataLoader(dss['train'], shuffle=False, **dl_args),
        'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        'test':  DataLoader(dss['test'], shuffle=False, **dl_args),
        'train_simsiam': DataLoader(dss['train_simsiam'], shuffle=True, **dl_args) if args.task == 'upstream' else None,
    }
    return dls