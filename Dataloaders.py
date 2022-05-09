import os
import subprocess
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision import transforms
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math
import torch.nn.functional as F

def get_datasets(args):

    # CIFAR-10 dataset
    if (args.dataset == 'cifar10') :
        # Image preprocessing modules
        transform = transforms.Compose([
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(34),
            transforms.ToTensor()])

        dataset = torchvision.datasets.CIFAR10(root=f'{args.data_base_path}/{args.dataset}',
                                                    train=True, 
                                                    transform=transform,
                                                    download=True)
        validation_split = .2
       # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.seed(1245)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        test_dataset = torchvision.datasets.CIFAR10(root=f'{args.data_base_path}/{args.dataset}',
                                                    train=False, 
                                                    transform=transforms.ToTensor())
    dss = {
        'train' : dataset,
        'train_sampler' : train_sampler,
        'valid_sampler' : valid_sampler,
        'test' : test_dataset,
    }
    return dss

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
        'train': torch.utils.data.DataLoader(dss['train'], sampler=dss['train_sampler'], **dl_args),
        'validation': torch.utils.data.DataLoader(dss['train'], sampler=dss['valid_sampler'], **dl_args),
        'test': torch.utils.data.DataLoader(dss['test'], shuffle=True, **dl_args),
    }
    return dls