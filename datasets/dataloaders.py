"""
Each dataset returns:
    images: torch tensor of shape [C, H, W], the image.
    labels: torch of shape [1], class of the image.
    relations: torch tensor of shape [NUM_PATCHES, NUM_PATCHES], the spatial relations of the patches.
"""


##################################################
# Imports
##################################################

import os
import subprocess
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, SVHN
import torch
import math
import torch.nn.functional as F
import tarfile
# import medmnist
# from medmnist import INFO


# Custom
from datasets.encode_locations import create_connectivity_matrix, create_dist_matrix, create_angle_matrix, SSLSignal
from datasets.transforms import get_transforms, transform_enc, image2patches, patches2image, create_megapatches_grid
from datasets.tiny_imagenet import TinyImagenetDataset
from datasets.caltech import Caltech256
from datasets.flower import Flower102
from datasets.cars import Cars
from datasets.oxford_pet import OxfordPet
import utils


class ShuffleDataset(Dataset):
    def __init__(self, image_dataset, shuffle=False, debug=False, patch_size=32, img_size=224, patch_transform=None, 
                 mega_patches=False, side_megapatches=5, use_relations=True, use_distances=True, use_angles=True, 
                 use_abs_positions=True):
        super().__init__()
        self.ds = image_dataset
        self.shuffle = shuffle
        self.patch_size = patch_size
        self.img_size = img_size
        self.patch_transform = patch_transform
        self.debug = debug
        self.ssl_signal = SSLSignal(use_relations=use_relations, use_distances=use_distances, use_angles=use_angles, 
                                    use_abs_positions=use_abs_positions)
        self.mega_patches = mega_patches
        self.side_megapatches = side_megapatches

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = {}
        x, y_cls = self.ds.__getitem__(idx)
        if isinstance(x, list):
            C, H, W = x[0].shape # simsiam case
        else:
            C, H, W = x.shape

        if self.debug:
            sample['images_orig'] = x
            sample['labels_orig'] = y_cls

        # Shuffle patches
        num_patches = int(self.img_size // self.patch_size) ** 2 if not self.mega_patches else self.side_megapatches ** 2
        side = int(math.sqrt(num_patches))
        idx_shuffle = torch.randperm(num_patches) if self.shuffle else torch.arange(num_patches)
    
        # Mega-patches
        if self.mega_patches:
            x = create_megapatches_grid(x, side=self.side_megapatches, patch_size=self.patch_size)
            _, H, W = x.shape

        if self.patch_transform is not None:
            x = image2patches(x, patch_size=self.patch_size) # [C, N_PATCH, H_PATCH, W_PATCH]
            x = self.patch_transform(x)
            x = x[:, idx_shuffle] # [C, N_PATCH, H_PATCH, W_PATCH]
            x = patches2image(x, output_size=(H, W)) # [C, H, W]
        ssl_labels = self.ssl_signal(idx_shuffle.numpy().reshape(side, side)) # dict of ssl matrices
        sample['images'] = x
        sample['labels'] = y_cls
        sample.update(ssl_labels)
        if self.debug:
            sample['idxs_shuffle'] = idx_shuffle
        return sample

class ImageDataset(Dataset):
    def __init__(self, ds, img_size, patch_size, use_relations=False, use_distances=False, use_angles=False, 
                 use_abs_positions=False):
        self.ds = ds
        self.img_size = img_size
        self.patch_size = patch_size
        self.ssl_signal = None
        if use_relations or use_distances or use_angles or use_abs_positions:
            self.ssl_signal = SSLSignal(use_relations=use_relations, use_distances=use_distances, 
                                        use_angles=use_angles, use_abs_positions=use_abs_positions)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds.__getitem__(i)
        sample = {
            'images': x,
            'labels': y,
        }
        if self.ssl_signal is not None:
            num_patches = int(self.img_size // self.patch_size) ** 2
            side = int(math.sqrt(num_patches))
            ssl_labels = self.ssl_signal(torch.arange(num_patches).numpy().reshape(side, side))
            sample.update(ssl_labels)
        return sample

def get_datasets(args, transform='default', target_transform='default', patch_transform='default'):
    """
    Return the PyTorch datasets.
    """

    # Transforms
    transform = get_transforms(args)[0] if transform == 'default' else transform
    target_transform = get_transforms(args)[1] if target_transform == 'default' else target_transform
    patch_transform = get_transforms(args)[2] if patch_transform == 'default' else patch_transform
    ds_args = {
        'root': args.data_base_path,
        'download': True,
    }

    if args.dataset == 'tiny_imagenet':
        ds_train = TinyImagenetDataset(train=True, transform=transform['train'], 
                                       target_transform=target_transform['train'], **ds_args)
        ds_train_aug = TinyImagenetDataset(train=True, transform=transform['train_aug'], 
                                           target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = TinyImagenetDataset(train=False, transform=transform['validation'], 
                                            target_transform=target_transform['validation'], **ds_args)
        ds_test = None
        ds_train_simsiam = TinyImagenetDataset(train=True, transform=transform['train_simsiam'], 
                                           target_transform=target_transform['train_simsiam'], **ds_args)

    elif args.dataset in ['cifar10', 'cifar10_36', 'cifar10_226', 'cifar10_224']:
        ds_train = CIFAR10(train=True, transform=transform['train'], target_transform=target_transform['train'], 
                           **ds_args)
        ds_train_aug = CIFAR10(train=True, transform=transform['train_aug'], 
                               target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = CIFAR10(train=False, transform=transform['validation'], 
                                target_transform=target_transform['validation'], **ds_args)
        ds_test = None
        ds_train_simsiam = CIFAR10(train=True, transform=transform['train_simsiam'], 
                               target_transform=target_transform['train_simsiam'], **ds_args)

    elif args.dataset == 'cifar100':
        ds_train = CIFAR100(train=True, transform=transform['train'], target_transform=target_transform['train'], 
                            **ds_args)
        ds_train_aug = CIFAR100(train=True, transform=transform['train_aug'], 
                                target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = CIFAR100(train=False, transform=transform['validation'], 
                                 target_transform=target_transform['validation'], **ds_args)
        ds_test = None
        ds_train_simsiam = CIFAR100(train=True, transform=transform['train_simsiam'], 
                                target_transform=target_transform['train_simsiam'], **ds_args)

    elif args.dataset == 'voc':
        ds_args = {
            'root': args.data_base_path,
            'year': '2012',
            'download': False,
        }

        # Download from unofficial link, the oxford one's is currently down.
        url = 'https://data.deepai.org/PascalVOC2012.zip'
        if not os.path.exists(f'{args.data_base_path}/VOCdevkit/PascalVOC2012.zip'):
            subprocess.run(f'mkdir {args.data_base_path}/VOCdevkit'.split())
            subprocess.run(f'wget -r -nc -P {args.data_base_path} {url}'.split())
            subprocess.run(f'unzip -qq -n {args.data_base_path}/VOCdevkit/PascalVOC2012.zip -d {self.data_dir}/VOCdevkit'.split())
        ds_train = VOCDetection(image_set='train', transform=transform['train'], 
                                target_transform=target_transform['train'], **ds_args)
        ds_train_aug = VOCDetection(image_set='train', transform=transform['train_aug'], 
                                    target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = VOCDetection(image_set='val', transform=transform['validation'], 
                                     target_transform=target_transform['validation'], **ds_args)
        ds_test = None

    elif args.dataset == 'imagenet':
        ds_train = ImageFolder(os.path.join(args.data_base_path, 'imagenet', 'train'), transform=transform['train'], target_transform=target_transform['train'])
        ds_train_aug = ImageFolder(os.path.join(args.data_base_path, 'imagenet', 'train'), transform=transform['train_aug'], target_transform=target_transform['train_aug'])
        ds_validation = ImageFolder(os.path.join(args.data_base_path, 'imagenet', 'val'), transform=transform['validation'], target_transform=target_transform['validation'])
        ds_test = None
        ds_train_simsiam = ImageFolder(os.path.join(args.data_base_path, 'imagenet', 'train'), transform=transform['train_simsiam'], target_transform=target_transform['train_simsiam'])

    elif args.dataset == 'caltech256':
        ds_train = Caltech256(train=True, transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = Caltech256(train=True, transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = Caltech256(train=False, transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = None
        ds_train_simsiam = Caltech256(train=True, transform=transform['train_simsiam'], target_transform=target_transform['train_simsiam'], **ds_args)

    elif args.dataset == 'flower102':
        ds_train = Flower102(split='train', transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = Flower102(split='train', transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = Flower102(split='val', transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = Flower102(split='test', transform=transform['test'], target_transform=target_transform['test'], **ds_args)
        ds_train_simsiam = Flower102(split='train', transform=transform['train_simsiam'], target_transform=target_transform['train_simsiam'], **ds_args)

    elif args.dataset == 'oxford_pet':
        ds_train = OxfordPet(train=True, transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = OxfordPet(train=True, transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = OxfordPet(train=False, transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = None
        ds_train_simsiam = OxfordPet(train=True, transform=transform['train_simsiam'], target_transform=target_transform['train_simsiam'], **ds_args)

    elif args.dataset == 'cars':
        ds_train = Cars(train=True, transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = Cars(train=True, transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = Cars(train=False, transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = None
        ds_train_simsiam = Cars(train=True, transform=transform['train_simsiam'], target_transform=target_transform['train_simsiam'], **ds_args)

    elif args.dataset == 'svhn':
        ds_train = SVHN(split='train', transform=transform['train'], target_transform=target_transform['train'], **ds_args)
        ds_train_aug = SVHN(split='train', transform=transform['train_aug'], target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = SVHN(split='test', transform=transform['validation'], target_transform=target_transform['validation'], **ds_args)
        ds_test = None
        ds_train_simsiam = SVHN(split='train', transform=transform['train_simsiam'], target_transform=target_transform['train_simsiam'], **ds_args)

    elif args.dataset == 'pathmnist':
        data_flag = 'pathmnist'
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info['python_class'])
        ds_train = DataClass(split='train', transform=transform['train'],target_transform=target_transform['train'], **ds_args)
        ds_train_aug = DataClass(split='train', transform=transform['train_aug'],target_transform=target_transform['train_aug'], **ds_args)
        ds_validation = DataClass(split='val', transform=transform['validation'],target_transform=target_transform['validation'], **ds_args)
        ds_test = DataClass(split='test', transform=transform['test'],target_transform=target_transform['test'], **ds_args)
    else:
        raise Exception(f'Error. Dataset {args.dataset} not supported.')

    # Datasets
    if args.task == 'upstream':
        shuffle_dataset_args = {
            'debug': args.debug_dataset,
            'patch_size': args.patch_size,
            'img_size': args.img_height,
            'mega_patches': args.mega_patches,
            'side_megapatches': args.side_megapatches,
            'use_relations': args.use_relations,
            'use_distances': args.use_distances,
            'use_angles': args.use_angles,
            'use_abs_positions': args.use_abs_positions,
        }
        dss = {
            'train': ShuffleDataset(ds_train, patch_transform=patch_transform['train'], **shuffle_dataset_args),
            'train_aug': ShuffleDataset(ds_train_aug, patch_transform=patch_transform['train_aug'], 
                                        shuffle=args.shuffle_patches, **shuffle_dataset_args),
            'validation': ShuffleDataset(ds_validation, patch_transform=patch_transform['validation'], 
                                         **shuffle_dataset_args),
            'test': None,
            'train_simsiam': ShuffleDataset(ds_train_simsiam, patch_transform=patch_transform['train_simsiam'], 
                                            **shuffle_dataset_args),
        }

    elif args.task == 'downstream':
        img_dataset_args = {
            'img_size': args.img_height,
            'patch_size': args.patch_size,
            'use_relations': args.use_relations,
            'use_distances': args.use_distances,
            'use_angles': args.use_angles,
            'use_abs_positions': args.use_abs_positions,
        }
        dss = {
            'train': ImageDataset(ds_train, **img_dataset_args),
            'train_aug': ImageDataset(ds_train_aug, **img_dataset_args),
            'validation': ImageDataset(ds_validation, **img_dataset_args),
            'test': None,
            'train_simsiam': None,
        }

    else:
        raise Exception(f'Error. Task "{args.task}" not supported.')

    return dss


def get_dataloaders(args):
    """
    Return the PyTorch dataloaders.
    """

    # Datasets
    transform = 'default'
    target_transform = 'default'
    # if args.backbone in ['vit', 'simsiam']:
    #     patch_transform = 'default'
    # else:
    #     patch_transform = {'train': None, 'train_aug': None, 'validation': None, 'test': None}
    dss = get_datasets(args, transform=transform, target_transform=target_transform, patch_transform='default')

    # Dataloaders
    dl_args = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    dls = {
        'train': DataLoader(dss['train'], shuffle=False, **dl_args),
        'train_aug': DataLoader(dss['train_aug'], shuffle=True, **dl_args),
        'validation': DataLoader(dss['validation'], shuffle=False, **dl_args),
        'test':  None,
        'train_simsiam': DataLoader(dss['train_simsiam'], shuffle=True, **dl_args) if args.task == 'upstream' else None,
    }
    return dls
