##################################################
# Imports
##################################################

from torchvision import transforms
import torch
import torch.nn.functional as F
import numpy as np
from PIL import ImageFilter
import random

# Custom
from datasets import constants


def image2patches(x, patch_size):
    """
    Extract non overlapped patches of a given patch size from a given image.
    """
    C, H, W = x.shape
    if (not isinstance(patch_size, list)) and (not isinstance(patch_size, tuple)):
        patch_size = (patch_size, patch_size)
    x = x.unsqueeze(0) # [1, C, H, W]
    x = torch.nn.functional.unfold(x, kernel_size=patch_size, stride=patch_size) # [1, C * prod(PATCH_SIZE), NUM_PATCHES]
    x = x.view(1, C, patch_size[0], patch_size[1], -1).transpose(0, 1) # [C, 1, PATCH_SIZE[0], PATCH_SIZE[1], NUM_PATCHES]
    x = x.transpose(1, -1).squeeze(-1) # [C, NUM_PATCHES, PATCH_SIZE[0], PATCH_SIZE[1]]
    return x

def patches2image(x, output_size):
    C, N_PATCH, H_PATCH, W_PATCH = x.shape
    x = x.unsqueeze(0) # [1, C, N_PATCH, H_PATCH, W_PATCH]
    x = x.reshape(1, C, N_PATCH, -1).transpose(2, 3).reshape(1, -1, N_PATCH) # [1, C, prod(PATCH_SIZE)]
    x = F.fold(x, output_size, kernel_size=(H_PATCH, W_PATCH), stride=(H_PATCH, W_PATCH)) # [1, C, H, W]
    return x.squeeze(0) # [C, H, W]


def compute_label_encoding():
    """
    It assumes 8 classes, from a grid of 3x3 patches.
    
    |  1  |  12 |  2  |  23 |  3  |
    |  14 |  1  |  2  |  3  |  35 |
    |  4  |  4  |  X  |  5  |  5  |   ->   Coded Location Classes Relative to "X".
    |  46 |  6  |  7  |  8  |  85 |
    |  6  |  67 |  7  |  78 |  8  |

    """
    _map_classes = np.array([
        [' 1',  '1',  '2',  '3',  '3'],
        [ '1',  '1',  '2',  '3',  '3'],
        [ '4',  '4', '-1',  '5',  '5'],
        [ '6',  '6',  '7',  '8',  '8'],
        [ '6',  '6',  '7',  '8',  '8'],
    ])
    _kernel = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ])

    labels_matrix = np.zeros([9, 9])
    for i in range(3):
        for j in range(3):
            k = _kernel[i, j]

            _map = _map_classes[2 - i: 5 - i, 2 - j : 5 - j]

            for ii in range(_map.shape[0]):
                for jj in range(_map.shape[1]):
                    cl_iijj = _map[ii, jj]
                    k_iijj = _kernel[ii, jj]
                    labels_matrix[k - 1, k_iijj - 1] = cl_iijj

    # Convert to one-hot vectors
    out = np.zeros([8, 9, 9])
    for i in range(9):
        for j in range(9):
            l = labels_matrix[i, j]
            l = 0 if l == -1 else l
            l = [int(ll) for ll in str(int(l))]
            l = [F.one_hot(torch.tensor(ll) - 1, 8).to(torch.float)
                    if ll != 0 else torch.zeros(8) for ll in l]
            l = torch.stack(l, 0).mean(0).numpy()
            out[:, i, j] = l
    return out

def transform_enc(labels):
    """
    Args:
        labels: tensor of shape [9]
    
    Output:
        labels_matrix: tensor of shape [num_classes, 3, 3]
    """
    enc = compute_label_encoding() # [8, 9, 9]
    enc = torch.tensor(enc)
    enc = enc[:, labels, :][:, :, labels]
    return enc


def smart_resize(x, h, w):
    """
    x: PIL Image.
    """
    h_in, w_in = x.size
    resize = False
    h_out = min(h_in, h)
    w_out = min(w_in, w)
    if h_out < h:
        resize = True
    if w_out < w:
        resize = True
    if resize:
        x = transforms.Resize((h, w))(x)
    return x

def gray2rgb_ifneeded(x):
    return x.repeat(3, 1, 1) if x.shape[0] == 1 else x

def rgba2rgb_ifneeded(x):
    return x[:3] if x.shape[0] > 3 else x

def gaussian_fn(M, std):
    n = torch.arange(0, M) - (M - 1.0) / 2.0
    sig2 = 2 * std * std
    w = torch.exp(-n ** 2 / sig2)
    return w

def gkern(kernlen=256, std=128):
    """Returns a 2D Gaussian kernel array."""
    gkern1d = gaussian_fn(kernlen, std=std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d

class GaussianMasking:
    def __init__(self, kernel_size, std):
        self.kernel_size = kernel_size
        self.std = std
        self.kernel = gkern(self.kernel_size, self.std)

    def __call__(self, x):
        """
        x: patch of size [c, h, w]
        """
        x_shape = x.shape
        if len(x_shape) == 3:
            kernel = self.kernel.to(x.device).reshape(1, self.kernel_size, self.kernel_size)
        elif len(x_shape) == 4:
            kernel = self.kernel.to(x.device).reshape(1, 1, self.kernel_size, self.kernel_size)
        return x * kernel

def to_rgb_ifneeded(x):
    """
    Given a PIL Image x it returns its rgb converted version, if x is not rgb.
    """
    return x.convert('RGB') if x.mode in ['L', 'RGBA'] else x

class PerPatchAug:
    """
    Transform, per patch (each patch has its own transformation).
    """
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        x_aug = []
        for x_p in x:
            x_aug += [self.fn(x_p)]
        x_aug = torch.stack(x_aug, 0)
        return x_aug

def get_transforms(args):
    """
    Return the transformations for the datasets.
    """
    if args.dataset in ['imagenet', 'flower102', 'oxford_pet', 'cars']:
        trans = {
            'train_aug': transforms.Compose([
                to_rgb_ifneeded,
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'train': transforms.Compose([
                to_rgb_ifneeded,
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'validation': transforms.Compose([
                to_rgb_ifneeded,
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'test': transforms.Compose([
                to_rgb_ifneeded,
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'train_simsiam': get_simsiam_transform(args),
        }
        trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
            'train_simsiam': None,
        }
        patch_trans = {
            'train': None,
            'train_aug': transforms.Compose([
                lambda x: x.transpose(0, 1),
                PerPatchAug(transforms.RandomCrop(int(np.round(args.patch_size / 1.15)))), # for patch_size = 32 -> 28
                transforms.Resize(args.patch_size),
                PerPatchAug(
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8)
                ),
                PerPatchAug(transforms.RandomGrayscale(p=0.2)),
                #GaussianMasking(args.patch_size, args.patch_size / 5),
                lambda x: x.transpose(0, 1),
            ]),
            'validation': None,
            'test': None,
            'train_simsiam': None,
        }

    elif args.dataset in ['cifar10_36', 'cifar10_226', 'cifar100_224', 'tiny_imagenet_224']:
        trans = {
            'train_aug': transforms.Compose([
                to_rgb_ifneeded,
                transforms.RandomResizedCrop(args.img_height),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'train': transforms.Compose([
                to_rgb_ifneeded,
                transforms.Resize(args.img_height),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'validation': transforms.Compose([
                to_rgb_ifneeded,
                transforms.Resize(args.img_height),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'test': transforms.Compose([
                to_rgb_ifneeded,
                transforms.Resize(args.img_height),
                transforms.ToTensor(),
                gray2rgb_ifneeded,
                rgba2rgb_ifneeded,
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'train_simsiam': get_simsiam_transform(args),
        }
        trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
            'train_simsiam': None,
        }
        patch_trans = {
            'train': None,
            'train_aug': transforms.Compose([
                lambda x: x.transpose(0, 1),
                PerPatchAug(transforms.RandomCrop(int(np.round(args.patch_size / 1.15)))), # for patch_size = 32 -> 28
                transforms.Resize(args.patch_size),
                PerPatchAug(
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8)
                ),
                PerPatchAug(transforms.RandomGrayscale(p=0.2)),
                lambda x: x.transpose(0, 1),
            ]),
            'validation': None,
            'test': None,
            'train_simsiam': None,
        }

    elif args.dataset in ['cifar10', 'cifar100', 'tiny_imagenet', 'svhn']:
        # Cifar10 with small model
        # trans = {
        #     'train_aug': transforms.Compose([
        #         to_rgb_ifneeded,
        #         transforms.Pad(4),
        #         transforms.RandomCrop(34),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(*constants.NORMALIZATION[args.dataset])
        #     ]),
        #     'train': transforms.Compose([
        #         to_rgb_ifneeded,
        #         transforms.Pad(4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(34),
        #         transforms.ToTensor(),
        #         transforms.Normalize(*constants.NORMALIZATION[args.dataset])
        #     ]),
        #     'validation': transforms.Compose([
        #         to_rgb_ifneeded,
        #         transforms.Pad(4),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomCrop(34),
        #         transforms.ToTensor(),
        #         transforms.Normalize(*constants.NORMALIZATION[args.dataset])
        #     ]),
        #     'test': transforms.Compose([
        #         to_rgb_ifneeded,
        #         transforms.ToTensor(),
        #         transforms.Normalize(*constants.NORMALIZATION[args.dataset])
        #     ]),
        #     'train_simsiam': get_simsiam_transform(args),
        # }
        trans = {
            'train_aug': transforms.Compose([
                to_rgb_ifneeded,
                transforms.RandomCrop(args.img_height, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'train': transforms.Compose([
                to_rgb_ifneeded,
                transforms.ToTensor(),
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'validation': transforms.Compose([
                to_rgb_ifneeded,
                transforms.ToTensor(),
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'test': transforms.Compose([
                to_rgb_ifneeded,
                transforms.ToTensor(),
                transforms.Normalize(*constants.NORMALIZATION[args.dataset])
            ]),
            'train_simsiam': get_simsiam_transform(args),
        }
        trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
            'train_simsiam': None,
        }
        patch_trans = {
            'train': None,
            'train_aug': transforms.Compose([
                lambda x: x.transpose(0, 1),
                PerPatchAug(transforms.RandomCrop(int(np.round(args.patch_size / 1.15)))), # for patch_size = 4 -> 3
                transforms.Resize(args.patch_size),
                PerPatchAug(
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ], p=0.8)
                ),
                #PerPatchAug(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03)),
                PerPatchAug(transforms.RandomGrayscale(p=0.2)),
                lambda x: x.transpose(0, 1),
            ]),
            'validation': None,
            'test': None,
            'train_simsiam': None,
        }

    elif args.dataset == 'voc':
        raise Exception('TO CHECK...')
        #trans = {
        #    'train': transforms.Compose([
        #        lambda x: smart_resize(x, 255, 255),
        #        transforms.CenterCrop((255, 255)),
        #        transforms.ToTensor(),
        #        lambda x: image2patches(x, patch_size=crop_size[patch_size]),
        #        transforms.CenterCrop((patch_size, patch_size)),
        #    ]),
        #    'train_aug': transforms.Compose([
        #        lambda x: smart_resize(x, 255, 255),
        #        transforms.RandomCrop((255, 255)),
        #        transforms.ToTensor(),
        #        lambda x: image2patches(x, patch_size=crop_size[patch_size]),
        #        transforms.RandomCrop((patch_size, patch_size)),
        #    ]),
        #    'validation': transforms.Compose([
        #        lambda x: smart_resize(x, 255, 255),
        #        transforms.CenterCrop((255, 255)),
        #        transforms.ToTensor(),
        #        lambda x: image2patches(x, patch_size=crop_size[patch_size]),
        #        transforms.CenterCrop((patch_size, patch_size)),
        #    ]),
        #    'test': transforms.Compose([
        #        lambda x: smart_resize(x, 255, 255),
        #        transforms.CenterCrop((255, 255)),
        #        transforms.ToTensor(),
        #        lambda x: image2patches(x, patch_size=crop_size[patch_size]),
        #        transforms.CenterCrop((patch_size, patch_size)),
        #    ]),
        #}
        #trans_target = {
        #    'train': lambda x: -1,
        #    'train_aug': lambda x: -1,   
        #    'validation': lambda x: -1,   
        #    'test': lambda x: -1,   
        #}

    else:
        print('No transformations match. No transformations are applied to the datasets.')
        trans = {
            'train': None,
            'train_aug': None,
            'validation': None,
            'test': None,
            'train_simsiam': None,
        }

        trans_target = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
            'train_simsiam': None,
        }
        patch_trans = {
            'train': None,
            'train_aug': None,   
            'validation': None,   
            'test': None,   
            'train_simsiam': None,
        }

    return trans, trans_target, patch_trans

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_simsiam_transform(args):
    augmentation = [
        transforms.RandomResizedCrop(args.img_height, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*constants.NORMALIZATION[args.dataset])
    ]
    transform = TwoCropsTransform(transforms.Compose(augmentation))
    return transform

def create_megapatches_grid(x, side, patch_size=32):
    """
    Inputs:
        x: image torch tensor of shape [c, h, w].
        side: num_patches along the height (or with).
        patch_size: int, patch size dimension.
    Outputs:
        x_patches: torch tensor of shape [c, side, side, patch_size, patch_size].
    """
    C, H, W = x.shape
    side_in = int(H / patch_size)
    resize = transforms.Resize((patch_size, patch_size))
    idxs_w = torch.randperm(side_in - 2)[:side - 1] + 1 # [side]
    idxs_w = torch.sort(torch.cat([idxs_w, torch.tensor([side_in])]))[0]
    idxs_w = (torch.cat([idxs_w, torch.tensor([0])]) - torch.cat([torch.tensor([0]), idxs_w]))[:-1]
    idxs_w = idxs_w * patch_size
    
    idxs_h = torch.randperm(side_in - 2)[:side - 1] + 1 # [side]
    idxs_h = torch.sort(torch.cat([idxs_h, torch.tensor([side_in])]))[0]
    idxs_h = (torch.cat([idxs_h, torch.tensor([0])]) - torch.cat([torch.tensor([0]), idxs_h]))[:-1]
    idxs_h = idxs_h * patch_size

    x_rows = torch.split(x, list(idxs_h.numpy()), dim=1)
    x_patches = []
    for x_row in x_rows:
        x_cols = torch.split(x_row, list(idxs_h.numpy()), dim=2)
        x_patches_row = []
        for x_patch in x_cols:
            x_patch = resize(x_patch) # [c, h, w]
            x_patches_row += [x_patch]
        x_patches_row = torch.stack(x_patches_row, 1)
        x_patches += [x_patches_row]
    x_patches = torch.stack(x_patches, 1)
    x = patches2image(x_patches.reshape(C, -1, patch_size, patch_size), (side * patch_size, side * patch_size))
    return x
