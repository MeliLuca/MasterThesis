import requests
import torch
from tqdm import tqdm
import tarfile
import os
from torch.utils.data import Dataset
import logging
import scipy.io
from PIL import Image
from torchvision import transforms
import numpy as np
import scipy.io
import utils

urls = {
    'url_images_train': 'http://ai.stanford.edu/~jkrause/car196/cars_train.tgz',
    'url_images_test': 'http://ai.stanford.edu/~jkrause/car196/cars_test.tgz',
    'url_annotations_train': 'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz',
    'url_annotations_test': 'http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat',
}

class Cars(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.root = root
        self.data_path = os.path.join(self.root, 'cars')
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self._setup_data()

        self.annotations = self._read_split()

    def _check_files(self):
        files = [
            os.path.exists(os.path.join(self.data_path, os.path.basename(url)))
            for url in urls.values()
        ]
        return all(files)

    def _setup_data(self):
        data_already_downloaded = self._check_files()

        if self.download:
            if data_already_downloaded:
                logging.info('Files already downloaded.')
            else:
                for key, url in urls.items():
                    fname = os.path.basename(url)
                    utils.download_file(url, os.path.join(self.data_path, fname))
                    if fname.endswith('.tgz'):
                        utils.extract_tar(os.path.join(self.data_path, fname))
        else:
            if not data_already_downloaded:
                logging.error('Data not found. You can download it passing "download=True".')

    def _read_split(self):
        annotations = {'bbox': [], 'class': [], 'fname': []}
        if self.train:
            path = os.path.join(self.data_path, 'devkit', 'cars_train_annos.mat')
        else:
            path = os.path.join(self.data_path, 'cars_test_annos_withlabels.mat')
        data = scipy.io.loadmat(path)['annotations'][0]
        for a in data:
            x1, x2, y1, y2, cl, fname = a
            annotations['bbox'] += [[int(x1), int(x2), int(y1), int(y2)]]
            annotations['class'] += [int(cl) - 1]
            annotations['fname'] += [fname[0]]
        return annotations

    def __len__(self):
        return len(self.annotations['class'])

    def __getitem__(self, idx):
        image_name = self.annotations['fname'][idx]
        img_folder = 'cars_train' if self.train else 'cars_test'
        image = Image.open(os.path.join(self.data_path, img_folder,  image_name))
        if self.transform is not None:
            image = self.transform(image)
        label = self.annotations['class'][idx]
        if self.target_transform is not None:
            label = self.target_transform(labels)
        return image, label
