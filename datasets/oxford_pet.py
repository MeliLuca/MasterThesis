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
import utils

urls = {
    'url_images': 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz',
    'url_img_labels': 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz',
}


class OxfordPet(Dataset):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.root = root
        self.data_path = os.path.join(self.root, 'oxford_pet')
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self._setup_data()

        self.annotations = self._read_split()
        self.labels = self._get_labels_set(self.annotations)

    def _get_labels_set(self, annotations):
        labels_set = sorted(list(set([f'{cl:02d}_{cl_str}' for cl_str, cl in zip(annotations['class'], annotations['class_id'])])))
        labels_set = ['_'.join(l.split('_')[1:]) for l in labels_set]
        return labels_set

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
                    utils.extract_tar(os.path.join(self.data_path, fname))
        else:
            if not data_already_downloaded:
                logging.error('Data not found. You can download it passing "download=True".')

    def _read_split(self):
        path = os.path.join(self.data_path, 'annotations/trainval.txt') if self.train else os.path.join(self.data_path, 'annotations/test.txt')
        annotations = {'class': [], 'class_id': [], 'species': [], 'breed_id': [], 'fname': []}
        with open(path, 'r') as f:
            for l in f.readlines():
                cl_str, cl, species, breed_id = l.strip().split()
                annotations['fname'] += [cl_str]
                cl_str = '_'.join(cl_str.split('_')[:-1])
                cl = int(cl) - 1
                species = int(species)
                breed_id = int(breed_id)
                annotations['class'] += [cl_str]
                annotations['class_id'] += [cl]
                annotations['species'] += [species]
                annotations['breed_id'] += [breed_id]
        return annotations

    def __len__(self):
        return len(self.annotations['class_id'])

    def __getitem__(self, idx):
        image_name = self.annotations['fname'][idx]
        image = Image.open(os.path.join(self.data_path, 'images',  f'{image_name}.jpg'))
        if self.transform is not None:
            image = self.transform(image)
        label = self.annotations['class_id'][idx]
        if self.target_transform is not None:
            label = self.target_transform(labels)
        return image, label

    
