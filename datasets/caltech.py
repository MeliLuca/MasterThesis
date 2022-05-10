import os
import tarfile
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
import logging

import utils


class Caltech256(Dataset):
    """
    Custom implementation of Caltech256 since the pytorch official implementatio is currently broken.
    Info of the dataset at: http://www.vision.caltech.edu/Image_Datasets/Caltech256/
    """
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        files_already_downloaded = self._check_files()
        if (not files_already_downloaded) and (not self.download):
            logging.error('Dataset not found. You can download it using "download=True".')
        if self.download:
            if files_already_downloaded:
                logging.info('Files already downloaded.')
            else:
                self._download()
        self.idxs = self._read_idxs(os.path.join(self.root, 'caltech256', 
                                                 f'caltech256_{"train" if self.train else "val"}.txt'))
        ds_full = ImageFolder(os.path.join(self.root, 'caltech256', '256_ObjectCategories'))
        self.ds_sub = Subset(ds_full, self.idxs)

    def __getitem__(self, idx):
        x, y = self.ds_sub.__getitem__(idx)
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.ds_sub)

    def _download(self):
        ds_path = os.path.join(self.root, 'caltech256')
        if not os.path.exists(ds_path):
            os.makedirs(ds_path)
        logging.info('Downloading caltech256...')
        utils.download_file_from_google_drive('1IYEMfrB_4jcd0sKXBeLMG_f6iaX6UXKV', 
            os.path.join(ds_path, '256_ObjectCategories.tar')) # Images
        utils.download_file_from_google_drive('19fqjP8k6BeAafOZDdQaThUM5eWQJOGQA', 
            os.path.join(ds_path, 'caltech256_train.txt')) # Train split
        utils.download_file_from_google_drive('1uZRsIbkFOp8tco688jgkT1Ejy-ax7Ony', 
            os.path.join(ds_path, 'caltech256_val.txt')) # Val split
        logging.info('Extracting caltech256...')
        tar = tarfile.open(os.path.join(ds_path, '256_ObjectCategories.tar'))
        tar.extractall(ds_path)
        tar.close()

    def _read_idxs(self, path):
        with open(path, 'r') as f:
            idxs = [int(l.strip()) for l in f.readlines()]
        return idxs

    def _check_files(self):
        return os.path.exists(os.path.join(self.root, 'caltech256', '256_ObjectCategories.tar'))

