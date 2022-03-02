# imports 
import os, time, math, random, json
from PIL import Image
import pytorch_lightning as pl
# torch imports
import torch
from torchvision import transforms
import torch.optim as optim
# torchvision
import torchvision
# cv2
import cv2
# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
from PIL import Image
# pandas
import pandas as pd
# glob (for path resolution with regexp)
from glob import glob
# sklearn
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
# skimage
import skimage
import skimage.transform
from skimage import img_as_ubyte, img_as_float32
# tqdm (for outputting a progress bar)
from tqdm import tqdm

# Generic config object
class Config():
    VOC_YEAR = '2007'
    VOC_TRAIN_PATH = '/VOC_train'
    VOC_TRAINVAL_PATH = '/VOC_trainval'
    VOC_TEST_PATH = '/VOC_test'
    DATASET_PATH = './dataset'
    SAVING_PATH = './models'

# Neural Network base model config object
class BaseModelConfig():
    patch_dim = 96
    gap = 48
    jitter = 7
    batch_size = 64
    num_epochs = 300
    lr = 0.0001