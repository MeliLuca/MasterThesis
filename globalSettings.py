import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Generic config object
class Config():
    DATASET_NAME = 'cifar_dataset'
    DATASET_PATH = './dataset'
    SAVING_PATH = './models'
    PATH_NAME = 'resnet-standard'
    num_workers = 8

# Neural Network base model config object
class BaseModelConfig():
    batch_size = 64
    num_epochs = 300
    learning_rate = 0.0001