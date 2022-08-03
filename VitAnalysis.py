from importlib.resources import path
from unittest import case
from cv2 import imshow
import torch
from torchvision import transforms
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
# import cv2 as cv

class Create():
    def __init__(self, model, dataloader, path):
        print('#######  Init Create Prediction ########')
        self.dataloader = dataloader
        self.model = model
        self.path = path
        self.data = next(iter(dataloader))
        self.image = self.data['images']
        self.image_label = self.data['labels']

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        print('######  saving the prediction')
        plt.savefig(os.path.join(path, '-prediction'))
    
    def get_prediction(self):
        print('######  Eval model')
        self.model.eval()
        print('#### start to compute the prediction')
        pred = self.model(self.image)
        print('##### start to print the predictioin]')
        imshow(pred, self.path)
        return 