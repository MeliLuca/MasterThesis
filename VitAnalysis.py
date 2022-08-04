from cProfile import label
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
    
    def get_prediction(self):
        print('######  Eval model')
        self.model.eval()
        print('#### start to compute the prediction')
        pred = self.model(self.image)['seg'].detach().numpy()
        pred = np.argmax(pred, axis=1)
        print('##### start to print the predictioin]')
        plt.figure(figsize=(20, 20))
        plt.title('Origin - Prediction - Mask')
        plt.subplot(1, 1, 1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        print(self.image.shape)
        plt.imshow(np.transpose(self.image.detach().numpy(), (1, 2, 0)))
        plt.subplot(2, 1, 2) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(np.transpose(pred, (1, 2, 0)))
        plt.subplot(3, 1, 3) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
        plt.imshow(np.transpose(self.image_label, (1, 2, 0)))
        print('######  saving the prediction')
        str_path = path + '_prediction'
        plt.savefig(str_path)
        return 