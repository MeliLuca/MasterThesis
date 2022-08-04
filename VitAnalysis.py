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
        self.images = self.data['images']
        self.images_label = self.data['labels']
        self.filename = self.data['filename']
    
    def get_prediction(self):
        print('######  Eval model')
        self.model.eval()
        B = len(self.images)
        batch_pred = self.model(self.images)['seg']
        print('#### start to compute the prediction')
        for i, img in enumerate (self.images) :
            print(f'before argmax pred is {batch_pred[i].shape}')
            pred = np.argmax(batch_pred[i].detach().numpy(), axis=1)
            print(f'after argmax prediction is {pred.shape}')
            print('##### start to print the predictioin]')
            fig = plt.figure(figsize=(100, 100))
            plt.title('Origin - Prediction - Mask')
            ax_1 = fig.add_subplot(B,3,i+1) 
            ax_1.set_title('Original')
            ax_1.imshow(np.transpose(img.detach().numpy(), (1, 2, 0)))
            ax_2 = fig.add_subplot(B,3,i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            ax_2.imshow(np.transpose(pred, (1, 2, 0)))
            ax_2.set_title('Prediction')
            ax_3 = fig.add_subplot(B,3,i+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
            ax_3.imshow(np.transpose(self.image_labels[i], (1, 2, 0)))
            ax_3.set_title('Mask')
        print('######  saving the prediction')
        str_path = self.path + '_prediction_' + self.filename[0].split('.')[1]
        plt.savefig(str_path)
        return 