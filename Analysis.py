from importlib.resources import path
import torch
from torchvision import transforms
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
# from captum.attr import LayerGradCam
# from captum.attr import LayerAttribution
# from matplotlib.colors import LinearSegmentedColormap
# from captum.attr import visualization as viz

def getConvFilters(model):
        ## GET CONV LAYER FROM MODEL 
        # we will save the conv layer weights in this list
        model_weights =[]
        #we will save the 49 conv layers in this list
        conv_layers = []
        # get all the model children as list
        model_children = list(list(model.children())[0].children())[0] # baseline model π
        #counter to keep count of the conv layers
        counter = 0
        #append all the conv layers and their respective wights to the list
        for i in range(len(model_children)):    
            if type(model_children[i]) == torch.nn.Conv2d:
                counter+=1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == torch.nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == torch.nn.Conv2d:
                            counter+=1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print(f"Total convolution layers: {counter}")
        print("conv_layers")
        # last_conv = conv_layers[-1]
        return conv_layers, model_weights

class Analysis():
    def __init__(self, dataloader, model, path):
        self.dataloader = dataloader
        self.model = model
        self.path = path
        self.conv_filters, self.weights = getConvFilters(model)
        self.image, self.image_label = next(iter(dataloader))

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def features_maps(self):
        # make a copy of the `results`
        outputs = []
        names = []
        image_to_process = self.image
        for layer in self.conv_layers[0:]:
            image_to_process = layer(image_to_process)
            outputs.append(image_to_process)
            names.append(str(layer))
        print(len(outputs))
        processed = []
        for feature_map in outputs:
            feature_map = feature_map.squeeze(0)
            gray_scale = torch.sum(feature_map,0)
            gray_scale = gray_scale / feature_map.shape[0]
            processed.append(gray_scale.data.cpu().numpy())

        fig = plt.figure(figsize=(15, 10))
        for i in range(len(processed)):
            a = fig.add_subplot(5, 11, i+1)
            imgplot = plt.imshow(processed[i])
            a.axis("off")
            a.set_title(names[i].split('(')[0], fontsize=15)
        plt.savefig(str(f'{path}-feature_maps.jpg'), bbox_inches='tight') # ONLY FOR SAVE THE IMAGE