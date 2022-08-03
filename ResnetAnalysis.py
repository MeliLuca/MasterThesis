from importlib.resources import path
from unittest import case
import torch
from torchvision import transforms
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# import cv2 as cv
from captum.attr import LayerGradCam
from captum.attr import LayerAttribution
from matplotlib.colors import LinearSegmentedColormap
from captum.attr import visualization as viz
from captum.attr import GuidedGradCam

def getConvFilters(model):
        ## GET CONV LAYER FROM MODEL 
        # we will save the conv layer weights in this list
        model_weights =[]
        #we will save the 49 conv layers in this list
        conv_layers = []
        # get all the model children as list
        model_children = list(list(model.children())[0].children()) # baseline model π
        #counter to keep count of the conv layers
        counter = 0
        #append all the conv layers and their respective wights to the list
        for i in range(len(model_children)):    
            if type(model_children[i]) == torch.nn.Conv2d:
                counter+=1
                model_weights.append(model_children[i].weight.detach())
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == torch.nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == torch.nn.Conv2d:
                            counter+=1
                            model_weights.append(child.weight.detach())
                            conv_layers.append(child)
        # print(f"Total convolution layers: {counter}")
        # print("conv_layers")
        # last_conv = conv_layers[-1]
        return conv_layers, model_weights

def filters(path, weights, num_layer=None):
        # visualize the first conv layer filters
        def print_filters(layer, idx):
            nomralized_filters = (layer - layer.min()) / (layer.max() - layer.min())
            plt.figure(figsize=(20, 17))
            plt.title(f'filter-{path}', fontsize=20)
            for j, filter in enumerate(nomralized_filters):
                plt.subplot(8, 8, j+1) # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.imshow(filter[0, ...])
                plt.axis('off')
                plt.savefig(f'filters/{idx+1}_layer-{path}_v2.png')

        if num_layer == None :
            for i, layer in enumerate(weights):
                print_filters(layer, i)
        else :
            print_filters(weights[num_layer], num_layer)
                

def features_map(image, conv_layers, path):
    # make a copy of the `results`
    outputs = []
    names = []
    image_to_process = image
    for layer in conv_layers[0:]:
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
    file_name = f'{path}-feature_maps.jpg'
    plt.savefig(str(file_name), bbox_inches='tight') # ONLY FOR SAVE THE IMAGE

def guided_grad_cam(model, conv_layers, image, image_label, path):
    model.eval()
    # pred = self.model(self.image).argmax(dim=1)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                [(0, '#ffffff'),
                                                (0.25, '#000000'),
                                                (1, '#000000')], N=256)

    guided_gc = GuidedGradCam(model, conv_layers[-1])
    attribution_gc = guided_gc.attribute(image, image_label, additional_forward_args=[True])
    printer_image = np.transpose((image.squeeze().cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))
    print('start to print stuff')
    printer_attr_gc = np.transpose(attribution_gc.squeeze(0).cpu().detach().numpy(), (1,2,0))
    fig = viz.visualize_image_attr_multiple(
        printer_attr_gc, 
        printer_image,
        ["original_image", "heat_map"],
        ["all", "positive"], 
        cmap=default_cmap, 
        show_colorbar=True,
        outlier_perc=1
        )
    fig.savefig('prova.png', bbox_inches='tight') 

def prediction(model, image, image_label):
    model.eval()
    pred = model(image)

class Create():
    def __init__(self, model, dataloader, path):
        self.dataloader = dataloader
        self.model = model
        self.path = path
        self.conv_layers, self.weights = getConvFilters(model)
        self.data = next(iter(dataloader))
        self.image = self.data['images']
        self.image_label = self.data['labels'].item()

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    def get_type(self, param):
        func = param.split("-")
        if func[0] == 'filters' :
            return filters(self.path, self.weights, int(func[1])) if len(func)>1 else filters(self.path, self.weights)
        elif func[0] == 'features_map' :
            return features_map(self.image, self.conv_layers, self.path)
        elif func[0] == 'grad_cam' :
            return guided_grad_cam(self.model, self.conv_layers, self.image, self.image_label, self.path)
        elif func[0] == 'pred' :
            return prediction(self.model, self.image, self.image_label, self.path)