"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
import torch
import numpy as np
import logging
import zipfile
import pickle

from PIL import Image
from torch.utils.data import Dataset
from datasets.utils_ade20k import loadAde20K

url = 'http://groups.csail.mit.edu/vision/datasets/ADE20K/toolkit/index_ade20k.pkl'

class ADE20KSegmentation(Dataset):
    BASE_DIR = 'ADE20K_2021_17_01'
    NUM_CLASS = 150
   
    def __init__(self, root, split='train', mode='train', download=False, transform=None,
                 target_transform=None, use_segmentation=False,
                 seg_transform=None, **kwargs):
        self.root = root
        self.data_path = os.path.join(root, self.BASE_DIR)
        self.split = split
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.use_segmentation = use_segmentation
        self.seg_transform = seg_transform
        self.mode = mode

        # assert os.path.exists(root), "Please setup the dataset using ../datasets/ade20k.py"
        
        self.images, self.masks = _get_ade20k_pairs(self.data_path, self.split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")
        print('Found {} images in the folder {}'.format(len(self.images), root))

    def __getitem__(self, index):
        img_path = os.path.join(self.root,self.images[index])
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img
        mask = loadAde20K(img_path)['class_mask']
        # mask = Image.open(os.path.join(self.root,self.masks[index]))
        img = np.array(img)
        mask = np.array(mask)
        # general resize, normalize and to Tensor. 
        # 
        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask) #Albumentations Transform
            img = transformed["image"]
            mask = transformed["mask"]
        return img, mask

    def _mask_transform(self, mask):
        return torch.LongTensor(np.array(mask).astype('int32') - 1)

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 1

    @property
    def classes(self):
        """Category names."""
        return ("wall", "building, edifice", "sky", "floor, flooring", "tree",
                "ceiling", "road, route", "bed", "windowpane, window", "grass",
                "cabinet", "sidewalk, pavement",
                "person, individual, someone, somebody, mortal, soul",
                "earth, ground", "door, double door", "table", "mountain, mount",
                "plant, flora, plant life", "curtain, drape, drapery, mantle, pall",
                "chair", "car, auto, automobile, machine, motorcar",
                "water", "painting, picture", "sofa, couch, lounge", "shelf",
                "house", "sea", "mirror", "rug, carpet, carpeting", "field", "armchair",
                "seat", "fence, fencing", "desk", "rock, stone", "wardrobe, closet, press",
                "lamp", "bathtub, bathing tub, bath, tub", "railing, rail", "cushion",
                "base, pedestal, stand", "box", "column, pillar", "signboard, sign",
                "chest of drawers, chest, bureau, dresser", "counter", "sand", "sink",
                "skyscraper", "fireplace, hearth, open fireplace", "refrigerator, icebox",
                "grandstand, covered stand", "path", "stairs, steps", "runway",
                "case, display case, showcase, vitrine",
                "pool table, billiard table, snooker table", "pillow",
                "screen door, screen", "stairway, staircase", "river", "bridge, span",
                "bookcase", "blind, screen", "coffee table, cocktail table",
                "toilet, can, commode, crapper, pot, potty, stool, throne",
                "flower", "book", "hill", "bench", "countertop",
                "stove, kitchen stove, range, kitchen range, cooking stove",
                "palm, palm tree", "kitchen island",
                "computer, computing machine, computing device, data processor, "
                "electronic computer, information processing system",
                "swivel chair", "boat", "bar", "arcade machine",
                "hovel, hut, hutch, shack, shanty",
                "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, "
                "motorcoach, omnibus, passenger vehicle",
                "towel", "light, light source", "truck, motortruck", "tower",
                "chandelier, pendant, pendent", "awning, sunshade, sunblind",
                "streetlight, street lamp", "booth, cubicle, stall, kiosk",
                "television receiver, television, television set, tv, tv set, idiot "
                "box, boob tube, telly, goggle box",
                "airplane, aeroplane, plane", "dirt track",
                "apparel, wearing apparel, dress, clothes",
                "pole", "land, ground, soil",
                "bannister, banister, balustrade, balusters, handrail",
                "escalator, moving staircase, moving stairway",
                "ottoman, pouf, pouffe, puff, hassock",
                "bottle", "buffet, counter, sideboard",
                "poster, posting, placard, notice, bill, card",
                "stage", "van", "ship", "fountain",
                "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
                "canopy", "washer, automatic washer, washing machine",
                "plaything, toy", "swimming pool, swimming bath, natatorium",
                "stool", "barrel, cask", "basket, handbasket", "waterfall, falls",
                "tent, collapsible shelter", "bag", "minibike, motorbike", "cradle",
                "oven", "ball", "food, solid food", "step, stair", "tank, storage tank",
                "trade name, brand name, brand, marque", "microwave, microwave oven",
                "pot, flowerpot", "animal, animate being, beast, brute, creature, fauna",
                "bicycle, bike, wheel, cycle", "lake",
                "dishwasher, dish washer, dishwashing machine",
                "screen, silver screen, projection screen",
                "blanket, cover", "sculpture", "hood, exhaust hood", "sconce", "vase",
                "traffic light, traffic signal, stoplight", "tray",
                "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, "
                "dustbin, trash barrel, trash bin",
                "fan", "pier, wharf, wharfage, dock", "crt screen",
                "plate", "monitor, monitoring device", "bulletin board, notice board",
                "shower", "radiator", "glass, drinking glass", "clock", "flag")


def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    with open(os.path.join(folder, 'index_ade20k.pkl'), 'rb') as f:
        items = pickle.load(f)
    file_name_list = items['filename']
    folder = items['folder']
    for i, filename in enumerate(file_name_list):
        if 'train' in filename:
            img_paths += [os.path.join(folder[i], file_name_list[i])]
            mask_paths += [os.path.join(folder[i], file_name_list[i].replace('.jpg', '_seg.png'))]
    
    return img_paths, mask_paths