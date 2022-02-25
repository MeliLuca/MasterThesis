from Utils import *
from globalSettings import *

def load_data():
    if not os.path.exists(Config.SAVING_PATH):
        os.makedirs(Config.SAVING_PATH)

    if not os.path.exists(Config.DATASET_PATH):
        os.makedirs(Config.DATASET_PATH)   
    if not os.path.exists(f'{Config.DATASET_PATH}/{Config.VOC_TRAIN_PATH}'):
        os.makedirs(f'{Config.DATASET_PATH}/{Config.VOC_TRAIN_PATH}')    
    if not os.path.exists(f'{Config.DATASET_PATH}/{Config.VOC_TRAINVAL_PATH}'):
        os.makedirs(f'{Config.DATASET_PATH}/{Config.VOC_TRAINVAL_PATH}')  
    if not os.path.exists(f'{Config.DATASET_PATH}/{Config.VOC_TEST_PATH}'):
        os.makedirs(f'{Config.DATASET_PATH}/{Config.VOC_TEST_PATH}')

    train_data = torchvision.datasets.VOCDetection(f'{Config.DATASET_PATH}/{Config.VOC_TRAIN_PATH}', Config.VOC_YEAR, 'train', True)

    trainval_data = torchvision.datasets.VOCDetection(f'{Config.DATASET_PATH}/{Config.VOC_TRAINVAL_PATH}', Config.VOC_YEAR, 'trainval', True)

    test_data = torchvision.datasets.VOCDetection(f'{Config.DATASET_PATH}/{Config.VOC_TEST_PATH}', Config.VOC_YEAR, 'test', True)

    train_df = create_train_dataset(train_data)
    train_val= create_train_dataset(trainval_data)
    return train_df, train_val, test_data

#############################
# Creating training dataset
#############################
def create_train_dataset(train_data):
    training_set_list = []

    for idx in range(0, len(train_data)):
        img, label = train_data[idx]
        img = resize_image(img)
        
        training_set_list.append([img])

    print(f'Total training set elements: {len(training_set_list)}')

    dataframe = pd.DataFrame(data=training_set_list, columns=['image'])

    df_trn = dataframe
    return df_trn

#############################
# Creating validation dataset
#############################
def create_val_dataset(trainval_data):
    validation_set_list = []

    for idx in range(0, len(trainval_data)):
        img, label = trainval_data[idx]
        img = resize_image(img)
        
        validation_set_list.append([img])
            
    dataframe = pd.DataFrame(data=validation_set_list, columns=['image'])

    df_val = dataframe
    return df_val

#########################################
# This class generates patches for training
#########################################

class OurDataset(torch.utils.data.Dataset):
    def __init__(self, patch_dim, gap, df, validate, transform=None, transform2 = None):
        self.patch_dim, self.gap = patch_dim, gap
        self.transform = transform
        self.transform2 = transform2

        if validate:
            self.train_data = df.values
        else:
            self.train_data = df.values
        
    def get_patch_from_grid(self, image, patch_dim, gap):        
        offset_y, offset_x = image.shape[0] - (BaseModelConfig.jitter * 2) - (patch_dim*3 + gap*2), image.shape[1] - (BaseModelConfig.jitter * 2) - (patch_dim*3 + gap*2)
        
        start_grid_x, start_grid_y = np.random.randint(0, offset_x) + BaseModelConfig.jitter, np.random.randint(0, offset_y + 1) + BaseModelConfig.jitter
                
        patch_loc_arr = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]
        loc = np.random.randint(len(patch_loc_arr))
        tempx, tempy = patch_loc_arr[loc]
        
        # jitter
        jitter_x = np.random.randint(- BaseModelConfig.jitter, + BaseModelConfig.jitter + 1)
        jitter_y = np.random.randint(- BaseModelConfig.jitter, + BaseModelConfig.jitter + 1)

        patch_x_pt = start_grid_x + patch_dim * (tempx-1) + gap * (tempx-1) + jitter_x
        patch_y_pt = start_grid_y + patch_dim * (tempy-1) + gap * (tempy-1) + jitter_y
        
        random_patch = image[patch_y_pt:patch_y_pt+patch_dim, patch_x_pt:patch_x_pt+patch_dim]

        # jitter
        jitter_x = np.random.randint(- BaseModelConfig.jitter, + BaseModelConfig.jitter + 1)
        jitter_y = np.random.randint(- BaseModelConfig.jitter, + BaseModelConfig.jitter + 1)
        
        patch_x_pt = start_grid_x + patch_dim * (2-1) + gap * (2-1) + jitter_x
        patch_y_pt = start_grid_y + patch_dim * (2-1) + gap * (2-1) + jitter_y
        
        uniform_patch = image[patch_y_pt:patch_y_pt+patch_dim, patch_x_pt:patch_x_pt+patch_dim]

        random_patch_label = loc

        return uniform_patch, random_patch, random_patch_label
    
    def __len__(self):
        return len(self.train_data)
    
    def __getitem__(self, index):
        image = self.train_data[index]
        
        image = np.copy(image)
        
        uniform_patch, random_patch, random_patch_label = self.get_patch_from_grid(image, self.patch_dim, self.gap)

        # save original patches
        uniform_patch_original = copy.deepcopy(uniform_patch)
        random_patch_original = copy.deepcopy(random_patch)   
        
        if uniform_patch.shape[0] != 96:
            uniform_patch = skimage.transform.resize(uniform_patch, (96, 96))
            random_patch = skimage.transform.resize(random_patch, (96, 96))
            uniform_patch_original = skimage.transform.resize(uniform_patch_original, (96, 96))
            random_patch_original = skimage.transform.resize(random_patch_original, (96, 96))

            uniform_patch = img_as_float32(uniform_patch)
            random_patch = img_as_float32(random_patch)
            uniform_patch_original = img_as_float32(uniform_patch_original)
            random_patch_original = img_as_float32(random_patch_original)

            
        # Dropped color channels 2 and 3 and replaced with gaussian noise(std ~1/100 of the std of the remaining channel)
        uniform_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]), (uniform_patch.shape[0],uniform_patch.shape[1]))
        uniform_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(uniform_patch[:, :, 0]), (uniform_patch.shape[0],uniform_patch.shape[1]))
        random_patch[:, :, 1] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]), (random_patch.shape[0],random_patch.shape[1]))
        random_patch[:, :, 2] = np.random.normal(0.485, 0.01 * np.std(random_patch[:, :, 0]), (random_patch.shape[0],random_patch.shape[1]))

        random_patch_label = np.array(random_patch_label).astype(np.int64)

        if self.transform:
            uniform_patch = self.transform(uniform_patch)
            random_patch = self.transform(random_patch)
            uniform_patch_original = self.transform(uniform_patch_original)
            random_patch_original = self.transform(random_patch_original)
            
        return uniform_patch, random_patch, random_patch_label, uniform_patch_original, random_patch_original

##################################################
# Creating Train/Validation dataset and dataloader
##################################################
def get_train_load_obj(df_trn):     
    traindataset = OurDataset(BaseModelConfig.patch_dim, BaseModelConfig.gap, df_trn['image'], False,
                            transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                    std=[0.229, 0.224, 0.225])]), 
                            transforms.Compose([transforms.ToTensor()]))
    trainloader = torch.utils.data.DataLoader(traindataset, 
                                            batch_size=BaseModelConfig.batch_size,
                                            shuffle=True,
                                            num_workers=Config.num_workers
                                            )
    return traindataset, trainloader

def get_val_load_obj(df_val):
    valdataset = OurDataset(BaseModelConfig.patch_dim, BaseModelConfig.gap, df_val['image'], True,
                            transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                                    std=[0.229, 0.224, 0.225])]), 
                            transforms.Compose([transforms.ToTensor()]))
    valloader = torch.utils.data.DataLoader(valdataset,
                                            batch_size=BaseModelConfig.batch_size,
                                            shuffle=False,
                                            num_workers=Config.num_workers
                                            )
    return valdataset, valloader
