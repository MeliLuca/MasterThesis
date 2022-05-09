import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
import logging
from Utils import cross_entropy, accuracy, get_rel_matrix
from matplotlib import cm
import numpy as np
from PIL import Image



# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNetCifar10(nn.Module):
    def __init__(self, block, layers, args, use_relations=False):
        super(ResNetCifar10, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, args.num_classes)
        self.learning_rate = args.leargning_rate
        self.use_relations = use_relations
        if(use_relations):
            self.rel_matrix = get_rel_matrix(args)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        if (self.use_relations):
            out = self.rel_head(out)
        else :
            out = self.avg_pool(out)
            out = out.view(out.size(0), -1)
            out = self.fc(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                           mode='min',
                                           patience=5,
                                           factor=0.3, verbose=True)
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler,
           'monitor': 'val_loss'
       }

    def training_step(self, batch, batch_idx):
        images, labels = batch[0], batch[1]
        # Forward pass
        outputs = self(images)
        if self.use_relations :
            target = self.rel_matrix.unsqueeze(0).repeat(outputs.shape[0], 1, 1)
            loss = cross_entropy(outputs, target)
        else :    
            loss = criterion(outputs, labels)
            self.log("train_loss", loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        uniform_patch, random_patch, random_patch_label = batch[0], batch[1], batch[2]
        output, _, _ = self(uniform_patch, random_patch)
        val_loss = torch.nn.functional.cross_entropy(output, random_patch_label)
        del uniform_patch, random_patch, random_patch_label
        self.log("val_loss", val_loss.item())

    def test_step(self, batch, batch_idx):
        uniform_patch, random_patch, random_patch_label = batch[0], batch[1], batch[2]
        output, _, _ = self(uniform_patch, random_patch)
        test_loss = torch.nn.functional.cross_entropy(output, random_patch_label)
        del uniform_patch, random_patch, random_patch_label
        self.log("test_loss", test_loss.item())


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Train the model
total_step = len(train_loader)
curr_lr = LEARNING_RATE
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), f'resnet_solo_{NUM_EPOCHS}_{LEARNING_RATE}_{BATCH_SIZE}_{DATASET_NAME}.ckpt')