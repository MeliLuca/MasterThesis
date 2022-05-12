from cgitb import reset
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
import logging
from utils import cross_entropy, accuracy
from matplotlib import cm
import numpy as np
from PIL import Image
from Layers import MultiHeadAttention



# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(pl.LightningModule):
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
class ResNetCifar10(pl.LightningModule):
    def __init__(self, block, layers, args, num_classes, num_rel_classes, use_relations=False, 
                 use_supervision=False, dropout=0.1, freeze_fet_enc=False, proj_dropout=0.0):
        super(ResNetCifar10, self).__init__()
        self.save_hyperparameters()
        self.args = args
        self.num_classes = num_classes
        self.use_supervision = use_supervision
        self.num_rel_classes = num_rel_classes
        self.use_relations = use_relations
        self.dropout = dropout
        self.proj_dropout = proj_dropout
        self.freeze_fet_enc = freeze_fet_enc
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        if self.use_relations:
            ## DA CAMBIAREEEE
            print(f'rel head viene creato con {num_rel_classes*num_rel_classes}, {num_rel_classes}')
            self.rel_head = MultiHeadAttention.MultiHeadAttentionScores(81, 9)
        else :
            self.avg_pool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64, args.num_classes)
        self.lr = args.lr
        self.use_relations = use_relations
       

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
        """
        Configure optimizer for the learning process.
        """
        if self.args.optimizer == 'adam':
            optimizer = Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            optimizer = SGD(self.parameters(), self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'adamw':
            optimizer = AdamW(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)
        else:
            raise Exception(f'Error. Optimizer "{self.args.optimizer}" not supported.')
        self.optimizer = optimizer
        return optimizer

    def training_step(self, batch, batch_idx):
        images = batch['images']
        # Forward pass
        outputs = self(images)
        if self.use_relations :
            target = batch['relations']
            loss = cross_entropy(outputs, target)
            acc_rel = accuracy(outputs, target)
            self.log(f'acc_rel', acc_rel, prog_bar=True)
            self.log(f'loss_rel', loss.item())
        else :    
            labels=batch['labels']
            loss = F.cross_entropy(outputs, labels)
            self.log("train_loss", loss.item())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # print('#### entro validation step')
        images = batch['images']
        # Forward pass
        outputs = self(images)
        if self.use_relations:
            target = batch['relations']
            # print(target)
            loss = F.cross_entropy(outputs, target)
            acc_rel = accuracy(outputs, target)
            self.log(f'val_acc', acc_rel, prog_bar=True)
            self.log(f'val_loss', loss.item())
        else :    
            labels=batch['labels']
            loss = F.cross_entropy(outputs, labels)
            self.log("val_loss", loss.item())
        return loss

    def test_step(self, batch, batch_idx):
        images = batch['images']
        # Forward pass
        outputs = self(images)
        labels=batch['labels']
        loss = F.cross_entropy(outputs, labels)
        acc = accuracy(outputs, labels)
        self.log("test_loss", loss.item())
        self.log("test_acc", acc.item())
        return loss
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    logging.info(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                logging.info(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

def get_model(args):
    model_args = {
        'args': args,
        'num_classes': args.num_classes,
        'num_rel_classes': 9,
        'use_relations': args.use_relations,
        'use_supervision': args.use_supervision,
        'dropout': args.dropout,
        'freeze_fet_enc': args.freeze_fet_enc,
        'proj_dropout': args.proj_dropout,
    }
    ResNetCifar10(ResidualBlock, [2, 2, 2], **model_args)
    model = ResNetCifar10(ResidualBlock, [2, 2, 2], **model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, strict=False, **model_args)
        print(f'Loaded model at "{args.model_checkpoint}"')
    return model

