from cProfile import label
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
import torchvision
from PIL import Image
from Layers import MultiHeadAttention

class ResNetModule(pl.LightningModule):
    def __init__(self, args):
        """
        Inputs:
            model_name - Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams - Hyperparameters for the model, as dictionary.
            optimizer_name - Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams - Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.
        """
        super().__init__()
        # Exports the hyperparameters to a YAML file, and create "self.hparams" namespace
        self.save_hyperparameters()
        self.args = args
        # Create model
        self.model = torchvision.models.resnet50(pretrained=False)
        # Create loss module
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        # Forward function that is run when visualizing the graph
        return self.model(imgs)

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
        # "batch" is the output of the training data loader.
        imgs = batch['images']
        labels = batch['labels']
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = accuracy(preds, labels)

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs = batch['images']
        labels = batch['labels']
        print(imgs.shape)
        preds = self.model(imgs)
        acc = accuracy(preds, labels)
        # By default logs it per epoch (weighted average over batches)
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        imgs = batch['images']
        labels = batch['labels']
        preds = self.model(imgs)
        acc = accuracy(preds, labels)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_acc", acc)

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
    model = ResNetModule(args)
    # if len(args.model_checkpoint) > 0:
    #     model = model.load_from_checkpoint(args.model_checkpoint, strict=False, args)
    #     print(f'Loaded model at "{args.model_checkpoint}"')
    return model