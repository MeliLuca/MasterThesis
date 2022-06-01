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
from Models import resnet_cifar10, resnet

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

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
        self.avg_loss = []
        self.avg_acc = []
        # Create model
        if self.args.backbone == 'cifar10_model':
            self.model = resnet_cifar10.resnet56()
        else :
            # self.model = resnet.ResNet50(use_relations=args.use_relations)
            self.model = torchvision.models.resnet50(pretrained = self.args.pretrained)
        if self.args.use_relations :
            self.avg_loss_rel = []
            self.avg_acc_rel = []
            self.rel_layer = MultiHeadAttention.MultiHeadAttentionScores(64,9)
            self.class_layer = torch.nn.Sequential((list(self.model.children())[-1]))
            self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
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
        if self.args.backbone == 'cifar10_model':
            scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=0.1, total_iters=40)
            return ([optimizer], [scheduler])

        return optimizer 

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs = batch['images']
        labels = batch['labels']
        loss=0
        preds = self.model(imgs)
        if self.args.use_relations :
            target = batch['relations']
            rel_preds = self.rel_layer(preds)
            loss_rel = cross_entropy(rel_preds, target)
            acc_rel = accuracy(rel_preds, target)
            self.log(f'acc_rel', acc_rel, prog_bar=True)
            self.log(f'loss_rel', loss_rel.item())

            ## print metrics
            self.avg_loss_rel += [loss_rel]
            self.avg_acc_rel += [acc_rel]

            loss += loss_rel.item()
            preds = F.avg_pool2d(preds, preds.size()[3])
            preds = preds.view(preds.size(0), -1)
            preds = self.class_layer(preds)

        loss += self.loss_module(preds, labels)
        acc = accuracy(preds, labels)
        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc)
        self.log("train_loss", loss)
        
        self.avg_loss += [loss]
        self.avg_acc += [acc]

        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs = batch['images']
        labels = batch['labels']
        preds = self.model(imgs)
        print('#### PROOOBLEEMIIIII')
        print(f'labels shape is {labels.shape}')
        print(f'pred shape is {preds.shape}')
        print(f'Am i use relations {self.args.use_relations}')
        if self.args.use_relations :
            preds = F.avg_pool2d(preds, preds.size()[3])
            preds = preds.view(preds.size(0), -1)
            preds = self.class_layer(preds)
            print(preds.shape)
        loss = self.loss_module(preds, labels)
        acc = accuracy(preds, labels)
        # By default logs it per epoch (weighted average over batches)
        self.log("val_loss", loss)
        self.log("val_acc", acc)


    def test_step(self, batch, batch_idx):
        imgs = batch['images']
        labels = batch['labels']
        preds = self.model(imgs)
        if self.args.use_relations :
            preds = F.avg_pool2d(preds, preds.size()[3])
            preds = self.class_layer(preds)
        loss = self.loss_module(preds, labels)
        acc = accuracy(preds, labels)
        # By default logs it per epoch (weighted average over batches), and returns it afterwards
        self.log("test_loss", loss)
        self.log("test_acc", acc)
    
    # def training_epoch_end(self):
    #     loss = self.avg_loss.loss.mean()
        
    #     result.log('train_loss', avg_loss, on_epoch=True, prog_bar=True)

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
    torch.backends.cudnn.deterministic = True
    torch.set_deterministic_debug_mode()
    model_args = {
        'args': args,
        # 'num_classes': args.num_classes,
        # 'num_rel_classes': args.num_rel_classes,
        # 'use_relations': args.use_relations,
        # 'use_supervision': args.use_supervision,
        # 'dropout': args.dropout,
        # 'freeze_fet_enc': args.freeze_fet_enc,
        # 'proj_dropout': args.proj_dropout,
    }
    model = ResNetModule(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, strict=False, **model_args)
        print(f'Loaded model at "{args.model_checkpoint}"')
    return model
