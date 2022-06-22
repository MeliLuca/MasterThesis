##################################################
# Imports
##################################################

from ast import arg
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torchvision.models as models
import logging

# Custom
from Layers.MultiHeadAttention import MultiHeadAttentionScores
from datasets.encode_locations import create_connectivity_matrix
from utils import cross_entropy, accuracy
from Models import resnet_cifar10


##################################################
# RelCNN Model
##################################################

class RelCNN(pl.LightningModule):
    def __init__(self, args, num_classes, *bargs, num_rel_classes=9, use_relations=False, 
                 use_supervision=True, **bkwargs):
        super().__init__()
        self.args = args
        if args.backbone == 'cifar10_model':
            self.backbone = resnet_cifar10.__dict__['resnet56'](*bargs, **bkwargs)
        else : 
            self.backbone = models.__dict__[args.backbone](*bargs, **bkwargs)
        self.use_relations = use_relations
        self.use_supervision = use_supervision
        self.num_rel_classes = num_rel_classes
        if self.use_relations:
            #self.rel_head = nn.Conv2d(self.backbone.layer1[0].expansion * 512, 
            #                          self.num_rel_classes, kernel_size=1)
            self.rel_head = MultiHeadAttentionScores(list(self.backbone.children())[-1].in_features, num_rel_classes)
        if self.use_angles:
            self.angle_head = MultiHeadAttentionScores(list(self.backbone.children())[-1].in_features, 1)

    def forward_feat(self, x):
        """
        Compute the features map feat of shape [bs, d_c, d_h, d_w].
        """
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if self.args.backbone != 'cifar10_model':
            x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        if self.args.backbone != 'cifar10_model':
            x = self.backbone.layer4(x)
        return x

    def forward_feat_with_rel(self, x):
        x = self.forward_feat(x)
        B, C, H, W = x.shape
        out={}
        x_att = torch.flatten(x, 2).transpose(1, 2)
        if self.use_relations:
            x_rel = self.rel_head(x_att)
            out['rel'] = x_rel.transpose(1, 2).reshape(B, self.num_rel_classes, H * H, W * W)

        if self.use_angles:
            x_angle = self.angle_head(x_att)
            out['angle'] = x_angle.transpose(1, 2).reshape(B, self.num_rel_classes, H * H, W * W)

        if self.args.backbone == 'cifar10_model':
            x_clf= self.backbone.avg_flat(x)

        else :
            x_clf = self.backbone.avgpool(x)
            x_clf = torch.flatten(x_clf, 1)
        out['clf'] = self.backbone.fc(x_clf)
        return out

    def forward(self, x):
        out = {}
        if self.use_relations or self.use_angles:
            out = self.forward_feat_with_rel(x)
        else:
            x = self.forward_feat(x)
            x = self.backbone.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.backbone.fc(x)
            out['clf'] = x
        return out

    def configure_optimizers(self):
        """                                                                                                             
        Configure optimizer for the learning process.                                                                   
        """                                                                                                             
        if self.args.optimizer == 'adam':                                                                               
            optimizer = Adam(self.parameters(), self.args.lr)                                                           
        elif self.args.optimizer == 'sgd':                                                                              
            optimizer = SGD(self.parameters(), self.args.lr, momentum=0.9, weight_decay=5e-5)                           
        else:                                                                                                           
            raise Exception(f'Error. Optimizer "{self.args.optimizer}" not supported.')                                 
        self.optimizer = optimizer                                                                                      
        return optimizer 

    def training_step(self, batch, batch_idx, part='train'):
        """
        Define a single training step.
        """

        # Retrieve data
        x = batch['images'] # [B, C, H, W]

        # Forward and loss
        logits = self(x)
        loss = 0.0

        if self.use_supervision:
            y_cls = batch['labels'] # [B]
            torch.use_deterministic_algorithms(False)
            loss_clf = F.cross_entropy(logits['clf'], y_cls)
            torch.use_deterministic_algorithms(True)
            loss += loss_clf
            acc_cls = accuracy(logits['clf'], y_cls)
            self.log(f'{part}_acc_clf', acc_cls, prog_bar=True)
            self.log(f'{part}_loss_clf', loss_clf)

        if self.use_relations:
            y_rel = batch['relations'] # [B, N_PATCH, N_PATCH]
            # y_rel = create_connectivity_matrix(logits['rel'].shape[-1])
            loss_rel = F.cross_entropy(logits['rel'], y_rel)
            loss += loss_rel
            acc_rel = accuracy(logits['rel'], y_rel)
            self.log(f'{part}_acc_rel', acc_rel, prog_bar=True)
            self.log(f'{part}_loss_rel', loss_rel)
        
        if self.args.use_angles:
            y_angle = batch['angles'].unsqueeze(1) # [B, 1, N_PATCH, N_PATCH]
            loss_angle = F.mse_loss(logits['angle'], y_angle)
            loss += loss_angle
            self.log(f'{part}_loss_angle', loss_angle)
            acc_angle = accuracy(logits['angle'], y_angle)
            self.log(f'{part}_acc_angle', acc_angle, prog_bar=True)

        # Log
        self.log(f'{part}_loss', loss)
        if part == 'train':
            self.log('lr', self.optimizer.param_groups[0]['lr'])

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Define a single validation step.
        """
        return self.training_step(batch, batch_idx, part='val')

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
    """
    Return the RelCNN model.
    """
    model_args = {
        'args': args,
        'num_classes': args.num_classes,
        'num_rel_classes': 9,
        'use_relations': args.use_relations,
        'use_angles':args.use_angles,
        'use_supervision': args.use_supervision,
    }
    model = RelCNN(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, strict=False, **model_args)
        print(f'Loaded model at "{args.model_checkpoint}"')
    return model
