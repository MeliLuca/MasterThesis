##################################################
# Imports
##################################################

from ast import arg
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
import torchvision.models as models
import logging
import wandb
from torchvision.utils import make_grid
from sklearn.decomposition import PCA


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
                 use_supervision=True, use_angles=False, use_distances=False, use_abs_positions=False, **bkwargs):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        if self.args.backbone == 'cifar10_model':
            self.backbone = resnet_cifar10.__dict__['resnet56'](*bargs, **bkwargs)
        else : 
            self.backbone = models.__dict__[args.backbone](*bargs, **bkwargs)
        self.use_relations = use_relations
        self.use_angles = use_angles
        self.use_supervision = use_supervision
        self.num_rel_classes = num_rel_classes
        self.use_distances = use_distances
        self.use_abs_positions = use_abs_positions

        ## list(self.backbone.children())[-1].in_features == self.backbone.embed_dim
        if self.use_relations:
            #self.rel_head = nn.Conv2d(self.backbone.layer1[0].expansion * 512, 
            #                          self.num_rel_classes, kernel_size=1)
            self.rel_head = MultiHeadAttentionScores(list(self.backbone.children())[-1].in_features, num_rel_classes)
        if self.use_angles:
            self.angle_head = MultiHeadAttentionScores(list(self.backbone.children())[-1].in_features, 1)
        if self.use_distances:
            self.dist_head = MultiHeadAttentionScores(list(self.backbone.children())[-1].in_features, 1)
        if self.use_abs_positions:
            num_patches = (self.args.img_height // self.args.patch_size) ** 2
            self.abs_pos_head = nn.Linear(list(self.backbone.children())[-1].in_features, num_patches)

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

    # def self.forward_feat_with_rel(x):
    def forward(self, x):
        x = self.forward_feat(x)
        B, C, H, W = x.shape
        out={}
        x_att = torch.flatten(x, 2).transpose(1, 2)
        if self.use_relations:
            out['rel'] = self.rel_head(x_att)
        if self.use_angles:
            out['angle'] = self.angle_head(x_att)
        if self.use_distances:
            out['dist'] = self.dist_head(x_att)
        if self.use_abs_positions:
            out['abs_pos'] = self.abs_pos_head(x_att)
        if self.use_supervision: 
            if self.args.backbone == 'cifar10_model':
                x_clf= self.backbone.avg_flat(x)
            else:
                x_clf = self.backbone.avgpool(x)
                x_clf = torch.flatten(x_clf, 1)
            out['clf'] = self.backbone.fc(x_clf)
        return out

    def configure_optimizers(self):
        """                                                                                                             
        Configure optimizer for the learning process.                                                                   
        """                                                                                                             
        if self.args.optimizer == 'adam':                                                                               
            optimizer = Adam(self.parameters(), self.args.lr)                                                           
        elif self.args.optimizer == 'sgd':                                                                              
            optimizer = SGD(self.parameters(), self.args.lr, momentum=0.9, weight_decay=5e-5)
        elif self.args.optimizer == 'adamw':
            optimizer = AdamW(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)                           
        else:                                                                                                           
            raise Exception(f'Error. Optimizer "{self.args.optimizer}" not supported.')                                 
        self.optimizer = optimizer 
        if self.args.backbone == 'cifar10_model':
            scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=0.1, total_iters=40)
            return ([optimizer], [scheduler])
                                                                                     
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
            torch.use_deterministic_algorithms(False)
            loss_rel = F.cross_entropy(logits['rel'], y_rel)
            torch.use_deterministic_algorithms(True)
            loss += loss_rel
            acc_rel = accuracy(logits['rel'], y_rel)
            self.log(f'{part}_acc_rel', acc_rel, prog_bar=True)
            self.log(f'{part}_loss_rel', loss_rel)
        
        if self.use_angles:
            y_angle = batch['angles'].unsqueeze(1) # [B, 1, N_PATCH, N_PATCH]
            loss_angle = F.mse_loss(logits['angle'], y_angle)
            loss += loss_angle
            self.log(f'{part}_loss_angle', loss_angle)
            acc_angle = accuracy(logits['angle'], y_angle)
            self.log(f'{part}_acc_angle', acc_angle, prog_bar=True)

        if self.use_distances:
            y_dist = batch['distances'].unsqueeze(1) # [B, 1, N_PATCH, N_PATCH]
            loss_dist = F.mse_loss(logits['dist'], y_dist)
            loss += loss_dist
            self.log(f'{part}_loss_dist', loss_dist)
        
        if self.use_abs_positions:
            y_abs_pos = batch['abs_positions'] # [B, N_PATCH]
            torch.use_deterministic_algorithms(False)
            loss_abs_pos = F.cross_entropy(logits['abs_pos'], y_abs_pos)
            torch.use_deterministic_algorithms(True)
            loss += loss_abs_pos
            acc_abs_pos = accuracy(logits['abs_pos'], y_abs_pos)
            self.log(f'{part}_acc_abs_pos', acc_abs_pos, prog_bar=True)
            self.log(f'{part}_loss_abs_pos', loss_abs_pos)
        
        # if batch_idx == 0 and self.args.backbone in ['vit', 'simsiam']:
        #     kernel = list(self.backbone.patch_embed.parameters())[0].detach().cpu().numpy()
        #     D, C, PS, PS = kernel.shape
        #     pca = PCA(n_components=28)
        #     rgb_filt = pca.fit_transform(kernel.reshape(D, -1).T).T.reshape(-1, C, PS, PS)
        #     rgb_filt = (rgb_filt - rgb_filt.min()) / (rgb_filt.max() - rgb_filt.min())
        #     rgb_filt = torch.tensor(rgb_filt)
        #     rgb_filt = make_grid(rgb_filt, nrow=7, padding=1)
        #     if self.args.logger == 'wandb':
        #         self.logger.experiment.log({'rgb_emb_filters': [wandb.Image(rgb_filt)]})


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
        'use_distances':args.use_distances,
        'use_abs_positions':args.use_abs_positions,
        'use_supervision': args.use_supervision,
    }
    model = RelCNN(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, strict=False, **model_args)
        print(f'Loaded model at "{args.model_checkpoint}"')
    return model
