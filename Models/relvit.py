##################################################
# Imports
##################################################

from pyexpat import model
from sched import scheduler
import pytorch_lightning as pl
from requests import patch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.models import ResNet50_Weights
from torch.optim import SGD, Adam, AdamW
import logging
import wandb
from matplotlib import cm
import numpy as np
from PIL import Image
import logging
from sklearn.decomposition import PCA
from math import sqrt

# Custom
#from models.vit import vit
#from models.vit.layers import MultiHeadAttention
from timm.models import vision_transformer as vit
import Models.vit_dino as vit_dino
from Layers.MultiHeadAttention import MultiHeadAttentionScores
from Layers import vit, decoder, PUP
from Models import segmenter
from utils import cross_entropy, accuracy, mIOU, resize_pos_embed, get_pos_embedding_sim, PolynomialLRDecay
from torchvision.utils import make_grid

cmap = cm.get_cmap('viridis')


##################################################
# RelViT Model
##################################################

class NaiveDecoder(nn.Module):
    def __init__(self, emb_dim, patch_size, img_size):
        super().__init__()
        self.patch_size = patch_size
        self.linear_channel = nn.Linear(emb_dim, 151)
        self.img_size = img_size
        self.upsample = nn.Upsample(scale_factor=self.patch_size, mode='bilinear')

    def forward(self, x):
        B,_,emb = x.shape
        segm_in = self.linear_channel(x).transpose(1,2)
        segm_in = segm_in.view([B, 151, self.img_size//self.patch_size, self.img_size//self.patch_size])
        segm_in = self.upsample(segm_in)
        return segm_in

class PUPDecoder(nn.Module):
    def __init__(self, emb_dim, patch_size):
        super().__init__()
        self.linear_channel = nn.Linear(emb_dim, 1024)
        self.decoder = PUP.Decoder2D(1024, 151, patch_size=patch_size)

    def forward(self, x):
        segm_in = self.linear_channel(x)
        B,num_patches,emb = segm_in.shape
        sqrt_patches = int(sqrt(num_patches))
        segm_in = segm_in.view([B,emb,sqrt_patches,sqrt_patches])
        segm_in = self.decoder(segm_in)
        return segm_in

# class SegmenterDecoder(nn.Module):
#     def __init__(self, args_model):
#         super().__init__()
#         self.decoder = Segmenter.MaskTransformer(**args_model)

#     def forward(self, x):
#         segm_in = self.decoder(x)
#         return segm_in

class RelViT(pl.LightningModule):
    """
    Lightning module that defines the Relational Vision Transformer Model.
    """
    def __init__(self, args, num_classes, num_rel_classes=9, use_relations=False, use_positional_emb=True, 
                 use_supervision=True, patch_size=32, dropout=0.1, freeze_fet_enc=False, use_distances=False, 
                 use_angles=False, use_abs_positions=False, proj_dropout=0.0, use_segmentation=True, pretrained=False):
        super().__init__()
        self.save_hyperparameters()
        self.args = args
        self.num_classes = num_classes
        self.use_supervision = use_supervision
        self.use_relations = use_relations
        self.use_distances = use_distances
        self.use_angles = use_angles
        self.use_abs_positions = use_abs_positions
        self.use_positional_emb = use_positional_emb
        self.patch_size = patch_size
        self.dropout = dropout
        self.proj_dropout = proj_dropout
        self.freeze_fet_enc = freeze_fet_enc
        self.use_segmentation = use_segmentation
        self.pretrained = pretrained
        self.num_patches = (self.args.img_height // self.args.patch_size) ** 2
        self.backbone = self._get_backbone()
        if self.use_segmentation :
            if self.args.model_version == 'naive':
                self.segm_head = NaiveDecoder(self.backbone.embed_dim, self.args.patch_size, self.args.img_height)
            # elif self.args.model_version == 'segmenter':
            #     args_dec = {
            #         'n_cls': self.num_classes+1,
            #         'patch_size': self.patch_size,
            #         'd_encoder': self.backbone.embed_dim,
            #         'd_model': self.backbone.embed_dim,
            #         'n_heads': self.backbone.embed_dim //64,
            #         'd_ff': 4 * self.backbone.embed_dim,
            #         'drop_path_rate': 0.0,
            #         'dropout': self.dropout,
            #         'img_size': (self.args.img_height, self.args.img_height)
            #         }
            #     self.segm_head = SegmenterDecoder(args_dec)
            elif self.args.model_version == 'pup':
                self.segm_head = PUPDecoder(self.backbone.embed_dim, self.patch_size)
            else :
                self.segm_head = nn.Identity()
        if self.use_relations:
            self.rel_head = MultiHeadAttentionScores(self.backbone.embed_dim, num_rel_classes)
        if self.use_supervision:
            self.clf_head = nn.Linear(self.backbone.embed_dim, args.num_classes)
        if self.use_distances:
            self.dist_head = MultiHeadAttentionScores(self.backbone.embed_dim, 1)
        if self.use_angles:
            self.angle_head = MultiHeadAttentionScores(self.backbone.embed_dim, 1)
        if self.use_abs_positions:
            num_patches = (self.args.img_height // self.args.patch_size) ** 2
            self.abs_pos_head = nn.Linear(self.backbone.embed_dim, num_patches)
            
        if self.freeze_fet_enc:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _get_backbone(self):
        if self.args.backbone in ['vit', 'simsiam']:
            backbone_args = {
                'img_size': self.args.img_height,
                'patch_size': self.args.patch_size,
                'attn_drop_rate': self.args.attn_drop,
                'use_positional_embeddings': self.use_positional_emb,
                'drop_path_rate': self.args.drop_path_rate,
                'proj_dropout': self.proj_dropout,
                'drop_rate': self.dropout,
            }
            backbone = vit_dino.__dict__[f'vit_{self.args.model_size}'](**backbone_args)
            if self.pretrained:
                trained_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
                num_extra_tokens = 1 +   trained_model.pos_embed.shape[1]
                out_dict = {}
                for k, v in trained_model.state_dict().items():
                    if k == "pos_embed" and v.shape != backbone.pos_embed.shape:
                        # To resize pos embedding when using model at different size from pretrained weights
                        v = resize_pos_embed(
                            v,
                            None,
                            (self.args.img_height // self.patch_size, self.args.img_height // self.patch_size),
                            num_extra_tokens,
                        )
                    out_dict[k] = v
                backbone.load_state_dict(out_dict, strict=True)
            
        elif self.args.backbone == 'fcn':
            if self.pretrained:
                # https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
                weights = ResNet50_Weights.DEFAULT
                backbone = fcn_resnet50(num_classes=self.num_classes+1, weights_backbone=weights)
            else:
                backbone = fcn_resnet50(num_classes=self.num_classes+1)
        elif self.args.backbone == 'segmenter':
            encoder_args = {
                'image_size': (self.args.img_height, self.args.img_height, 3),
                'patch_size': self.args.patch_size,
                'drop_path_rate': self.args.drop_path_rate,
                'dropout': self.proj_dropout,
                'd_model': 384,
                'n_layers' : 6,
                'n_heads': 12,
                'n_cls': 150,
                'd_ff':1536
            }
            args_dec = {
                    'n_cls': self.num_classes+1,
                    'patch_size': self.patch_size,
                    'd_encoder': 384,
                    'd_model': 384,
                    'n_heads': 384 //64,
                    'd_ff': 4 *384,
                    'drop_path_rate': 0.0,
                    'dropout': self.dropout,
                    'img_size': (self.args.img_height, self.args.img_height)
                    }
            encoder = vit.VisionTransformer(**encoder_args)
            dec = decoder.MaskTransformer(**args_dec)
            backbone = segmenter.Segmenter(encoder, dec, n_cls=150)
        return backbone

    def forward(self, x):
        """
        Forward function of the model.
        """
        out = {}
        # feats = self.backbone(x, mask=mask)
        feats = self.backbone(x)
        if self.args.backbone in ['vit']:
            if self.args.add_clf_to_patches:
                z_patches = feats['z_patches'] + feats['z'].unsqueeze(1)
            else:
                z_patches = feats['z_patches'] # [B x ? x emb_dim]
        elif self.args.backbone in ['fcn']:
            z_patches = feats['out']
        if self.use_relations:
            out['rel'] = self.rel_head(z_patches)
        if self.use_distances:
            out['dist'] = self.dist_head(z_patches)
        if self.use_angles:
            out['angle'] = self.angle_head(z_patches)
        if self.use_abs_positions:
            out['abs_pos'] = self.abs_pos_head(z_patches).transpose(1, 2)
        if self.use_supervision:
            out['clf'] = self.clf_head(feats['z'])
        if self.use_segmentation:
            out['seg'] = self.segm_head(z_patches) if self.args.backbone in ['vit'] else self.segm_head(feats)
        return out

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

        if self.use_segmentation:
            y_target = batch['labels'].long()
            torch.use_deterministic_algorithms(False)
            loss_seg = F.cross_entropy(logits['seg'], y_target, ignore_index=0)
            if self.args.num_gpus > 0:
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.use_deterministic_algorithms(True)
            loss += loss_seg
            miou = mIOU(y_target, logits['seg'], num_classes=self.num_classes)
            self.log(f'{part}_loss_seg', loss_seg)
            self.log(f'{part}_miou', miou)

        if self.use_relations:
            y_rel = batch['relations'] # [B, N_PATCH, N_PATCH]
            torch.use_deterministic_algorithms(False)
            loss_rel = cross_entropy(logits['rel'], y_rel, smooth=self.args.smooth_rel, 
                                     dropout=self.args.dropout_rel if part == 'train' else 0.0)
            if self.args.num_gpus > 0:
                torch.use_deterministic_algorithms(True, warn_only=True)
            else:
                torch.use_deterministic_algorithms(True)
            loss += loss_rel
            acc_rel = accuracy(logits['rel'], y_rel)
            self.log(f'{part}_acc_rel', acc_rel, prog_bar=True)
            self.log(f'{part}_loss_rel', loss_rel)

        if self.use_distances:
            y_dist = batch['distances'].unsqueeze(1) # [B, 1, N_PATCH, N_PATCH]
            loss_dist = F.mse_loss(logits['dist'], y_dist)
            loss += loss_dist
            self.log(f'{part}_loss_dist', loss_dist)

        if self.use_angles:
            y_angle = batch['angles'].unsqueeze(1) # [B, 1, N_PATCH, N_PATCH]
            loss_angle = F.mse_loss(logits['angle'], y_angle)
            loss += loss_angle
            self.log(f'{part}_loss_angle', loss_angle)

        if self.use_abs_positions:
            y_abs_pos = batch['abs_positions'] # [B, N_PATCH]
            torch.use_deterministic_algorithms(False)
            loss_abs_pos = F.cross_entropy(logits['abs_pos'], y_abs_pos)
            torch.use_deterministic_algorithms(True)
            loss += loss_abs_pos
            acc_abs_pos = accuracy(logits['abs_pos'], y_abs_pos)
            self.log(f'{part}_acc_abs_pos', acc_abs_pos, prog_bar=True)
            self.log(f'{part}_loss_abs_pos', loss_abs_pos)

        # Log
        self.log(f'{part}_loss', loss)
        if part == 'train':
            self.log('lr', self.optimizer.param_groups[0]['lr'])
        elif part == 'val':
            if self.use_positional_emb and batch_idx == 0 and self.args.backbone in ['vit', 'simsiam']:
                pos_sim_matrix = get_pos_embedding_sim(self.backbone.pos_embed.detach().cpu()) # [npatch, npatch, npatch, npatch]
                num_patches = pos_sim_matrix.shape[0]
                pos_sim_matrix = pos_sim_matrix.view(-1, num_patches, num_patches).unsqueeze(1) # [npatch * npatch, 1, npatch, npatch]
                pos_sim_matrix = (pos_sim_matrix + 1.0) / 2.0 # pos_sim_matrix in [0, 1]
                pos_sim_matrix = make_grid(pos_sim_matrix, nrow=num_patches, padding=1, pad_value=-1)[0] # [npatch + p, npatch + p]
                mask = torch.where(pos_sim_matrix >= 0, torch.ones_like(pos_sim_matrix), torch.zeros_like(pos_sim_matrix))
                mask = torch.stack([torch.ones_like(mask), torch.ones_like(mask), torch.ones_like(mask), mask], -1)
                pos_sim_matrix = torch.tensor(cmap(pos_sim_matrix)) # [npatch + p, npatch + p, 4]
                pos_sim_matrix = (pos_sim_matrix * mask).permute(2, 0, 1) # [4, npatch + p, npatch + p]
                if self.args.logger == 'wandb':
                    self.logger.experiment.log({'pos_embed_sim': [wandb.Image(pos_sim_matrix)]})

            if batch_idx == 0 and self.args.backbone in ['vit', 'simsiam']:
                kernel = list(self.backbone.patch_embed.parameters())[0].detach().cpu().numpy()
                D, C, PS, PS = kernel.shape
                pca = PCA(n_components=28)
                rgb_filt = pca.fit_transform(kernel.reshape(D, -1).T).T.reshape(-1, C, PS, PS)
                rgb_filt = (rgb_filt - rgb_filt.min()) / (rgb_filt.max() - rgb_filt.min())
                rgb_filt = torch.tensor(rgb_filt)
                rgb_filt = make_grid(rgb_filt, nrow=7, padding=1)
                if self.args.logger == 'wandb':
                    self.logger.experiment.log({'rgb_emb_filters': [wandb.Image(rgb_filt)]})

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Define a single validation step.
        """
        return self.training_step(batch, batch_idx, part='val')

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
        if self.args.decay_lr_sched == 'polynomial':
            self.scheduler = PolynomialLRDecay(optimizer, self.args.epochs)
            self.optimizer = optimizer
            return {"optimizer": optimizer, "lr_scheduler": self.scheduler}
        self.optimizer = optimizer
        return optimizer

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
    Return the RelViT model.
    """
    model_args = {
        'args': args,
        'num_classes': args.num_classes,
        'num_rel_classes': 9,
        'use_relations': args.use_relations,
        'use_positional_emb': args.use_positional_embeddings,
        'use_supervision': args.use_supervision,
        'use_distances': args.use_distances,
        'use_angles': args.use_angles,
        'patch_size': args.patch_size,
        'dropout': args.dropout,
        'freeze_fet_enc': args.freeze_fet_enc,
        'use_abs_positions': args.use_abs_positions,
        'proj_dropout': args.proj_dropout,
        'use_segmentation': args.use_segmentation,
        'pretrained': args.pretrained,
    }
    model = RelViT(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, strict=False, **model_args)
        logging.info(f'Loaded model at "{args.model_checkpoint}"')
    return model
