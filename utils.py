##################################################
# Imports
##################################################

import pytorch_lightning as pl
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import tarfile
import requests
import logging
import string
import random
import datetime
from tqdm import tqdm


def cross_entropy(logits, labels, smooth=0.0, dropout=0.0):
    """
    Compute the cross entropy.
    Args:
        logits: tensor of shape [bs, num_classes(, ...)].
        labels: tensor of shape [bs(, num_classes, ...)].
        smooth: scalar, for label smoothing.
    Output:
        xent: scalar.
    """
    if smooth == 0.0:
        return F.cross_entropy(logits, labels)

    logprobs = F.log_softmax(logits, dim=1)
    nll_loss = - logprobs.gather(dim=1, index=labels.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = - logprobs.mean(dim=1)
    loss = (1.0 - smooth) * nll_loss + smooth * smooth_loss 
    if dropout > 0.0:
        loss = F.dropout(loss, p=dropout)
    return loss.mean()

def accuracy(preds, labels, preds_with_logits=True):
    """
    Compute the accuracy.

    Args:
        preds: tensor of shape [bs, num_classes(, ...)].
        labels: tensor of shape [bs(, ...)].
        preds_with_logits: bool.

    Output:
        acc: scalar.
    """
    if preds_with_logits:
        preds = F.softmax(preds, 1)
    preds = preds.argmax(1)
    acc = (1.0 * (preds == labels)).mean()
    return acc

def cos_anneal(e0, e1, t0, t1, e):
    """ ramp from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    alpha = 1.0 - math.cos(alpha * math.pi / 2) # warp through cosine
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

def linear(e0, e1, t0, t1, e):
    """ linear from (e0, t0) -> (e1, t1) through a cosine schedule based on e \in [e0, e1] """
    alpha = max(0, min(1, (e - e0) / (e1 - e0))) # what fraction of the way through are we
    t = alpha * t1 + (1 - alpha) * t0 # interpolate accordingly
    return t

def cos_anneal_warmup(e0, e1, t0, t1, e_w, e):
    if e >= e_w:
        t = cos_anneal(e0, e1, t0, t1, e)
    else:
        t = linear(e0, e_w, 0, t0, e)
    return t

class DecayLR(pl.Callback):
    def __init__(self, lr_init=3e-4, lr_end=1.25e-6, log_lr=False):
        super(DecayLR, self).__init__()
        print('use cosine decay')
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.log_lr = log_lr

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # The step size is annealed from 1e10−4 to 1.25e10−6 over 1,200,000 updates. I use 3e-4
        #t = cos_anneal(0, trainer.max_steps, self.lr_init, self.lr_end, trainer.global_step)
        t = cos_anneal_warmup(0, trainer.max_steps, self.lr_init, self.lr_end, trainer.max_steps // 10, trainer.global_step)
        for g in pl_module.optimizer.param_groups:
            g['lr'] = t

        if self.log_lr:
            pl_module.log('lr', t)

def get_callbacks(args):
    """
    Callbacks for the PyTorch Lightning Trainer.
    """
    # Model checkpoint
    dir_path = f'stats/{args.dataset}/{args.backbone}/checkpoints' if args.stats_path == '' else f'stats/{args.stats_path}/checkpoints'
    model_checkpoint_clbk = pl.callbacks.model_checkpoint.ModelCheckpoint(
        dirpath=os.path.join(os.getcwd(), dir_path),
        filename='best',
        monitor=args.metric_monitor,
        save_last=True,
        mode=args.metric_maxmin,
    )
    model_checkpoint_clbk.CHECKPOINT_NAME_LAST = '{epoch}-{step}'
    if args.decay_lr_sched == 'cosine' :
        callbacks = [
            model_checkpoint_clbk,
            DecayLR(lr_init=args.lr),
        ]
    else :
        callbacks = [
            model_checkpoint_clbk,
        ]
    return callbacks

def get_logger(args):
    """
    Logger for the PyTorchLightning Trainer.
    """
    logger_kind = 'csv' if 'logger' not in args.__dict__ else args.logger
    path = f'stats/{args.dataset}/{args.backbone}/logs' if args.stats_path == '' else f'stats/{args.stats_path}/logs'
    save_dir = os.path.join(os.getcwd(), path)
    if logger_kind == 'csv':
        logger = pl.loggers.CSVLogger(
            save_dir=save_dir,
        )
    elif logger_kind == 'tensorboard':
        logger = pl.loggers.tensorboard.TensorBoardLogger(
            save_dir=save_dir,
        )
    elif logger_kind == 'wandb':
        task_str = []
        if args.use_supervision:
            task_str += ['clf']
        if args.use_relations:
            task_str += ['rel']
        if args.use_distances:
            task_str += ['dist']
        if args.use_angles:
            task_str += ['angle']
        if args.use_abs_positions:
            task_str += ['abs_pos']
        if args.use_segmentation:
            task_str += ['seg']
        if args.use_positional_embeddings:
            task_str += ['pos_emb']
        name = [
            args.exp_id, args.dataset, '-'.join(task_str), 
            '-'.join([str(args.backbone), str(args.model_version) ,str(args.model_size), str(args.patch_size), str(args.img_height)])
        ]
        logger = pl.loggers.WandbLogger(
            save_dir= save_dir,
            name='/'.join(name),
            project='master_thesis',
        )
    else:
        raise Exception(f'Error. Logger "{logger_kind}" is not supported.')
    return logger

def get_pos_embedding_sim(pos_embed):
    """
    inputs:
        pos_embed: [1, num_embeddings, embed_dim]
    outputs:
        sim_matrix: [num_patches, num_patches, num_patches, num_patches]
    """
    pos_embed = pos_embed.detach()[0, 1:] # don't consider the clf token
    num_embed = pos_embed.shape[0]
    num_patches = int(np.sqrt(num_embed))
    sim_matrix = np.zeros((num_patches, num_patches, num_patches, num_patches))

    for i in range(num_patches):
        for j in range(num_patches):
            ij_flat = i * num_patches + j
            p_ij = pos_embed[ij_flat]
            for k in range(num_patches):
                for l in range(num_patches):
                    kl_flat = k * num_patches + l
                    p_kl = pos_embed[kl_flat]
                    sim_matrix[i, j, k, l] = F.cosine_similarity(p_ij, p_kl, dim=0)
    return torch.tensor(sim_matrix)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def get_split_idxs(ds, train_size=0.8):
    """
    Split a torch dataset into train and validation and return the train and validation idxs.
    """
    from sklearn import model_selection
    from tqdm import tqdm
    SEED = 35571
    cls = []
    for _, y in tqdm(ds):
        cls += [y]
    idx_train, idx_val = model_selection.train_test_split(np.arange(len(ds)), 
                                                          random_state=SEED, 
                                                          shuffle=True, train_size=train_size, 
                                                          stratify=cls)
    return sorted(idx_train.tolist()), sorted(idx_val.tolist())

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_file(url, fname):
    folder = os.path.dirname(fname)
    if not os.path.exists(folder) and (len(folder) > 0):
        os.makedirs(folder)
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def extract_tar(tarpath, dest=None):
    """
    Extract a tar file (".tar", ".tar.gz") in dest. If dest is None the tar
    will be extracted in the same folder of the tarpath.
    """
    if dest is None:
        folder = os.path.dirname(tarpath)
        dest = folder if len(folder) > 0 else './'
    tar = tarfile.open(tarpath)
    tar.extractall(dest)
    tar.close()
    logging.info(f'Extracted tar to {folder}')

def get_dataset_constant(ds):
    """
    Given a pytorch dataset of images, compute the normalization tensors (mean and std) used for transforming 
    the input images.
    """
    mu, energy = torch.zeros((3,)), torch.zeros((3,))
    num_samples = len(ds)
    for x, _ in tqdm(ds):
        mu += x.mean(-1).mean(-1) / num_samples
        energy += x.pow(2).mean(-1).mean(-1) / num_samples
    return mu, torch.sqrt(energy - mu.pow(2))

def mIOU(label, pred, num_classes=19):
    pred = F.softmax(pred, dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)

def unpadding(y, target_size):
    H, W = target_size
    H_pad, W_pad = y.size(2), y.size(3)
    # crop predictions on extra pixels coming from padding
    extra_h = H_pad - H
    extra_w = W_pad - W
    if extra_h > 0:
        y = y[:, :, :-extra_h]
    if extra_w > 0:
        y = y[:, :, :, :-extra_w]
    return y

def padding(im, patch_size, fill_value=0):
    # make the image sizes divisible by patch_size
    H, W = im[0], im[1]
    pad_h, pad_w = 0, 0
    if H % patch_size > 0:
        pad_h = patch_size - (H % patch_size)
    if W % patch_size > 0:
        pad_w = patch_size - (W % patch_size)
    im_padded = im
    if pad_h > 0 or pad_w > 0:
        im_padded = F.pad(im, (0, pad_w, 0, pad_h), value=fill_value)
    return im_padded


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, torch.nn.Linear) and m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.bias, 0)
        torch.nn.init.constant_(m.weight, 1.0)


def resize_pos_embed(posemb, grid_old_shape, grid_new_shape, num_extra_tokens):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # posemb_tok, posemb_grid = (
    #     posemb[:, :num_extra_tokens],
    #     posemb[0, num_extra_tokens:],
    # )
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    if grid_old_shape is None:
        gs_old_h = int(math.sqrt(len(posemb_grid)))
        gs_old_w = gs_old_h
    else:
        gs_old_h, gs_old_w = grid_old_shape

    gs_h, gs_w = grid_new_shape
    posemb_grid = posemb_grid.reshape(1, gs_old_h, gs_old_w, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb