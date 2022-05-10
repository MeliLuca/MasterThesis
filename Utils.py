import torch
import math
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl

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

## per creazione del target 
## patches sarà il vettore contenente tutte le 64 posizioni delle patches.(TR, TC, TL ecc..) alto destra, alto sinistra, alto in centro ...
def create_patch_matrix(num_patches):
    side = int(math.sqrt(num_patches))
    patches = np.arange(num_patches).reshape(side, side)
    return patches

def create_label_matrix(num_patches):
    """
    Create the spatial location matrix (as the example above).
    """
    labels = np.arange(9) # Spatial location encoded as int...
    labels_matrix = labels.reshape(3, 3)
    patch_matrix_side = int(math.sqrt(num_patches))

    side = 2 * patch_matrix_side - 1
    lu, cu, ru = labels_matrix[0, 0], labels_matrix[0, 1], labels_matrix[0, 2]
    lc, cc, rc = labels_matrix[1, 0], labels_matrix[1, 1], labels_matrix[1, 2]
    ld, cd, rd = labels_matrix[2, 0], labels_matrix[2, 1], labels_matrix[2, 2]

    corner_side = int((side - 1) / 2)
    cor = np.ones([corner_side, corner_side]).astype(labels.dtype)
    cen = np.ones([1, 1]).astype(labels.dtype)
    ver = np.ones([corner_side, 1]).astype(labels.dtype)
    hor = np.ones([1, corner_side]).astype(labels.dtype)

    labels_matrix = np.concatenate([
        np.concatenate([lu * cor, cu * ver, ru * cor], 1),
        np.concatenate([lc * hor, cc * cen, rc * hor], 1),
        np.concatenate([ld * cor, cd * ver, rd * cor], 1),
    ], 0)

    return labels_matrix

def create_connectivity_matrix(patches):
    patch_flat = patches.reshape(-1)
    num_patches = len(patch_flat)
    labels_matrix = create_label_matrix(num_patches)

    side = int(math.sqrt(num_patches))
    conn_matrix = - 1 * np.ones([num_patches, num_patches]).astype(labels_matrix.dtype)
    center = int((labels_matrix.shape[0] - 1) / 2)
    for i in range(side):
        for j in range(side):
            labels_ij_matrix = labels_matrix[center - i: center - i  + side, center - j: center - j  + side]
            labels_ij_matrix = labels_ij_matrix.reshape(-1)
            #idx_flat = patch_flat[i * side + j]
            idx_flat = i * side + j
            conn_matrix[idx_flat, :] = labels_ij_matrix

    conn_matrix = conn_matrix.T
    conn_matrix = conn_matrix[patch_flat, :][:, patch_flat]
    return conn_matrix

def get_rel_matrix (args):
    num_patches = args.classes
    side = int(np.sqrt(num_patches))
    patches_init = torch.arange(num_patches).reshape(side, side).numpy()
    conn_matrix = create_connectivity_matrix(patches_init)
    conn_matrix = torch.from_numpy(conn_matrix).to(torch.int64)
    target = conn_matrix.unsqueeze(0).repeat(64, 1, 1)
    return target

def get_callbacks(args):
    """
    Callbacks for the PyTorch Lightning Trainer.
    """
    # Model checkpoint
    model_checkpoint_clbk = pl.callbacks.model_checkpoint.ModelCheckpoint(
        monitor=args.metric_monitor,
        save_last=True,
        mode=args.metric_maxmin,
    )
    model_checkpoint_clbk.CHECKPOINT_NAME_LAST = '{epoch}-{step}'
    callbacks = [
        model_checkpoint_clbk,
    ]
    return callbacks
