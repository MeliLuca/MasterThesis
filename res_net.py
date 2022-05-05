import torch
from torchvision import transforms
import torch.optim as optim
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(34),
    transforms.ToTensor()])

batch_size = 64

dataset = torchvision.datasets.CIFAR10('./cifar_dataset',train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
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
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

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
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

cifar_model = ResNet(ResidualBlock, [2, 2, 2])

new_cifar_model = torch.torch.torch.nn.Sequential(*(list(cifar_model.children())[:-2]))
last_layers = torch.torch.torch.nn.Sequential(*(list(cifar_model.children())[-2:]))
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(new_cifar_model.parameters(), lr=0.001, momentum=0.9)

def attention_scores(q, k, v, mask=None):
    """
    Implementation of the cross-attention and the self-attention.
    Args:
        q: tensor of shape [B, T_Q, D_K]
        k: tensor of shape [B, T_V, D_K]
        v: tensor of shape [B, T_V, D_V]
    Output:
        out: tensor of shape [B, T_Q, D_v]
    """
    B = q.shape[0]
    #print(f'k shape is {k.shape}')
    scale = math.sqrt(k.shape[2])
    att = torch.bmm(q, k.transpose(1, 2)) / scale # [B, T_Q, T_V]
    if mask is not None:
        mask = mask.unsqueeze(0).repeat(B, 1, 1)
        att = torch.where(mask > 0.0, att, - math.inf * torch.ones_like(att))
    #att = F.softmax(att, 2)
    #out = torch.bmm(att, v)
    #return out
   # print(f'after attention scores is {att.shape}')
    return att


##################################################
# Attention layers
##################################################

class HeadScores(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    ::
             Linear proj.     Linear proj.     Linear proj.
               (query: q)       (key: k)        (value: v)
                  ↓                ↓                ↓
                   --------        |        --------
                           ↓       ↓       ↓
                          Attention (q, k, v)
    """
    def __init__(self, h_dim, head_out_dim):
        super().__init__()
        self.q_lin = nn.Linear(h_dim, head_out_dim, bias=False)
        self.k_lin = nn.Linear(h_dim, head_out_dim, bias=False)
        self.v_lin = nn.Linear(h_dim, head_out_dim, bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        q= torch.transpose(q, 1,2)
        if k is None:
            k = q
        if v is None:
            v = k
        #print(f'q shape is {q.shape}')
        q = self.q_lin(q)
        #print(f'q shape after linearization is {q.shape}')
        k = self.k_lin(k)
        v = self.v_lin(v)
        x = attention_scores(q, k, v, mask=mask) # [B, NUM_PATCH, NUM_PATCH]
        return x

class MultiHeadAttentionScores(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    ::
            [Head_1, Head_2, ..., Head_h]
                           ↓
                       Cat (dim=2)
                           ↓
            Linear (in=h * h_dim, out=h_dim)
    """
    def __init__(self, h_dim, num_heads):
        super().__init__()
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            HeadScores(h_dim, h_dim // num_heads) for _ in range(num_heads)
        ])
        #self.linear = nn.Linear((h_dim // num_heads) * num_heads, h_dim)

    def forward(self, input, k=None, v=None, mask=None):
        B, num_patches = input.shape[0:2]
        q= input.reshape(B, num_patches, -1)
        #q= torch.transpose(q, 1,2)
        x = [head(q, k, v, mask=mask) for head in self.heads]
        x = torch.stack(x, 1) # [B, num_heads, NUM_PATCH, NUM_PATCH]
        #x = self.linear(x) # [B, T, h_dim]
        return x
rel_head = MultiHeadAttentionScores(64, 9)
model = nn.Sequential(
    new_cifar_model,
    rel_head,
)

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

num_patches = 9
side = int(np.sqrt(num_patches))
patches_init = torch.arange(num_patches).reshape(side, side).numpy()
conn_matrix = create_connectivity_matrix(patches_init)
conn_matrix = torch.from_numpy(conn_matrix).to(torch.int64)

for epoch in range(100):  # loop over the dataset multiple times
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        outputs = torch.transpose(model(inputs), 1,3)


print('Finished Training')
