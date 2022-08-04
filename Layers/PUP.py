import torch
import torch.nn as nn

""" NON official implementation from https://github.com/920232796/SETR-pytorch/blob/master/SETR/transformer_seg.py 
    of Progressive UPsampling of SERT paper https://fudan-zvg.github.io/SETR/
"""
class Decoder2D(nn.Module):
    def __init__(self, in_channels, out_channels,  patch_size, features=[1024, 512, 256 ,256, 256]):#features=[512, 512, 256, 256 ,256]):
        super().__init__()
        self.patch_size = patch_size
        self.decoder_1 = nn.Sequential(
                    nn.Conv2d(in_channels, features[0], 3, padding=1),
                    nn.BatchNorm2d(features[0]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_2 = nn.Sequential(
                    nn.Conv2d(features[0], features[1], 3, padding=1),
                    nn.BatchNorm2d(features[1]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        if patch_size == 32:
            self.decoder_5 = nn.Sequential(
                nn.Conv2d(features[3], features[4], 3, padding=1),
                nn.BatchNorm2d(features[4]),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            )

        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        if self.patch_size == 32:
            x = self.decoder_5(x)           
        x = self.final_out(x)
        return x