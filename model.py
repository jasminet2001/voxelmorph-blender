import torch
import torch.nn as nn


class VoxelMorphNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=3, features=[16, 32, 64, 128]):
        super(VoxelMorphNet, self).__init__()

        self.encoder1 = self.conv_block(2, features[0])
        self.encoder2 = self.conv_block(features[0], features[1])
        self.encoder3 = self.conv_block(features[1], features[2])
        self.encoder4 = self.conv_block(features[2], features[3])

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU()
        )
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
    )
