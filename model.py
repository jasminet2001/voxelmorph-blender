import torch
import torch.nn as nn


class VoxelMorphNet(nn.Module):

    def __init__(self, in_channels=2, out_channels=3, features=[16, 32, 64, 128]):
        super(VoxelMorphNet, self).__init__()

        self.encoder1 = self.conv_block(2, features[0])
        self.encoder2 = self.conv_block(features[0], features[1])
        self.encoder3 = self.conv_block(features[1], features[2])
        self.encoder4 = self.conv_block(features[2], features[3])

        self.dec1 = self.deconv_block(features[3], features[2])
        self.dec2 = self.deconv_block(features[2]*2, features[1])
        self.dec3 = self.deconv_block(features[1]*2, features[0])
        self.dec4 = self.deconv_block(features[0]*2, features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels,
                               kernel_size=2, stride=2),
            nn.ReLU()
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec1 = self.dec1(enc4)
        dec2 = self.dec2(torch.cat((dec1, enc3), dim=1))
        dec3 = self.dec3(torch.cat((dec2, enc2), dim=1))
        dec4 = self.dec4(torch.cat((dec3, enc1), dim=1))

        return self.final_conv(dec4)
