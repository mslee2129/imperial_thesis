from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# discriminator 1
# shape of input is spectro, output is mel
class DiscriminatorX(nn.Module):
    '''
    Dx for cGAN
    70x70 PatchGAN
    mel+spect as input, patch prediction as output
    '''
    def __init__(self, input_channels):
        super(DiscriminatorX, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    def save(self, filepath):
        pass

# add discriminator2
# change to fit shape of input mel and output spectro
class DiscriminatorY(nn.Module):
    '''
    Dy fo CycleGAN
    70x70 PatchGAN
    spect as input, patch prediction as output
    '''
    def __init__(self, input_channels):
        super(DiscriminatorY, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# generator G
class GeneratorG(nn.Module):
    '''
    Maps from X->Y (spectrogram to mel)
    Conditional GAN, UNet architecture with skip connections
    Encoder and decoder 
    Adversarial Loss
    Spectrogram as input, mel as output
    '''
    def __init__(self):
        super(GeneratorG, self).__init__()

    def resnet_block():
        



# generator F
class GeneratorF(nn.Module):
    '''
    Maps from Y->X
    CycleGAN, ResNet architecture
    Cycle Consistency Loss between Y and G(F(y))
    '''
