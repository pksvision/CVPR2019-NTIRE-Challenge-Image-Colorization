# Image Colorization using Conditional GAN with Representation Loss
# Code for the training of Representation Nework 
# =================================================================
# Authors:
# ========
# Prasen K Sharma
# Priyankar Jain
# Dr. Arijit Sur
# Department of Computer Science and Engineering
# Indian Institute of Technology Guwahati, India
# Contact : kumar176101005@iitg.ac.in

import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from representation import Color_Generator

parser = argparse.ArgumentParser()
parser.add_argument('--color_dir', default='./color/')
parser.add_argument('--checkpoints_dir', default='./Models/Main/')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_images', type=int, default=790)
parser.add_argument('--learning_rate_g', type=float, default=0.00000000018995)
parser.add_argument('--learning_rate_d', type=float, default=0.00000000015)
parser.add_argument('--end_epoch', type=int, default=4000)
parser.add_argument('--img_extension', default='.png')
parser.add_argument('--batch_mse_loss', type=float, default=0.0)
parser.add_argument('--total_mse_loss', type=float, default=0.0)
parser.add_argument('--batch_feature_loss', type=float, default=0.0)
parser.add_argument('--total_feature_loss', type=float, default=0.0)
parser.add_argument('--batch_adv_loss', type=float, default=0.0)
parser.add_argument('--total_adv_loss', type=float, default=0.0)
parser.add_argument('--batch_G_loss', type=float, default=0.0)
parser.add_argument('--total_G_loss', type=float, default=0.0)
parser.add_argument('--batch_D_loss', type=float, default=0.0)
parser.add_argument('--total_D_loss', type=float, default=0.0)
parser.add_argument('--batch_R_loss', type=float, default=0.0)
parser.add_argument('--total_R_loss', type=float, default=0.0)
parser.add_argument('--lambda_mse', type=float, default=1.0)
parser.add_argument('--lambda_adv', type=float, default=0.0063935)
parser.add_argument('--lambda_r', type=float, default=0.0000145)
parser.add_argument('--image_size', type=int, default=512)

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ToTensor(object):

    def __call__(self, sample):
        Y_channel, chroma_channels = sample['Y'], sample['CrCb']
        Y_channel = Y_channel.astype(np.float32)
        Y_channel = torch.from_numpy(Y_channel)

        Y_channel = torch.transpose(torch.transpose(Y_channel, 2, 0), 1, 2)
        Y_channel = Y_channel / 255.0

        chroma_channels = chroma_channels.astype(np.float32)
        chroma_channels = torch.from_numpy(chroma_channels)
        chroma_channels = torch.transpose(torch.transpose(chroma_channels, 2, 0), 1, 2)
        chroma_channels = chroma_channels / 255.0

        return {'Y': Y_channel,
                'CrCb': chroma_channels}


class Color_Generator_Rep(nn.Module):

    def __init__(self):
        super(Color_Generator_Rep, self).__init__()

        # layers for encoder network

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.prelu1 = nn.PReLU(num_parameters=64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.prelu2 = nn.PReLU(num_parameters=64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=64)
        self.prelu3 = nn.PReLU(num_parameters=64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)
        self.prelu4 = nn.PReLU(num_parameters=64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.prelu5 = nn.PReLU(num_parameters=32)

        self.conv6 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=2)
        self.prelu6 = nn.PReLU(num_parameters=2)

        # layers for decoder network
        self.d_conv1 = nn.ConvTranspose2d(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.d_bn1 = nn.BatchNorm2d(num_features=32)
        self.d_relu1 = nn.PReLU(num_parameters=32)

        self.d_conv2 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.d_bn2 = nn.BatchNorm2d(num_features=64)
        self.d_relu2 = nn.PReLU(num_parameters=64)

        self.d_conv3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.d_bn3 = nn.BatchNorm2d(num_features=64)
        self.d_relu3 = nn.PReLU(num_parameters=64)

        self.d_conv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.d_bn4 = nn.BatchNorm2d(num_features=64)
        self.d_relu4 = nn.PReLU(num_parameters=64)

        self.d_conv5 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.d_bn5 = nn.BatchNorm2d(num_features=64)
        self.d_relu5 = nn.PReLU(num_parameters=64)

        self.d_conv6 = nn.ConvTranspose2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.d_bn6 = nn.BatchNorm2d(num_features=2)
        self.d_relu6 = nn.ReLU()

    def forward(self, input):

        # encoder
        en_layer1 = self.prelu1(self.bn1(self.conv1(input)))
        en_layer2 = self.prelu2(self.bn2(self.conv2(en_layer1)))
        en_layer3 = self.prelu3(self.bn3(self.conv3(en_layer2)))
        en_layer4 = self.prelu4(self.bn4(self.conv4(en_layer3)))
        en_layer5 = self.prelu5(self.bn5(self.conv5(en_layer4)))
        en_layer6 = self.prelu6(self.bn6(self.conv6(en_layer5)))

        # decoder
        de_layer1 = self.d_relu1(self.d_bn1(self.d_conv1(en_layer6)))
        de_layer2 = self.d_relu2(self.d_bn2(self.d_conv2(de_layer1)))

        # skip connection
        de_layer2 = de_layer2 + en_layer2
        de_layer3 = self.d_relu3(self.d_bn3(self.d_conv3(de_layer2)))
        de_layer4 = self.d_relu4(self.d_bn4(self.d_conv4(de_layer3)))

        # skip connection
        de_layer4 = de_layer4 + en_layer4
        de_layer5 = self.d_relu5(self.d_bn5(self.d_conv5(de_layer4)))
        de_layer6 = self.d_relu6(self.d_bn6(self.d_conv6(de_layer5)))

        return de_layer6

# Discriminator network
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.prelu1 = nn.PReLU(num_parameters=8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.prelu2 = nn.PReLU(num_parameters=16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)
        self.prelu3 = nn.PReLU(num_parameters=32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=2)
        self.prelu4 = nn.PReLU(num_parameters=2)

    def forward(self, input):

        layer1 = self.prelu1(self.bn1(self.conv1(input)))
        layer2 = self.prelu2(self.bn2(self.conv2(layer1)))
        layer3 = self.prelu3(self.bn3(self.conv3(layer2)))
        layer4 = self.prelu4(self.bn4(self.conv4(layer3)))

        final_logits = torch.mean(torch.sigmoid(layer4.view(opt.batch_size, -1)), 1)
        
        return final_logits
