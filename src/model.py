import torch 
import torch.nn as nn 
import numpy as np 
import math

from torch.nn.modules.container import ModuleList
import torchvision
from torchvision import models 

import torch.nn.functional as F

def gen_conv(in_channels, hidden_dim, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_dim, 3, 1, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_dim, hidden_dim * 2, 3, 1, 2, bias=False),
        nn.BatchNorm2d(hidden_dim * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 1, 2, bias=False),
        nn.BatchNorm2d(hidden_dim * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_dim * 4, hidden_dim * 8, 3, 1, 2, bias=False),
        nn.BatchNorm2d(hidden_dim * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_dim * 8, hidden_dim * 16, 3, 1, 2, bias=False),
        nn.BatchNorm2d(hidden_dim * 16),
        nn.LeakyReLU(0.2, inplace=True),

        # down block
        nn.Conv2d(hidden_dim * 16, hidden_dim * 8, 3, 1, 2, bias=False),
        nn.BatchNorm2d(hidden_dim * 8),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hidden_dim * 8, out_channels, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True)
    )

class InitialLinear(nn.Module):
    def __init__(self, z_size) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, z):
        return self.seq(z) # outputs w

class Generator_Vector_Image(nn.Module):
    def __init__(self, z_size, hidden_dim=None, num_vectors=2, output_dim=1) -> None:
        super().__init__()

        self.num_vectors = num_vectors
        edge_size = math.sqrt(z_size)

        assert math.floor(edge_size) == math.ceil(edge_size), "Given z_size is not n^2."

        self.edge_size = int(edge_size)

        if hidden_dim is None:
            hidden_dim = self.edge_size

        self.linear = nn.Sequential(
            nn.Linear(num_vectors * z_size, z_size),
            nn.Tanh(),
            nn.Linear(z_size, z_size),
            nn.Tanh(),
        )

        self.conv = gen_conv(1, hidden_dim, output_dim)
    
    def forward(self, *vectors):

        concatinated = torch.cat(vectors, dim=1)

        hidden1 = self.linear(concatinated)
        hidden1 = hidden1.view(-1, 1, self.edge_size, self.edge_size)
        return self.conv(hidden1)


def block(in_channels, hidden_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
        nn.BatchNorm2d(hidden_channels),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(hidden_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )

class Generator_Vector_Vector(nn.Module):
    def __init__(self, z_size, num_vectors) -> None:
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(num_vectors * z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(z_size),
            nn.Linear(z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(z_size),
            nn.Linear(z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(z_size),
            nn.Linear(z_size, z_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, *vectors):
        hidden = torch.cat(vectors, dim=1)
        return self.seq(hidden)

class Generator_All_Image(nn.Module):
    def __init__(self, z_size, hidden_dim=None, num_vectors=2, num_images=1) -> None:
        super().__init__()

        self.num_vectors = num_vectors
        self.num_images = num_images

        self.edge_size = int(math.sqrt(z_size))

        if hidden_dim is None:
            hidden_dim = self.edge_size

        self.vec_to_image = Generator_Vector_Image(z_size, 
            hidden_dim=hidden_dim, 
            num_vectors=num_vectors, 
            output_dim=hidden_dim)

        self.conv = gen_conv(hidden_dim + num_images, hidden_dim, 1)
    
    def forward(self, vectors=[], images=[]):
        vec_images = self.vec_to_image(*vectors)
        con_images = torch.cat(images + [vec_images], dim=1)

        return self.conv(con_images)

class Discriminator(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.encode_img = nn.Sequential( 
            nn.Conv2d(in_c, 32, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True)) 

        self.rf_logits = nn.Sequential( nn.Conv2d(128, 1, kernel_size=4, stride=1), nn.Sigmoid())

    def forward(self, x):
        x = F.interpolate( x, [126,126], mode='bilinear', align_corners=True )
        x = self.encode_img(x)
        return  self.rf_logits(x)  

# model = StageOneGenerator(64)
# w = torch.zeros((4, 64))
# b = torch.zeros((4, 64))

# generated = model(w, b)

# print(generated.shape)
