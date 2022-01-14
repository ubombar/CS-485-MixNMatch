from typing import Any, ForwardRef
import torch 
import torch.nn as nn 
import numpy as np 
import math

from torch.nn.modules.activation import ReLU

import model

from torch.nn.modules.container import ModuleList
import torchvision

import torch.nn.functional as F

import pytorch_lightning as pl 

class TrainableModel(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        Z_SIZE = 64
        HIDDEN_SIZE = 64

        # Generators in stage
        self.G_b = model.Generator_Vector_Image(Z_SIZE, hidden_dim=HIDDEN_SIZE, num_vectors=2, output_dim=1)
        self.G_pm = model.Generator_All_Image(Z_SIZE, hidden_dim=HIDDEN_SIZE, num_vectors=1, num_images=1)
        self.G_cm = model.Generator_All_Image(Z_SIZE, hidden_dim=HIDDEN_SIZE, num_vectors=1, num_images=2)

        # Generators outside stage
        self.G_Ba = model.Generator_Vector_Vector(Z_SIZE, num_vectors=2)
        self.G_Ca = model.Generator_Vector_Vector(Z_SIZE, num_vectors=2)

        # Initial layer
        self.I = model.InitialLinear(Z_SIZE)

        # Discriminators
        self.D_b = model.Discriminator(1)
        self.D_p = model.Discriminator(1)
        self.D_c = model.Discriminator(1)

        # Hidden units
        self.B: torch.Tensor = None
        self.P: torch.Tensor = None
        self.C: torch.Tensor = None
    
    def forward(self, z, b, p, c):
        w = self.I(z)
        F_p = self.G_ba(p, w)
        F_c = self.G_Ca(c, F_p)

        B = self.G_b(b, w)
        P = self.G_pm([F_p], [B])
        C = self.G_cm([F_c], [B, P])

        return C

    def training_step(self, batch, batch_idx, optimizer_idx):
        z, b, p, c, background, segmap, original = batch

        if optimizer_idx == 0:
            w = self.I(z)
            F_p = self.G_ba(p, w)
            F_c = self.G_Ca(c, F_p)

            B = self.G_b(b, w)
            P = self.G_pm([F_p], [B])
            C = self.G_cm([F_c], [B, P])

            self.B = b 
            self.P = P 
            self.C = C

            return 0 # Do not update the weights
        else:
            B_loss = torch.sum(torch.log(self.D_b(background))) + torch.sum(torch.log(1 - self.D_b(self.B)))
            P_loss = torch.sum(torch.log(self.D_p(segmap))) + torch.sum(torch.log(1 - self.D_p(self.P)))
            C_loss = torch.sum(torch.log(self.D_c(original))) + torch.sum(torch.log(1 - self.D_c(self.C)))

            L_adverserial = (B_loss + P_loss + C_loss) / 3

            return L_adverserial


