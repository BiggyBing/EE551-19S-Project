#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 23:02:38 2019

@author: bingyangwen
"""

import torch.nn as nn
from options import GanConfiguration



class Discriminator(nn.Module):
   
    def __init__(self, config: GanConfiguration):

        super(Discriminator, self).__init__()
        
        self.DIM = config.DIM
        self.main = nn.Sequential(
            nn.Conv2d(3, self.DIM, 3, 2, padding=1),
            nn.BatchNorm2d(self.DIM),
            nn.LeakyReLU(),
            nn.Conv2d(self.DIM, 2 * self.DIM, 3, 2, padding=1),
            nn.BatchNorm2d(2 * self.DIM),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 3, 2, padding=1),
            nn.BatchNorm2d(4 * self.DIM),
            nn.LeakyReLU(),
        )

        self.linear = nn.Linear(4*4*4*self.DIM, 1)

    def forward(self, image):
        output = self.main(image)
        output = output.view(-1, 4*4*4*self.DIM)
        output = self.linear(output)
        return output
    



