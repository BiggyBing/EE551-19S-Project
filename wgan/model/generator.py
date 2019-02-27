#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 22:52:35 2019

@author: bingyangwen
"""

import torch
import torch.nn as nn
from options import GanConfiguration


class Generator(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, config: GanConfiguration):
        super(Generator, self).__init__()
        
        self.DIM = config.DIM
        
        self.preprocess_1 = nn.Linear(config.noise_length, 4 * 4 * 4 * self.DIM)
        self.preprocess_2 =nn.BatchNorm1d(4 * 4 * 4 * self.DIM)
        self.preprocess_3 =nn.ReLU()
       

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.DIM, 2 * self.DIM, 2, stride=2),
            nn.BatchNorm2d(2 * self.DIM),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.DIM, self.DIM, 2, stride=2),
            nn.BatchNorm2d(self.DIM),
            nn.ReLU()
        )
        
        self.output = nn.Sequential(
                nn.ConvTranspose2d(self.DIM, 3, 2, stride=2),
                nn.Tanh()
        )
        


    def forward(self, noise):

        output = self.preprocess_1(noise)
        output = self.preprocess_2(output)
        output = self.preprocess_3(output)
        output = output.view(-1, 4 * self.DIM, 4, 4)
        output = self.block1(output)
        output = self.block2(output)
        output = self.output(output)
        return output.view(-1, 3, 32, 32)
    
    